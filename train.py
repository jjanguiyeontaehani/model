import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence

from model import TransformerModel, precompute_freqs_cis, create_attention_mask, ModelConfig
from tokenizer import myTokenizer

class LazyTextIterableDataset(IterableDataset):
    def __init__(self, text_generator_fn, tokenizer):
        self.text_generator_fn = text_generator_fn
        self.tokenizer = tokenizer

    def __iter__(self):
        for text in self.text_generator_fn():
            token_ids = self.tokenizer.encode(text).ids
            yield torch.tensor(token_ids, dtype=torch.long)

def create_dataloader(fileName, tokenizer, config):
    
    def yield_lines():
        with open(fileName, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    with open(fileName, "r", encoding="utf-8") as f:
        print('total lines:', sum(1 for line in f if line.strip()))

    dataset = LazyTextIterableDataset(yield_lines, tokenizer)

    def collate_fn(batch):
        input_ids = pad_sequence(batch, batch_first=True, padding_value=config.pad_token_id)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        attention_mask = create_attention_mask(input_ids, config.pad_token_id)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

@torch.no_grad()
def reinit_bottom_n_percent_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    n_percent: float = 10.0,
    init_fn_map: dict = None,
    param_filter: callable = None
):
    if init_fn_map is None:
        init_fn_map = {
            'default': lambda t: nn.init.kaiming_normal_(t, nonlinearity='relu'),
            'bias': lambda t: t.zero_()
        }
    if param_filter is None:
        param_filter = lambda name, t: True

    for name, param in model.named_parameters():
        if not param.requires_grad or not param_filter(name, param):
            continue

        tensor = param.data
        flat_abs = tensor.abs().flatten()

        if flat_abs.numel() == 0:
            continue

        threshold = torch.quantile(flat_abs, n_percent / 100.0)
        mask = tensor.abs() <= threshold
        if not mask.any():
            continue

        new_tensor = torch.empty_like(tensor)
        key = 'bias' if 'bias' in name else 'default'
        init_fn = init_fn_map.get(key, init_fn_map['default'])
        init_fn(new_tensor)

        tensor[mask] = new_tensor[mask]

        # if optimizer is not None:
        #     state = optimizer.state.get(param)
        #     if state is not None:
        #         state['exp_avg'][mask] = 0
        #         state['exp_avg_sq'][mask] = 0

def train_model(model, dataloader, validation_dataloader, config):
    startTime = time.time()
    avg_loss = config.reinit_percent
    trainLossHistory = []
    validationLossHistory = []
    validationAccuracyHistory = []
    limit = int(config.epoch * 0.9)
    step = 0
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    freqs_cis = precompute_freqs_cis(config.max_len, config.dim_head, device=config.device)

    for epoch in range(config.epoch):
        epochStartTime = time.time()
        epoch_loss = 0.0
        batch_count = 0
        optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)

            outputs = model(input_ids, freqs_cis=freqs_cis, attention_mask=attention_mask)
            loss = F.cross_entropy(
                outputs.view(-1, config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

            loss.backward()
            epoch_loss += loss.item()
            batch_count += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / batch_count
        scheduler.step(avg_loss)
        trainLossHistory.append(avg_loss)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == config.epoch:
            val_loss, val_acc = validate_model(model, validation_dataloader, freqs_cis, config)
            print(f"Validation Loss after Epoch {epoch+1}: {val_loss:.6f}, Validation Accuracy: {val_acc:.2f}")
            model.train()
        else:
            val_loss = None
            val_acc = None
        validationLossHistory.append(val_loss)
        validationAccuracyHistory.append(val_acc)

        reinit_epoch = step * 20
        if epoch >= reinit_epoch and reinit_epoch <= limit:
            current_rate = min(config.reinit_percent, avg_loss ** 2)
            if current_rate > 0:
                reinit_bottom_n_percent_model(
                    model,
                    optimizer=optimizer,
                    n_percent=current_rate,
                    init_fn_map={
                        'default': lambda t: init.kaiming_normal_(t, nonlinearity='relu'),
                        'bias':    lambda t: t.zero_()
                    },
                    param_filter=lambda name, t: all(
                        k not in name.lower() for k in ['norm', 'embedding']
                    )
                )
                print(f"Reinitialized {current_rate:.2f}% of model parameters")
            else:
                print("No parameters were reinitialized.")
            step += 1

        epochEndTime = time.time()
        print(f"Epoch {epoch+1}/{config.epoch}, train Loss: {avg_loss:.6f} completed in {epochEndTime - epochStartTime:.2f} seconds")

    elapsedTime = time.time() - startTime
    print(f"Training completed in {elapsedTime:.2f} seconds")
    return model, trainLossHistory, validationLossHistory, validationAccuracyHistory, elapsedTime

def validate_model(model, dataloader, freqs_cis, config):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    validated_cnt = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)

            for i in range(config.valid_pred_cnt):
                # print("input text:", tokenizer.decode(input_ids[0].tolist()))

                outputs = model(input_ids, freqs_cis=freqs_cis, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    outputs.view(-1, config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100
                )
                accuracy = (torch.argmax(outputs, dim=-1) == labels).float().mean().item()

                predict = torch.argmax(outputs, dim=-1)
                tokens = predict[0].tolist()

                # print("Predicted Text:", tokenizer.decode(tokens))
                # print("Loss:", loss.item())

                new_input = torch.cat(
                    (input_ids[:, :], torch.tensor([[tokens[0]]], dtype=input_ids.dtype)),
                    dim=1
                ).clone().detach()

                input_ids = new_input
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100
                attention_mask = create_attention_mask(input_ids, config.pad_token_id).to(config.device)

                running_loss += loss.item()
                running_accuracy += accuracy
                validated_cnt += 1

    running_loss /= validated_cnt
    running_accuracy = running_accuracy * 100 / validated_cnt
    return running_loss, running_accuracy

if __name__ == "__main__":
    start = time.time()
    trainfilename = "basic_math_dataset.txt"
    validationfilename = "basic_math_dataset2.txt"

    print("Start reading data")

    tokenizer = myTokenizer()
    config = ModelConfig()
    config.dim_head = config.dim_model // config.num_heads
    validation_config = ModelConfig()
    validation_config.batch_size = 1

    if config.seed:
        def setSeed(seed):
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        setSeed(config.seed)

    dataloader = create_dataloader(trainfilename, tokenizer, config)
    validation_dataloader = create_dataloader(validationfilename, tokenizer, validation_config)
    print("Data loaded successfully")

    model = TransformerModel(config)
    model.to(config.device)
    model.train()
    trained_model, trainLossHistory, validationLossHistory, validationAccuracyHistory, elapsedTime = train_model(model, dataloader, validation_dataloader, config)


    with open("performance.txt", "a") as f:
        f.write("model config: ")
        f.write(str(config) + "\n")
        f.write("train loss history: " + str(trainLossHistory) + "\n")
        f.write("validation loss history: " + str(validationLossHistory) + "\n")
        f.write("best validation loss: " + str(min([loss for loss in validationLossHistory if loss is not None])) + "\n")
        f.write("validation accuracy history: " + str(validationAccuracyHistory) + "\n")
        f.write("best validation accuracy: " + str(max([acc for acc in validationAccuracyHistory if acc is not None])) + "\n")
        f.write(f"training time: {elapsedTime:.2f} seconds\n")

    print("Performance saved to performance.txt")