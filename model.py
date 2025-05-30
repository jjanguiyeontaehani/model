import math
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
from dataclasses import dataclass

from tokenizer import myTokenizer

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    dim_model: int = 1024
    dim_ff: int = 1024
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    batch_size: int = 64
    max_len: int = 1024
    learning_rate: float = 0.0005
    reinit_percent: int = 10
    epoch: int = 400
    pad_token_id: int = 0
    device: str = "cpu"
    valid_pred_cnt: int = 1
    seed: int = 18

class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim_model)
        
    def forward(self, x):
        return self.embedding(x)

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, max_len: int, dim_model: int):
        super().__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2)
            * (-math.log(10000.0) / dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.dim() == 3:
            seq_len = x.size(1)
        else:
            seq_len = x.size(1)
        return self.pe[:seq_len].unsqueeze(0).expand(x.size(0), -1, -1)
    
def precompute_freqs_cis(max_len: int, dim_head: int, device: str = 'cpu') -> torch.Tensor:
    freqs = torch.arange(0, dim_head // 2, dtype=torch.float32, device=device)
    freqs = 1.0 / (10000 ** (freqs / (dim_head // 2)))
    freqs = freqs.unsqueeze(0).unsqueeze(0)

    positions = torch.arange(max_len, dtype=torch.float32, device=device).unsqueeze(1)
    theta = positions * freqs

    freqs_cis = torch.polar(torch.ones_like(theta), theta)

    return freqs_cis.to(device)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, num_heads, dim_head = x.size()

    freqs_cis_reshaped = freqs_cis[:, :seq_len, :].unsqueeze(2).expand(-1, -1, num_heads, -1)

    real = x[..., :dim_head // 2]
    imag = x[..., dim_head // 2:]

    x_complex = torch.view_as_complex(torch.stack((real, imag), dim=-1))

    x_rotated_complex = x_complex * freqs_cis_reshaped

    x_rotated_real = x_rotated_complex.real
    x_rotated_imag = x_rotated_complex.imag

    return torch.cat((x_rotated_real, x_rotated_imag), dim=-1)
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.dim_model % config.num_heads == 0, "dim_model must be divisible by num_heads"
        
        self.num_heads = config.num_heads
        self.dim_model = config.dim_model
        self.dim_head = config.dim_head
        self.scale = self.dim_head ** -0.5

        self.qkv_proj = nn.Linear(config.dim_model, config.dim_model * 3)
        self.out_proj = nn.Linear(config.dim_model, config.dim_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis, mask=None):
        batch_size, seq_len, _ = x.size()
        
        xqkv = self.qkv_proj(x).chunk(3, dim=-1)
        xq, xk, xv = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2), xqkv)

        q = apply_rotary_emb(xq, freqs_cis)
        k = apply_rotary_emb(xk, freqs_cis)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        out = torch.matmul(scores, xv)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim_model)
        
        return self.out_proj(out)

class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_ff, config.dim_model)
        )
        self.norm1 = RMSNorm(config.dim_model, eps=1e-6)
        self.norm2 = RMSNorm(config.dim_model, eps=1e-6)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis, mask=None):
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, freqs_cis=freqs_cis, mask=mask)
        x = x + self.dropout1(attn_out)
        
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout2(ff_out)
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.max_len = config.max_len
        self.dim_head = config.dim_head

        self.token_embedding = TokenEmbedding(config)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.dim_model, eps=1e-6)
        self.output_projection = nn.Linear(config.dim_model, config.vocab_size)
        
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
        
    def forward(self, x, freqs_cis=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = create_attention_mask(x, self.config.pad_token_id)
        if freqs_cis is None:
            freqs_cis = precompute_freqs_cis(self.max_len, self.dim_head, device=x.device)

        x = self.token_embedding(x)
        x = self.embedding_dropout(x)
        
        for layer in self.layers:
            x = layer(x, freqs_cis=freqs_cis, mask=attention_mask)

        x = self.norm(x)
        output = self.output_projection(x)
        
        return output
    
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


def create_attention_mask(input_ids, pad_token_id=0):

    batch_size, seq_len = input_ids.size()
    padding_mask = (input_ids == pad_token_id).unsqueeze(1).unsqueeze(2)
 
    causal_mask = torch.triu(torch.full((seq_len, seq_len), True, dtype=torch.bool, device=input_ids.device), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    combined_mask = torch.logical_or(padding_mask, causal_mask)

    return combined_mask

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


def train_model(model, dataloader, validation_dataloader, config):


    startTime = time.time()
    avg_loss = config.reinit_percent
    trainLossHistory = []
    validationLossHistory = []
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
            val_loss = validate_model(model, validation_dataloader, freqs_cis, config)
            print(f"Validation Loss after Epoch {epoch+1}: {val_loss:.6f}")
            model.train()
        else:
            val_loss = None
        validationLossHistory.append(val_loss)

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
    return model, trainLossHistory, validationLossHistory, elapsedTime

def validate_model(model, dataloader, freqs_cis, config):
    model.eval()
    running_loss = 0.0
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
                validated_cnt += 1

    running_loss /= validated_cnt
    return running_loss


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
    trained_model, trainLossHistory, validationLossHistory, elapsedTime = train_model(model, dataloader, validation_dataloader, config)


    with open("performance.txt", "a") as f:
        f.write("model config: ")
        f.write(str(config) + "\n")
        f.write("train loss history: " + str(trainLossHistory) + "\n")
        f.write("validation loss history: " + str(validationLossHistory) + "\n")
        f.write("best validation loss: " + str(min([loss for loss in validationLossHistory if loss is not None])) + "\n")
        f.write(f"training time: {elapsedTime:.2f} seconds\n")

    print("Performance saved to performance.txt")