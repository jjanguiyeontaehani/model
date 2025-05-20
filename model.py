import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    dim_model: int = 1024
    dim_ff: int = 1024
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.15
    batch_size: int = 32
    max_len: int = 1024
    learning_rate: float = 0.001
    epoch: int = 300
    pad_token_id: int = 0
    device: str = "cpu"

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

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.dim_model % config.num_heads == 0, "dim_model must be divisible by num_heads"
        
        self.num_heads = config.num_heads
        self.dim_model = config.dim_model
        self.dim_head = config.dim_model // config.num_heads
        self.scale = self.dim_head ** -0.5

        self.qkv_proj = nn.Linear(config.dim_model, config.dim_model * 3)
        self.out_proj = nn.Linear(config.dim_model, config.dim_model)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2), qkv)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
                mask = mask & mask.transpose(-2, -1)
            
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim_model)
        
        return self.out_proj(out)

class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_ff, config.dim_model)
        )
        self.norm1 = nn.LayerNorm(config.dim_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.dim_model, eps=1e-6)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, mask)
        x = x + self.dropout1(attn_out)
        
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout2(ff_out)
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = TokenEmbedding(config)
        self.position_encoding = SinusoidalPositionEncoding(config.max_len, config.dim_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.dim_model, eps=1e-6)
        self.output_projection = nn.Linear(config.dim_model, config.vocab_size)
        
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def create_padding_mask(self, src, pad_idx=None):
        if pad_idx is None:
            pad_idx = self.config.pad_token_id
        
        mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
        
    def forward(self, x, attention_mask=None):
        if attention_mask is None:
            attention_mask = self.create_padding_mask(x)
        
        x = self.token_embedding(x)
        pos_enc = self.position_encoding(x)
        x = x + pos_enc
        x = self.embedding_dropout(x)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        output = self.output_projection(x)
        
        return output

from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

def create_attention_mask(input_ids, pad_token_id=0):
    batch_size, seq_len = input_ids.size()
    mask = (input_ids != pad_token_id).float()
    attention_mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
    return attention_mask

class LazyTextIterableDataset(IterableDataset):
    def __init__(self, text_generator_fn, tokenizer):
        self.text_generator_fn = text_generator_fn
        self.tokenizer = tokenizer

    def __iter__(self):
        for text in self.text_generator_fn():
            token_ids = self.tokenizer.encode(text).ids
            yield torch.tensor(token_ids, dtype=torch.long)

def create_dataloader(fileName, tokenizer, config, shuffle=False):
    
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
        shuffle=shuffle,
        collate_fn=collate_fn,
    )



def train_model(model, dataloader, config, device="cpu"):
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_loss = float('inf')

    for epoch in range(config.epoch):
        epoch_loss = 0.0
        batch_count = 0
        optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)


            outputs = model(input_ids, attention_mask)
            loss = F.cross_entropy(
                outputs.view(-1, config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            print(f"Batch {i+1}, Loss: {loss.item():.3f}")

            loss.backward()
            epoch_loss += loss.item()
            batch_count += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()


        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{config.epoch}, Average Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)


        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"transformer_model_best.pth")

    print("Training completed!")
    return model

def validate_model(model, dataloader, config):
    model.eval()
    running_loss = 0.0

    running_loss = 0.0
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)

            for i in range(10):
                # print("Input IDs:", input_ids[0].tolist())
                print("input text:", tokenizer.decode(input_ids[0].tolist()))

                outputs = model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    outputs.view(-1, config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100
                )

                predict = torch.argmax(outputs, dim=-1)
                tokens = predict[0].tolist()

                # print("Predicted Tokens:", tokens)
                # print("Predicted Text:", tokenizer.decode(tokens))
                print("Loss:", loss.item())

                new_input = torch.cat(
                    (input_ids[:, :], torch.tensor([[tokens[0]]], dtype=input_ids.dtype)),
                    dim=1
                ).clone().detach()

                input_ids = new_input
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100
                attention_mask = create_attention_mask(input_ids, config.pad_token_id).to(config.device)

                predictions.append(tokens[0])
                running_loss += loss.item()

    print("Final prediction:", predictions)
    print("Total loss:", running_loss)


    with open("performance.txt", "w") as f:
        f.write("model config:\n")
        f.write(str(config))
        f.write("\nvalidation loss:\n")
        f.write(str(running_loss))
    print("Validation completed!")
    print("Performance saved to performance.txt")

import gc
from tokenizer import myTokenizer

if __name__ == "__main__":
    # filename = "test.txt"
    filename = "depression_dataset.txt"

    print("Start reading data")

    def setSeed(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setSeed(18)

    tokenizer = myTokenizer()
    config = ModelConfig()
    config.epoch = 10

    dataloader = create_dataloader(filename, tokenizer, config)

    print("Data loaded successfully")

    model = TransformerModel(config)
    
    model.to(config.device)
    # model = torch.compile(model)
    model.train()
    trained_model = train_model(model, dataloader, config)

    del dataloader
    gc.collect()

    model.eval()
    config.batch_size = 1
    print("Start validation")

    validation_dataloader = create_dataloader(filename, tokenizer, config)

    validate_model(model, validation_dataloader, config)