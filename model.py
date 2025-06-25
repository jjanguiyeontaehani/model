import math
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
    dropout: float = 0.1
    batch_size: int = 64
    max_len: int = 1024
    learning_rate: float = 0.0005
    reinit_percent: int = 10
    epoch: int = 200
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


def create_attention_mask(input_ids, pad_token_id=0):

    batch_size, seq_len = input_ids.size()
    padding_mask = (input_ids == pad_token_id).unsqueeze(1).unsqueeze(2)
 
    causal_mask = torch.triu(torch.full((seq_len, seq_len), True, dtype=torch.bool, device=input_ids.device), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    combined_mask = torch.logical_or(padding_mask, causal_mask)

    return combined_mask