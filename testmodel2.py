import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from tokenizer import myTokenizer #tokenizer.py


class Config:
    def __init__(self):
        self.vocab_size = 10000
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_act = "gelu"
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.initializer_range = 0.02
        self.layer_norm_eps = 1e-12