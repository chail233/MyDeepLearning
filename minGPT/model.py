import torch
import torch.nn as nn
from data import vocab_size, get_batch
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, n_embd=128, block_size=256):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)

    def forward(self, idx: torch.Tensor):
        T = idx.shape[1]
        position = torch.arange(T)
        x = self.tok(idx) + self.pos(position)
        return x


class SelfAttention(nn.Module):
    def __init__(self, n_embd=128):
        super().__init__()
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        att:torch.Tensor = q @ k.transpose(-2, -1) / (C ** 0.5)
        mask = torch.tril(torch.ones(T, T))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = att.softmax(dim=-1)
        out = att @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd=128, n_head=4):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x:torch.Tensor):
        B, T, C = x.shape
        H, d = self.n_head, self.head_dim
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(B, T, H, d).transpose(1, 2)
        k = k.view(B, T, H, d).transpose(1, 2)
        v = v.view(B, T, H, d).transpose(1, 2)
        att:torch.Tensor = q @ k.transpose(-2, -1) / (d ** 0.5)
        mask = torch.tril(torch.ones(T, T))
        att = att.masked_fill(mask==0, float('-inf'))
        att = att.softmax(dim=-1)
        out = att @ v
        out = out.transpose(1,2).reshape(B, T, C)
        out = self.wo(out)
        return out


class MLP(nn.Module):
    def __init__(self, n_embd = 128):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4*n_embd)
        self.proj = nn.Linear(4*n_embd, n_embd)

    def forward(self, x:torch.Tensor):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd=128, n_head=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x:torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

