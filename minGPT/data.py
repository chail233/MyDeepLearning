import os
import torch

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")
with open(file=PATH, mode="r", encoding="utf-8") as f:
    text =f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i : ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]

def decode(idx):
    return "".join([itos[i] for i in idx])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split, block_size=256, batch_size = 32):
    d = train_data if split=="train" else val_data
    ix =torch.randint(len(d)-block_size, (batch_size,))
    xs, ys = [], []
    for i in ix:
        xs.append(d[i:i+block_size])
        ys.append(d[i+1:i+1+block_size])
    return torch.stack(xs), torch.stack(ys)
