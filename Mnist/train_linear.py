import numpy as np
from data import load, flatten, batch_iter


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


Xtr, ytr = load("train")
Xte, yte = load("test")
Xtr, Xte = flatten(Xtr), flatten(Xte)

rng = np.random.default_rng(0)
W = rng.normal(0.0, np.sqrt(2.0 / 784.0), size=(784, 10)).astype(np.float32)
b = np.zeros(10, dtype=np.float32)

lr = 0.1
for epoch in range(1, 11):
    for xb, yb in batch_iter(Xtr, ytr, batch_size=128, seed=epoch):
        n = xb.shape[0]

        # forward
        P = softmax(xb @ W + b)

        # backward  ← 全部内容只有这 4 行
        dZ = P.copy()
        dZ[np.arange(n), yb] -= 1.0
        dZ /= n
        dW = xb.T @ dZ
        db = dZ.sum(axis=0)

        # update
        W -= lr * dW
        b -= lr * db

    Ptr, Pte = softmax(Xtr @ W + b), softmax(Xte @ W + b)
    loss = -np.log(Ptr[np.arange(len(ytr)), ytr] + 1e-12).mean()
    acc_tr = (Ptr.argmax(1) == ytr).mean()
    acc_te = (Pte.argmax(1) == yte).mean()
    print(f"epoch {epoch:2d} | loss {loss:.4f} | train {acc_tr:.4f} | test {acc_te:.4f}")
