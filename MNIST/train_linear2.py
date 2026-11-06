import numpy as np
from data import load, flatten, batch_iter

def softmax(Z: np.ndarray):
    Z = Z - Z.max(axis=1, keepdims=True)
    Z_e = np.exp(Z)
    return Z_e / Z_e.sum(axis=1, keepdims=True)


Xtr, Ytr = load("train")
Xte, Yte = load("test")
Xtr = flatten(Xtr)
Xte = flatten(Xte)

rng = np.random.default_rng(114514)
W1 = rng.normal(0.0, np.sqrt(2.0 / 784.0), size=(784, 128)).astype(np.float32)
W2 = rng.normal(0.0, np.sqrt(2.0 / 128.0), size=(128, 10)).astype(np.float32)
b1 = np.zeros(128, dtype=np.float32)
b2 = np.zeros(10, dtype=np.float32)
lr = 0.1

for epoch in range(1, 21):
    for X, Y in batch_iter(Xtr, Ytr, batch_size=100):
        # nn1 forward
        n = X.shape[0]
        Z1 = X @ W1 + b1
        A1 = np.maximum(Z1, 0)
        # nn2 forward
        Z2 = A1 @ W2 + b2
        P = softmax(Z2)

        # backward
        dZ2 = P.copy()
        dZ2[np.arange(n), Y] -= 1.0
        dZ2 /= n
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)

        W1 -= lr*dW1
        b1 -= lr*db1
        W2 -= lr*dW2
        b2 -= lr*db2

    A1_tr = np.maximum(Xtr @ W1 + b1, 0)
    P_tr = softmax(A1_tr @ W2 + b2)
    L_tr = -np.log(P_tr[np.arange(len(Ytr)), Ytr] + 1e-12).mean()
    acc_tr = (P_tr.argmax(axis=1) == Ytr).mean()

    A1_te = np.maximum(Xte @ W1 + b1, 0)
    P_te = softmax(A1_te @ W2 + b2)
    L_te = -np.log(P_te[np.arange(len(Yte)), Yte] + 1e-12).mean()
    acc_te = (P_te.argmax(axis=1) == Yte).mean()

    print(f"epoch {epoch:2d} | loss {L_tr:.4f}/{L_te:.4f} | "
          f"train {acc_tr:.4f} | test {acc_te:.4f}")
