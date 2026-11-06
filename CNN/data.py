import os
import urllib.request
import numpy as np

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist.npz")


def download(path=PATH, url=URL):
    if os.path.exists(path):
        return path
    print(f"downloading: {url}")
    urllib.request.urlretrieve(url, path)
    print(f"saved to {path}")
    return path


def load(split="train", path=PATH):
    download(path)
    d = np.load(path)
    x = d[f"x_{split}"].astype(np.float32) / 255.0
    y = d[f"y_{split}"].astype(np.int64)
    return x, y


def flatten(x):
    return x.reshape(x.shape[0], -1)


def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def batch_iter(x, y, batch_size, seed=None):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x.shape[0])
    for i in range(0, idx.size, batch_size):
        b = idx[i:i + batch_size]
        yield x[b], y[b]


if __name__ == "__main__":
    xtr, ytr = load("train")
    xte, yte = load("test")
    print("x_train", xtr.shape, xtr.dtype, xtr.min(), xtr.max())
    print("y_train", ytr.shape, ytr.dtype, np.bincount(ytr))
    print("x_test ", xte.shape, "y_test", yte.shape)

    xtr_flat = flatten(xtr)
    print("flatten:", xtr.shape, "->", xtr_flat.shape)

    print("one_hot sample:\n", one_hot(ytr[:3]))

    for xb, yb in batch_iter(xtr_flat, ytr, batch_size=128, seed=0):
        print("one batch:", xb.shape, yb.shape, "mean=%.3f" % xb.mean())
        break
