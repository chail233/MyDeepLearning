import numpy as np

def conv2d_naive(x:np.ndarray, k:np.ndarray):
    H, W = x.shape
    KH,KW = k.shape
    out_h = H-KH+1
    out_w = W-KW+1
    out = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            out[i,j] = k.ravel() @ x[i:i+KH, j:j+KW].ravel()

    return out

