import numpy as np

from CNN.conv1 import conv2d_naive


def conv2d(x:np.ndarray, W:np.ndarray, b:np.ndarray, stride=1, padding=0):
    C_in, H, Wd = x.shape
    C_out, C_in2, KH, KW = W.shape
    if padding>0:
        x = np.pad(x, ((0,0), (padding,padding), (padding,padding)))
    Hp, Wp = x.shape[1], x.shape[2]
    out_h = (Hp-KH)//stride +1
    out_w = (Wp-KW)//stride +1
    out:np.ndarray = np.zeros((C_out, out_h, out_w), dtype=np.float32)
    for co in range(C_out):
        for i in range(out_h):
            for j in range(out_w):
                out[co,i,j] = x[:, i*stride:i*stride+KH, j*stride:j*stride+KW].ravel() @ W[co].ravel() + b[co]

    return out

