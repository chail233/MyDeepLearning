import numpy as np

def im2col(x:np.ndarray, KH,KW,stride=1, padding=0):
    C_in, H, W = x.shape
    if padding>0:
        x = np.pad(x, ((0,0), (padding,padding),(padding,padding)))
    Hp, Wp = x.shape[1],x.shape[2]
    out_h = (Hp-KH)//stride+1
    out_w = (Wp-KW)//stride+1
    win = np.lib.stride_tricks.sliding_window_view(x, (KH,KW), axis=(1,2))
    win = win[:, ::stride, ::stride]
    win = win.transpose(1,2,0,3,4)
    cols = win.reshape(out_h*out_w, C_in*KH*KW)
    return cols, out_h, out_w

def col2im(dcols:np.ndarray, x_shape, KH, KW, stride=1, padding=0):
    C_in, H, W = x_shape
    Hp, Wp = H + 2*padding, W + 2*padding
    out_h = (Hp-KH)//stride+1
    out_w = (Wp-KW)//stride+1

    dxp = np.zeros((C_in, Hp, Wp), dtype=dcols.dtype)

    for i in range(out_h):
        for j in range(out_w):
            n = i*out_w+j
            win = dcols[n].reshape(C_in, KH, KW)
            dxp[:, i*stride:i*stride+KH, j*stride:j*stride+KW] += win

    if padding>0:
        return dxp[:, padding:padding+H, padding:padding+W]
    else:
        return dxp

def conv2d_forward(x:np.ndarray, W:np.ndarray, b:np.ndarray, stride=1, padding=0):
    C_out, C_in, KH, KW = W.shape
    cols, out_h, out_w = im2col(x, KH, KW, stride, padding)
    Wf = W.reshape(C_out, -1).T
    out_flat = cols @ Wf + b
    out = np.ascontiguousarray(out_flat.T).reshape(C_out, out_h, out_w)
    cache = (cols, Wf, x.shape, W.shape, stride, padding, out_h, out_w)
    return out, cache

def conv2d_backward(dout:np.ndarray, cache):
    cols, Wf, x_shape, W_shape, stride, padding, out_h, out_w = cache
    C_out, C_in, KH, KW = W_shape
    N = out_h*out_w

    dout_flat = np.ascontiguousarray(np.ascontiguousarray(dout).reshape(C_out, N).T)

    dWf = cols.T @ dout_flat
    db = dout_flat.sum(axis=0)
    dcols = dout_flat @ Wf.T

    dW:np.ndarray = dWf.T.reshape(C_out, C_in, KH, KW)
    dx:np.ndarray = col2im(dcols, x_shape, KH, KW, stride, padding)
    return dx, dW, db

