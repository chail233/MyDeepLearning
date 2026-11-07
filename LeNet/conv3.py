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

def conv2d_im2col(x:np.ndarray, W:np.ndarray, b, stride=1, padding=0):
    C_out, C_in, KH, KW = W.shape
    cols,out_h,out_w = im2col(x, KH, KW, stride=stride, padding=padding)
    Wf = W.reshape(C_out, -1).T
    out = cols @ Wf + b
    out = out.reshape(out_h, out_w, C_out)
    out = out.transpose(2,0,1)
    return np.ascontiguousarray(out)
