import numpy as np

def maxpool_forward(x:np.ndarray, k=2):
    C, H, W = x.shape
    oh, ow = H//k, W//k
    xr = x[:, :oh*k, :ow*k].reshape(C, oh, k, ow, k)
    out = xr.max(axis=(2,4))
    mask = (xr==out[:,:,None,:,None]).astype(x.dtype)
    return out, mask, (x.shape, k, oh, ow)

def maxpool_backward(dout:np.ndarray, mask:np.ndarray, cache):
    x_shape, k, oh, ow = cache
    C, H, W = x_shape

    dup = np.repeat(np.repeat(dout, k, axis=1), k, axis=2)
    dup = dup.reshape(C, oh, k, ow, k)
    dx = dup*mask
    dx = dx.reshape(C, oh*k, ow*k)
    if(H, W) != (oh*k, ow*k):
        full = np.zeros((C, H, W), dtype=dout.dtype)
        full[:, :oh*k, :ow*k] = dx
        return full
    return dx


def avgpool_forward(x:np.ndarray, k=2):
    C, H, W = x.shape
    oh, ow = H//k, W//k
    xr = x[:, :oh*k, :ow*k].reshape(C, oh, k, ow, k)
    return xr.mean(axis=(2,4)), (x.shape, k, oh, ow)


def avgpool_backward(dout:np.ndarray, cache):
    x_shape, k, oh, ow = cache
    C, H, W = x_shape
    dup = np.repeat(np.repeat(dout/(k*k), k, axis=1), k, axis=2)
    dup = dup.reshape(C, oh*k, ow*k)
    if(H, W)!=(oh*k, ow*k):
        full = np.zeros((C, H, W), dtype=dout.dtype)
        full[:, :oh*k, :ow*k] = dup
        return full
    return dup