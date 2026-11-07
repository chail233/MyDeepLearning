"""
LeNet on MNIST —— NumPy 手写前向 + 反向（批量 im2col + 向量化 col2im）

网络：
    Conv(1->8,  3x3, p=1) + ReLU + MaxPool(2)      (1,28,28)  -> (8,14,14)
    Conv(8->16, 3x3, p=1) + ReLU + MaxPool(2)      (8,14,14)  -> (16,7,7)
    Flatten(784) + FC(784->10) + Softmax

参考结果：60000 张，12 epoch，约 4 分钟，test acc ≈ 0.9858
"""
import time
import numpy as np

from data import load, batch_iter

C1, C2 = 8, 16          # 两个卷积层的输出通道数
KS = 3                  # 卷积核边长
POOL = 2                # 池化窗口
H_OUT = 7               # 28 -> 14 -> 7
FLAT = C2 * H_OUT * H_OUT       # 784


# ==================================================================
#  im2col / col2im（批量 + 向量化）
# ==================================================================
def im2col_batch(x, KH, KW, stride=1, padding=0):
    """x: (B,C,H,W) -> cols (B*out_h*out_w, C*KH*KW), out_h, out_w"""
    B, C, H, W = x.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    Hp, Wp = x.shape[2], x.shape[3]
    out_h = (Hp - KH) // stride + 1
    out_w = (Wp - KW) // stride + 1

    # (B,C,Hp-KH+1,Wp-KW+1,KH,KW) —— 纯视图，零拷贝
    win = np.lib.stride_tricks.sliding_window_view(x, (KH, KW), axis=(2, 3))
    win = win[:, :, ::stride, ::stride]                 # (B, C, oh, ow, KH, KW)
    win = win.transpose(0, 2, 3, 1, 4, 5)               # (B, oh, ow, C, KH, KW)
    cols = win.reshape(B * out_h * out_w, C * KH * KW)   # <- 这里才真正复制
    return cols, out_h, out_w


def col2im_batch(dcols, x_shape, KH, KW, stride=1, padding=0):
    """dcols: (B*out_h*out_w, C*KH*KW)  ->  dx: (B,C,H,W)"""
    B, C, H, W = x_shape
    Hp, Wp = H + 2 * padding, W + 2 * padding
    out_h = (Hp - KH) // stride + 1
    out_w = (Wp - KW) // stride + 1

    d5 = dcols.reshape(B, out_h, out_w, C, KH, KW)
    dxp = np.zeros((B, C, Hp, Wp), dtype=dcols.dtype)

    # 只循环 KH*KW 次：固定 (u,v) 时，所有窗口的这个偏移位置
    # 正好落在目标图的 [u::stride, v::stride] 上（一个步长切片就能选中）
    for u in range(KH):
        for v in range(KW):
            ys = slice(u, u + (out_h - 1) * stride + 1, stride)
            xs = slice(v, v + (out_w - 1) * stride + 1, stride)
            dxp[:, :, ys, xs] += d5[:, :, :, :, u, v].transpose(0, 3, 1, 2)

    if padding > 0:
        return dxp[:, :, padding:padding + H, padding:padding + W]
    return dxp


# ==================================================================
#  卷积层
# ==================================================================
def conv2d_forward(x, W, b, stride=1, padding=0):
    """x: (B,C,H,W)  W: (C_out,C_in,KH,KW)  b: (C_out,)  -> out (B,C_out,oh,ow)"""
    B, C_in, H, Wd = x.shape
    C_out, C_in2, KH, KW = W.shape
    assert C_in == C_in2, f"通道不匹配: x={C_in}, W={C_in2}"

    cols, out_h, out_w = im2col_batch(x, KH, KW, stride, padding)
    Wf = W.reshape(C_out, -1).T                          # (K2, C_out)

    out = cols @ Wf + b                                   # (B*N, C_out)
    out = out.reshape(B, out_h, out_w, C_out).transpose(0, 3, 1, 2)

    cache = (cols, Wf, x.shape, W.shape, stride, padding, out_h, out_w)
    return np.ascontiguousarray(out), cache


def conv2d_backward(dout, cache):
    """-> (dx, dW, db)"""
    cols, Wf, x_shape, W_shape, stride, padding, out_h, out_w = cache
    B = x_shape[0]
    C_out, C_in, KH, KW = W_shape
    N = out_h * out_w

    # (B,C_out,oh,ow) -> (B*N, C_out)，行号 = b*N + i*out_w + j
    dout_flat = np.ascontiguousarray(dout) \
        .reshape(B, C_out, N).transpose(0, 2, 1).reshape(B * N, C_out)

    dWf = cols.T @ dout_flat                              # (K2, C_out)，自动对 B*N 求和
    db = dout_flat.sum(axis=0)                            # (C_out,)
    dcols = dout_flat @ Wf.T                              # (B*N, K2)

    dW = dWf.T.reshape(C_out, C_in, KH, KW)
    dx = col2im_batch(dcols, x_shape, KH, KW, stride, padding)
    return dx, dW, db


# ==================================================================
#  池化层（argmax 路由，严格保证 Σdx == Σdout）
# ==================================================================
def maxpool_forward(x, k=2):
    """x: (B,C,H,W) -> out (B,C,H//k,W//k), cache"""
    B, C, H, W = x.shape
    oh, ow = H // k, W // k
    xr = x[:, :, :oh * k, :ow * k].reshape(B, C, oh, k, ow, k)

    # (B,C,oh,k,ow,k) -> (B,C,oh,ow,k*k)
    xr2 = xr.transpose(0, 1, 2, 4, 3, 5).reshape(B, C, oh, ow, k * k)
    idx = xr2.argmax(axis=-1)                                   # (B,C,oh,ow)
    out = np.take_along_axis(xr2, idx[..., None], axis=-1)[..., 0]

    return out, (x.shape, k, oh, ow, idx)


def maxpool_backward(dout, cache):
    x_shape, k, oh, ow, idx = cache
    B, C, H, W = x_shape

    # 每个 dout 只回传给 argmax 那一个位置
    d = np.zeros((B, C, oh, ow, k * k), dtype=dout.dtype)
    np.put_along_axis(d, idx[..., None], dout[..., None], axis=-1)

    # (B,C,oh,ow,k,k) -> (B,C,oh,k,ow,k) -> (B,C,oh*k,ow*k)
    d = d.reshape(B, C, oh, ow, k, k).transpose(0, 1, 2, 4, 3, 5)
    dx = d.reshape(B, C, oh * k, ow * k)

    if (H, W) != (oh * k, ow * k):
        full = np.zeros((B, C, H, W), dtype=dout.dtype)
        full[:, :, :oh * k, :ow * k] = dx
        return full
    return dx


# ==================================================================
#  网络
# ==================================================================
def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def init_params(seed=0):
    """He 初始化，fan_in = C_in * KH * KW"""
    rng = np.random.default_rng(seed)

    def he(shape, fan_in):
        return rng.normal(0.0, np.sqrt(2.0 / fan_in), shape).astype(np.float32)

    return [
        he((C1, 1, KS, KS), 1 * KS * KS),        # W1  fan_in 9
        np.zeros(C1, np.float32),                 # b1
        he((C2, C1, KS, KS), C1 * KS * KS),      # W2  fan_in 72
        np.zeros(C2, np.float32),                 # b2
        he((FLAT, 10), FLAT),                     # W3  fan_in 784
        np.zeros(10, np.float32),                 # b3
    ]


def forward(x, params):
    """x: (B,1,28,28) -> P (B,10), cache"""
    W1, b1, W2, b2, W3, b3 = params

    z1, c1 = conv2d_forward(x, W1, b1, padding=1)     # (B,8,28,28)
    a1 = np.maximum(z1, 0.0)
    p1, pc1 = maxpool_forward(a1, POOL)               # (B,8,14,14)

    z2, c2 = conv2d_forward(p1, W2, b2, padding=1)    # (B,16,14,14)
    a2 = np.maximum(z2, 0.0)
    p2, pc2 = maxpool_forward(a2, POOL)               # (B,16,7,7)

    f = p2.reshape(x.shape[0], -1)                    # (B,784)
    z3 = f @ W3 + b3                                  # (B,10)
    P = softmax_rows(z3)

    return P, dict(z1=z1, p1=p1, c1=c1, pc1=pc1,
                   z2=z2, p2=p2, c2=c2, pc2=pc2, f=f)


def backward(P, y, cache, params):
    """y: (B,) 整数标签 -> [dW1, db1, dW2, db2, dW3, db3]"""
    W1, b1, W2, b2, W3, b3 = params
    B = P.shape[0]

    dZ = P.copy()
    dZ[np.arange(B), y] -= 1.0
    dZ /= B                                           # 批量：在 softmax 梯度这里除 B

    dW3 = cache["f"].T @ dZ                           # (784,10)
    db3 = dZ.sum(axis=0)                              # (10,)
    df = dZ @ W3.T                                    # (B,784)

    dp2 = df.reshape(B, C2, H_OUT, H_OUT)
    da2 = maxpool_backward(dp2, cache["pc2"])
    dz2 = da2 * (cache["z2"] > 0)

    dp1, dW2, db2 = conv2d_backward(dz2, cache["c2"])
    da1 = maxpool_backward(dp1, cache["pc1"])
    dz1 = da1 * (cache["z1"] > 0)

    _, dW1, db1 = conv2d_backward(dz1, cache["c1"])
    return [dW1, db1, dW2, db2, dW3, db3]


def evaluate(X, Y, params, B=256):
    """-> (平均 loss, 准确率)"""
    n = X.shape[0]
    loss_sum, correct = 0.0, 0
    for i in range(0, n, B):
        xb, yb = X[i:i + B], Y[i:i + B]
        P, _ = forward(xb, params)
        loss_sum += -np.log(P[np.arange(len(yb)), yb] + 1e-12).sum()
        correct += (P.argmax(axis=1) == yb).sum()
    return loss_sum / n, correct / n


# ==================================================================
#  训练
# ==================================================================
def train(Xtr, ytr, Xte, yte, epochs=12, B=64, lr=0.15,
          momentum=0.9, decay=0.85, seed=0):
    params = init_params(seed)
    vel = [np.zeros_like(p) for p in params]

    print(f"训练  N={Xtr.shape[0]}  B={B}  lr={lr}  momentum={momentum}  "
          f"decay={decay}  epochs={epochs}")
    print(f"总参数 {sum(p.size for p in params)}")

    best = 0.0
    for epoch in range(1, epochs + 1):
        lr_t = lr * (decay ** (epoch - 1))            # 逐 epoch 衰减
        t0 = time.perf_counter()
        loss_sum, seen = 0.0, 0

        for xb, yb in batch_iter(Xtr, ytr, B, seed=epoch):
            P, cache = forward(xb, params)
            loss_sum += -np.log(P[np.arange(len(yb)), yb] + 1e-12).sum()
            seen += len(yb)

            grads = backward(P, yb, cache, params)
            for p, g, v in zip(params, grads, vel):    # SGD + momentum
                v *= momentum
                v -= lr_t * g
                p += v

        dt = time.perf_counter() - t0
        tr_loss, tr_acc = evaluate(Xtr, ytr, params)
        te_loss, te_acc = evaluate(Xte, yte, params)
        best = max(best, te_acc)
        print(f"epoch {epoch:2d} | lr {lr_t:.4f} | loss {loss_sum/seen:.4f} | "
              f"tr {tr_acc:.4f} | te {te_acc:.4f} | te_loss {te_loss:.4f} | "
              f"{dt:5.1f}s")

    print(f"最佳 test acc = {best:.4f}")
    return params, best


if __name__ == "__main__":
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    Xtr = Xtr[:, None, :, :]        # (60000,1,28,28)  加通道维
    Xte = Xte[:, None, :, :]
    print(f"train {Xtr.shape}  test {Xte.shape}")

    train(Xtr, ytr, Xte, yte)
