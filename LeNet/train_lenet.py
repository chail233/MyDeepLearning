import time
import numpy as np

from data import load, batch_iter
from conv4 import conv2d_forward, conv2d_backward
from pool import maxpool_forward, maxpool_backward

C1, C2 = 8, 16          # 两个卷积层的输出通道数
KS = 3                  # 卷积核边长
POOL = 2                # 池化窗口
H_OUT = 7               # 28 -> 14 -> 7
FLAT = C2 * H_OUT * H_OUT       # 16*7*7 = 784


# ============================ 基础组件 ============================
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def init_params(seed=0, dtype=np.float32):
    """He 初始化。fan_in = C_in * KH * KW"""
    rng = np.random.default_rng(seed)

    def he(shape, fan_in):
        return rng.normal(0.0, np.sqrt(2.0 / fan_in), shape).astype(dtype)

    return [
        he((C1, 1, KS, KS), 1 * KS * KS),        # W1  fan_in = 9    std 0.471
        np.zeros(C1, dtype),                      # b1
        he((C2, C1, KS, KS), C1 * KS * KS),      # W2  fan_in = 72   std 0.167
        np.zeros(C2, dtype),                      # b2
        he((FLAT, 10), FLAT),                     # W3  fan_in = 784  std 0.0505
        np.zeros(10, dtype),                      # b3
    ]


# ============================ 前向 ============================
def forward(x, params):
    """x: (1, 28, 28)  ->  P: (10,), cache"""
    W1, b1, W2, b2, W3, b3 = params

    z1, c1 = conv2d_forward(x,  W1, b1, stride=1, padding=1)   # (8, 28, 28)
    a1 = np.maximum(z1, 0.0)                                    # (8, 28, 28)
    p1, m1, pm1 = maxpool_forward(a1, POOL)                     # (8, 14, 14)

    z2, c2 = conv2d_forward(p1, W2, b2, stride=1, padding=1)   # (16, 14, 14)
    a2 = np.maximum(z2, 0.0)                                    # (16, 14, 14)
    p2, m2, pm2 = maxpool_forward(a2, POOL)                     # (16, 7, 7)

    f = p2.reshape(-1)                                          # (784,)
    z3 = f @ W3 + b3                                            # (10,)
    P = softmax(z3)                                             # (10,)

    cache = dict(z1=z1, p1=p1, c1=c1, m1=m1, pm1=pm1,
                 z2=z2, p2=p2, c2=c2, m2=m2, pm2=pm2,
                 f=f)
    return P, cache


# ============================ 反向 ============================
def backward(P, y, cache, params):
    """单样本反向。返回 [dW1, db1, dW2, db2, dW3, db3]"""
    W1, b1, W2, b2, W3, b3 = params

    # ── 第 3 层（FC + softmax + CE，和项目0 完全一样）──
    dz3 = P.copy()
    dz3[y] -= 1.0                                               # (10,)
    dW3 = np.outer(cache["f"], dz3)                             # (784, 10)
    db3 = dz3.copy()                                            # (10,)
    df = W3 @ dz3                                               # (784,)

    # ── 第 2 层：池化 -> ReLU -> 卷积 ──
    dp2 = df.reshape(C2, H_OUT, H_OUT)                          # (16, 7, 7)
    da2 = maxpool_backward(dp2, cache["m2"], cache["pm2"])      # (16, 14, 14)
    dz2 = da2 * (cache["z2"] > 0)                               # (16, 14, 14)

    dp1, dW2, db2 = conv2d_backward(dz2, cache["c2"])           # (8,14,14) (16,8,3,3) (16,)

    # ── 第 1 层：池化 -> ReLU -> 卷积 ──
    da1 = maxpool_backward(dp1, cache["m1"], cache["pm1"])      # (8, 28, 28)
    dz1 = da1 * (cache["z1"] > 0)                               # (8, 28, 28)
    dx, dW1, db1 = conv2d_backward(dz1, cache["c1"])            # (1,28,28) (8,1,3,3) (8,)

    return [dW1, db1, dW2, db2, dW3, db3]


# ============================ 评估 ============================
def evaluate(X, Y, params, max_n=None):
    """返回 (平均 loss, 准确率)。X: (N,1,28,28)"""
    n = X.shape[0] if max_n is None else min(max_n, X.shape[0])
    loss_sum, correct = 0.0, 0
    for i in range(n):
        P, _ = forward(X[i], params)
        loss_sum += -np.log(P[Y[i]] + 1e-12)
        correct += int(P.argmax() == Y[i])
    return loss_sum / n, correct / n


# ============================ 验收 1 / 2：形状链 ============================
def check_shapes():
    print("\n" + "=" * 60)
    print("验收 1 / 2：形状链")
    print("=" * 60)

    params = init_params(seed=0)
    x = np.random.randn(1, 28, 28).astype(np.float32)

    P, cache = forward(x, params)
    assert P.shape == (10,), P.shape
    assert np.isclose(P.sum(), 1.0), P.sum()
    assert cache["z1"].shape == (C1, 28, 28), cache["z1"].shape
    assert cache["p1"].shape == (C1, 14, 14), cache["p1"].shape
    assert cache["z2"].shape == (C2, 14, 14), cache["z2"].shape
    assert cache["p2"].shape == (C2, 7, 7), cache["p2"].shape
    assert cache["f"].shape == (FLAT,), cache["f"].shape
    print("  前向：x(1,28,28) -> z1(8,28,28) -> p1(8,14,14) -> "
          "z2(16,14,14) -> p2(16,7,7) -> f(784) -> P(10)   OK")

    grads = backward(P, 3, cache, params)
    assert grads[0].shape == (C1, 1, KS, KS), grads[0].shape
    assert grads[2].shape == (C2, C1, KS, KS), grads[2].shape
    assert grads[4].shape == (FLAT, 10), grads[4].shape
    print("  反向：dW1(8,1,3,3) dW2(16,8,3,3) dW3(784,10)      OK")
    print("验收 1 / 2 通过")


# ============================ 验收 3：梯度检验 ============================
def check_grads(seed=0, n_sample=8, eps=1e-5):
    print("\n" + "=" * 60)
    print("验收 3：单样本梯度检验（float64）")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (1, 28, 28)).astype(np.float64)
    y = 3
    params = init_params(seed=1, dtype=np.float64)

    def loss_of():
        P, _ = forward(x, params)
        return -np.log(P[y] + 1e-12)

    P, cache = forward(x, params)
    grads = backward(P, y, cache, params)

    names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    worst = 0.0
    for name, p, g in zip(names, params, grads):
        flat = p.reshape(-1)
        assert np.shares_memory(flat, p), "参数不连续，原地修改无效"

        ks = rng.choice(flat.size, min(n_sample, flat.size), replace=False)
        num = np.empty(len(ks))
        for t, k in enumerate(ks):
            old = flat[k]
            flat[k] = old + eps; fp = loss_of()
            flat[k] = old - eps; fm = loss_of()
            flat[k] = old
            num[t] = (fp - fm) / (2 * eps)

        ana = g.reshape(-1)[ks]
        err = np.abs(ana - num).max()
        rel = err / (np.abs(num).max() + 1e-12)
        worst = max(worst, rel)
        print(f"  {name:3s}: 采样 {len(ks):2d} 点 | 相对误差 {rel:.3e}")

    assert worst < 1e-7, f"梯度检验失败, 最大相对误差 {worst:.3e}"
    print(f"验收 3 通过 —— 最差相对误差 {worst:.3e}")


# ============================ 验收 4：单样本过拟合 ============================
def check_overfit(steps=300, lr=0.1):
    print("\n" + "=" * 60)
    print("验收 4：单样本过拟合（最强的端到端检查）")
    print("=" * 60)

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (1, 28, 28)).astype(np.float32)
    y = 5
    params = init_params(seed=2)

    for s in range(1, steps + 1):
        P, cache = forward(x, params)
        loss = -np.log(P[y] + 1e-12)
        grads = backward(P, y, cache, params)
        for p, g in zip(params, grads):
            p -= lr * g                     # 单样本：不除以 B

        if s % 50 == 0 or s == 1:
            print(f"  step {s:3d} | loss {loss:.6f} | P[y] {P[y]:.6f}")

    assert loss < 0.01, f"过拟合失败, loss={loss:.4f}（反向链路可能有 bug）"
    print(f"验收 4 通过 —— final loss {loss:.6f}, P[y] {P[y]:.6f}")


# ============================ 验收 5：正式训练 ============================
def train(Xtr, ytr, Xte, yte, epochs=5, B=32, lr=0.05, seed=0):
    print("\n" + "=" * 60)
    print(f"验收 5：正式训练  N={Xtr.shape[0]}  B={B}  lr={lr}  epochs={epochs}")
    print("=" * 60)

    params = init_params(seed=seed)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        n_batch = (Xtr.shape[0] + B - 1) // B
        loss_sum, seen = 0.0, 0

        for xb, yb in batch_iter(Xtr, ytr, B, seed=epoch):
            m = xb.shape[0]
            acc = [np.zeros_like(p) for p in params]

            for j in range(m):
                P, cache = forward(xb[j], params)
                loss_sum += -np.log(P[yb[j]] + 1e-12)
                grads = backward(P, yb[j], cache, params)
                for a, g in zip(acc, grads):
                    a += g
            seen += m

            # 梯度是线性的 -> 累加后除以样本数 == 真批量梯度
            for p, a in zip(params, acc):
                p -= lr * a / m

        tr_loss, tr_acc = evaluate(Xtr, ytr, params, max_n=2000)
        te_loss, te_acc = evaluate(Xte, yte, params)
        print(f"epoch {epoch:2d} | train_loss {loss_sum/seen:.4f} "
              f"| tr {tr_acc:.4f} | te {te_acc:.4f} | te_loss {te_loss:.4f} "
              f"| {time.time()-t0:.1f}s")

    te_loss, te_acc = evaluate(Xte, yte, params)
    print(f"\n最终 test loss {te_loss:.4f} | test acc {te_acc:.4f}")
    return params, te_acc


# ============================ main ============================
if __name__ == "__main__":
    print("加载 MNIST ...")
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    Xtr = Xtr[:, None, :, :]        # (60000,1,28,28)   <- 加通道维，别忘
    Xte = Xte[:, None, :, :]
    print(f"  train {Xtr.shape}  test {Xte.shape}")

    check_shapes()          # 验收 1 / 2
    check_grads()           # 验收 3
    check_overfit()         # 验收 4

    # 先用子集跑通，确认能学
    N = 8000
    train(Xtr[:N], ytr[:N], Xte, yte, epochs=5, B=32, lr=0.05)

    # 想上 98%：把上面这行换成全量（会慢约 8 倍）
    # train(Xtr, ytr, Xte, yte, epochs=10, B=32, lr=0.05)
