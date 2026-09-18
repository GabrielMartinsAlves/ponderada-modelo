"""Confere que o forward do MLP no firmware (run_model, row-major) == ONNX == numpy."""
import numpy as np
import features as F

d = np.load("params.npz")
mean, scale = d["mean"], d["scale"]
W0, b0, W1, b1, W2, b2 = d["W0"], d["b0"], d["W1"], d["b1"], d["W2"], d["b2"]
H0, H1 = W0.shape[1], W1.shape[1]
W0f, W1f, W2f = W0.ravel(), W1.ravel(), W2.ravel()
b2v = float(b2.ravel()[0])


def forward_numpy(x):
    z = (x - mean) / scale
    h0 = np.maximum(z @ W0 + b0, 0)
    h1 = np.maximum(h0 @ W1 + b1, 0)
    return float(1 / (1 + np.exp(-(h1 @ W2 + b2)[0])))


def forward_c_style(x):
    """Replica EXATAMENTE run_model() do main.cpp (loops + row-major)."""
    z = [(x[i] - mean[i]) / scale[i] for i in range(F.N_FEATURES)]
    h0 = []
    for j in range(H0):
        acc = b0[j]
        for i in range(F.N_FEATURES):
            acc += z[i] * W0f[i * H0 + j]
        h0.append(max(acc, 0.0))
    h1 = []
    for j in range(H1):
        acc = b1[j]
        for i in range(H0):
            acc += h0[i] * W1f[i * H1 + j]
        h1.append(max(acc, 0.0))
    out = b2v
    for i in range(H1):
        out += h1[i] * W2f[i]
    return float(1 / (1 + np.exp(-out)))


data = np.load("real_features.npz")
X, y = data["X"], data["y"]
idx = np.random.RandomState(1).choice(len(y), 10, replace=False)

sess = None
try:
    import onnxruntime as ort
    sess = ort.InferenceSession("bemtevi_detector.onnx", providers=["CPUExecutionProvider"])
except Exception as e:
    print("(onnxruntime indisponivel:", type(e).__name__, ")")

print("idx  label   numpy    c_style   onnx")
max_diff = 0.0
for i in idx:
    x = X[i]
    pn, pc = forward_numpy(x), forward_c_style(x)
    po = (float(sess.run(None, {"features": x[None].astype(np.float32)})[0].ravel()[0])
          if sess else float("nan"))
    max_diff = max(max_diff, abs(pn - pc), (abs(pn - po) if sess else 0.0))
    print(f"{i:4d}   {y[i]}     {pn:.4f}   {pc:.4f}   {po:.4f}")

print(f"\nmax |numpy - (c_style, onnx)| = {max_diff:.2e}")
probs = np.array([forward_numpy(x) for x in X])
print("accuracy (dataset todo):", round(float(((probs > 0.5).astype(int) == y).mean()), 4))
