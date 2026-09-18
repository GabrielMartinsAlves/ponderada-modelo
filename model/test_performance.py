"""
test_performance.py — Simula anomalias (bem-te-vi) e mede desempenho.

Entregável exigido pela ponderada: "script que simula anomalias e mede
performance". Este script:

  1) Gera um fluxo (stream) de janelas rotuladas com sementes NOVAS
     (não vistas no treino).
  2) Roda o pipeline completo (extração de features + regressão logística
     usando os pesos exportados em params.npz).
  3) Mede a LATÊNCIA de cada etapa (extração e detecção) por janela.
  4) Reporta acurácia/precisão/recall/F1 e latências (média/p50/p95).
  5) Estima o throughput e compara com o tempo real (0.784 s/janela) para
     mostrar folga de processamento.

Não importa sklearn nem onnx (evita conflito de DLL no Windows); a inferência
é reproduzida com numpy a partir dos mesmos pesos que vão para o firmware.
"""
import os
import time
import numpy as np

import features as F
from generate_dataset import build_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "params.npz")


def load_model():
    if not os.path.exists(PARAMS_PATH):
        raise SystemExit("params.npz não encontrado. Rode train_model.py primeiro.")
    d = np.load(PARAMS_PATH)
    p = dict(mean=d["mean"], scale=d["scale"], threshold=float(d["threshold"]),
             W0=d["W0"], b0=d["b0"], W1=d["W1"], b1=d["b1"], W2=d["W2"], b2=d["b2"])
    return p


def infer(feat, p):
    z = (feat - p["mean"]) / p["scale"]
    h0 = np.maximum(z @ p["W0"] + p["b0"], 0.0)
    h1 = np.maximum(h0 @ p["W1"] + p["b1"], 0.0)
    return float(1.0 / (1.0 + np.exp(-(h1 @ p["W2"] + p["b2"])[0])))


def pct(a, p):
    return float(np.percentile(a, p))


def evaluate_quality(model):
    """Qualidade num held-out dos dados REAIS (se existir) ou sintetico."""
    real = os.path.join(HERE, "real_features.npz")
    if os.path.exists(real):
        d = np.load(real)
        X, y = d["X"], d["y"]
        rng = np.random.default_rng(123)
        idx = rng.permutation(len(y))[: max(1, len(y) // 4)]  # 25% held-out
        X, y = X[idx], y[idx]
        origem = "REAIS (xeno-canto, held-out 25%)"
    else:
        Xw, y = build_dataset(n_pos=150, n_neg=150, seed=9999)
        X = np.array([F.extract(w) for w in Xw])
        origem = "sinteticos"
    preds = np.array([1 if infer(f, model) > model["threshold"] else 0 for f in X])
    y = np.array(y)
    tp = int(np.sum((preds == 1) & (y == 1))); tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0))); fn = int(np.sum((preds == 0) & (y == 1)))
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"\n== Qualidade da deteccao (dados {origem}) ==")
    print(f"  amostras : {len(y)} (pos={int(y.sum())}, neg={int((1-y).sum())})")
    print(f"  accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")
    print(f"  confusion [[TN,FP],[FN,TP]] = [[{tn},{fp}],[{fn},{tp}]]")


def main(seed=9999):
    model = load_model()
    evaluate_quality(model)

    # Latencia: cronometra extracao de features + inferencia em janelas geradas
    # (o TEMPO independe da origem do modelo; serve para medir performance).
    print("\n== Medindo latencia (janelas geradas) ==")
    Xw, _ = build_dataset(n_pos=40, n_neg=40, seed=seed)
    feat_lat, det_lat = [], []
    for w in Xw:
        t0 = time.perf_counter()
        feat = F.extract(w)
        t1 = time.perf_counter()
        infer(feat, model)
        t2 = time.perf_counter()
        feat_lat.append((t1 - t0) * 1000.0)
        det_lat.append((t2 - t1) * 1000.0)
    feat_lat = np.array(feat_lat)
    det_lat = np.array(det_lat)

    print("== Latencia por janela (proxy em PC; ESP32 e mais lento) ==")
    print(f"  extracao features : media={feat_lat.mean():.2f} ms  "
          f"p50={pct(feat_lat,50):.2f}  p95={pct(feat_lat,95):.2f}")
    print(f"  deteccao (MLP)    : media={det_lat.mean():.3f} ms  "
          f"p50={pct(det_lat,50):.3f}  p95={pct(det_lat,95):.3f}")
    total = feat_lat + det_lat
    print(f"  total pipeline    : media={total.mean():.2f} ms  p95={pct(total,95):.2f}")

    win_ms = F.WINDOW_SAMPLES / F.SR * 1000.0
    print("\n== Throughput vs tempo real ==")
    print(f"  duracao da janela : {win_ms:.1f} ms de audio")
    print(f"  custo por janela  : {total.mean():.2f} ms de CPU")
    print(f"  fator tempo-real  : {win_ms/total.mean():.1f}x (>1 = processa mais rapido que grava)")


if __name__ == "__main__":
    main()
