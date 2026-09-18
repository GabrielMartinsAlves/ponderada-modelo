"""
train_model.py — Treina o detector de bem-te-vi e exporta os artefatos.

Fluxo:
  1) Carrega o dataset: real (real_features.npz) com "python train_model.py real",
     ou sintetico (generate_dataset.py) por padrao.
  2) Treina um MLP 28->32->16->1 (ReLU/sigmoide), pequeno e portavel p/ o ESP32.
  3) Escreve ../include/model_params.h (pesos p/ inferência no firmware) e
     params.npz (usado pelo exportador ONNX).
  4) Chama export_onnx.py em um PROCESSO SEPARADO para gerar
     bemtevi_detector.onnx.

  * Por que processo separado?  No Windows, carregar sklearn (MKL/OpenMP) e
    onnx (protobuf/abseil) no MESMO processo causa conflito de DLL e crash.
    Separar treino e exportação resolve de forma limpa.

A padronização z=(x-mean)/scale é embutida tanto no header C quanto no ONNX,
então a entrada do modelo é sempre o vetor de 28 features cru.
"""
import os
import sys
import subprocess
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# Arquitetura do MLP (pequena, portável para o ESP32): 28 -> 32 -> 16 -> 1
HIDDEN = (32, 16)

import features as F
from generate_dataset import build_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER_PATH = os.path.join(HERE, "..", "include", "model_params.h")
PARAMS_PATH = os.path.join(HERE, "params.npz")
METRICS_PATH = os.path.join(HERE, "metrics.txt")
EXPORT_SCRIPT = os.path.join(HERE, "export_onnx.py")


def extract_matrix(X_windows):
    return np.array([F.extract(w) for w in X_windows], dtype=np.float64)


def _arr(a):
    a = np.asarray(a).ravel()
    return ",\n    ".join(", ".join(f"{v:.8e}f" for v in a[i:i + 4])
                          for i in range(0, len(a), 4))


def write_header(mean, scale, Ws, bs, threshold, metrics, source):
    """Ws/bs = listas de 3 pesos/vieses do MLP (28->32->16->1)."""
    n = F.N_FEATURES
    h0, h1 = HIDDEN
    txt = f"""/*
 * model_params.h  —  GERADO por model/train_model.py  (NAO editar a mao)
 *
 * Detector acustico de BEM-TE-VI (Pitangus sulphuratus).
 * Fonte de treino: {source}
 * Modelo: MLP {n}->{h0}->{h1}->1 (ReLU nas ocultas, sigmoide na saida).
 * Normalizacao embutida:  z = (x - MODEL_MEAN) / MODEL_SCALE
 * Forward:  h0=relu(z*W0+B0); h1=relu(h0*W1+B1); p=sigmoid(h1*W2+B2)
 * Anomalia (bem-te-vi) se p > MODEL_THRESHOLD.
 * Pesos em ordem row-major: W0[i*{h0}+j] = entrada i -> oculta0 j, etc.
 *
 * Metricas (teste): accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}
 */
#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#define MODEL_N_FEATURES {n}
#define MODEL_IN  {n}
#define MODEL_H0  {h0}
#define MODEL_H1  {h1}
#define MODEL_THRESHOLD {threshold:.8e}f

static const float MODEL_MEAN[{n}] = {{
    {_arr(mean)}
}};

static const float MODEL_SCALE[{n}] = {{
    {_arr(scale)}
}};

static const float MODEL_W0[{n} * {h0}] = {{
    {_arr(Ws[0])}
}};
static const float MODEL_B0[{h0}] = {{
    {_arr(bs[0])}
}};

static const float MODEL_W1[{h0} * {h1}] = {{
    {_arr(Ws[1])}
}};
static const float MODEL_B1[{h1}] = {{
    {_arr(bs[1])}
}};

static const float MODEL_W2[{h1}] = {{
    {_arr(Ws[2])}
}};
#define MODEL_B2 ({float(np.asarray(bs[2]).ravel()[0]):.8e}f)

#endif // MODEL_PARAMS_H
"""
    os.makedirs(os.path.dirname(HEADER_PATH), exist_ok=True)
    with open(HEADER_PATH, "w", encoding="utf-8") as f:
        f.write(txt)


def load_dataset(source):
    """Retorna (X, y) de features. source: 'synthetic' ou 'real'."""
    if source == "real":
        path = os.path.join(HERE, "real_features.npz")
        if not os.path.exists(path):
            raise SystemExit("real_features.npz nao encontrado. Rode "
                             "fetch_xenocanto.py e build_real_dataset.py antes.")
        d = np.load(path)
        X, y = d["X"], d["y"]
        print(f"== Dataset REAL (xeno-canto): {len(y)} janelas "
              f"(pos={int(y.sum())}, neg={int((1 - y).sum())}) ==")
        return X, y
    print("== Gerando dataset SINTETICO de bem-te-vi ==")
    Xw, y = build_dataset(n_pos=500, n_neg=500)
    print(f"  janelas: {len(y)}  (pos={int(y.sum())}, neg={int((1 - y).sum())})")
    print("== Extraindo features (28 por janela) ==")
    return extract_matrix(Xw), y


def main():
    source = "real" if (len(sys.argv) > 1 and sys.argv[1] == "real") else "synthetic"
    X, y = load_dataset(source)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    mean = Xtr.mean(axis=0)
    scale = Xtr.std(axis=0)
    scale[scale < 1e-8] = 1e-8

    Ztr = (Xtr - mean) / scale
    Zte = (Xte - mean) / scale

    print(f"== Treinando MLP {F.N_FEATURES}->{HIDDEN[0]}->{HIDDEN[1]}->1 ==")
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, activation="relu",
                        max_iter=1500, random_state=0, early_stopping=True,
                        n_iter_no_change=25)
    clf.fit(Ztr, ytr)

    # pesos: coefs_ = [W0(28,32), W1(32,16), W2(16,1)]; intercepts_ = [b0,b1,b2]
    Ws = [c.astype(np.float64) for c in clf.coefs_]
    bs = [b.astype(np.float64) for b in clf.intercepts_]
    threshold = 0.5

    yp = (clf.predict_proba(Zte)[:, 1] > threshold).astype(int)
    metrics = dict(
        accuracy=accuracy_score(yte, yp),
        precision=precision_score(yte, yp, zero_division=0),
        recall=recall_score(yte, yp, zero_division=0),
        f1=f1_score(yte, yp, zero_division=0),
    )
    cm = confusion_matrix(yte, yp)
    print("== Metricas (teste) ==")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"  confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")

    print(f"== Escrevendo header C -> {HEADER_PATH} ==")
    write_header(mean, scale, Ws, bs, threshold, metrics, source)

    np.savez(PARAMS_PATH, mean=mean, scale=scale, threshold=threshold,
             n_features=F.N_FEATURES,
             W0=Ws[0], b0=bs[0], W1=Ws[1], b1=bs[1], W2=Ws[2], b2=bs[2])

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("Detector de bem-te-vi - metricas de teste\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"confusion_matrix [[TN,FP],[FN,TP]]:\n{cm}\n")

    print("== Exportando ONNX (processo separado p/ evitar conflito de DLL) ==")
    r = subprocess.run([sys.executable, EXPORT_SCRIPT])
    if r.returncode != 0:
        print("  AVISO: exportacao ONNX falhou (veja mensagens acima).")
    print("== Concluido ==")


if __name__ == "__main__":
    main()
