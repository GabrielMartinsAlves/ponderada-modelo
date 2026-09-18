"""
build_real_dataset.py — Converte os MP3 baixados em features rotuladas.

Fluxo por arquivo:
  MP3 --ffmpeg--> PCM float 16 kHz mono --> janelas de WINDOW_SAMPLES
  --> descarta janelas quase-silenciosas --> extrai 28 features (features.py).

Positivos (data/pos) -> label 1 (bem-te-vi)
Negativos (data/neg) -> label 0 (outras aves / ruído)

Saída: model/real_features.npz  (X: [N,28], y: [N])
Requer ffmpeg no PATH (usado via subprocess; sem libs de áudio em Python).
"""
import os
import glob
import subprocess
import numpy as np

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "real_features.npz")

# quantas janelas no máximo por arquivo (evita 1 gravação longa dominar)
MAX_WIN_POS = 10
MAX_WIN_NEG = 5
MAX_SECONDS = 60          # decodifica no máx. 60 s por arquivo
RMS_SILENCE = 0.008       # janelas mais quietas que isto são descartadas
# Limpeza de rótulo: uma gravação de bem-te-vi tem trechos SEM o canto
# (silêncio, fundo). Só aceitamos janelas positivas com energia real na
# banda 1.5-4 kHz (feature band_ratio_mean, índice 5) acima deste limiar.
POS_MIN_BAND_RATIO = 0.15
BAND_RATIO_IDX = 5
HOP_POS = F.WINDOW_SAMPLES // 2   # 50% overlap -> mais positivos
HOP_NEG = F.WINDOW_SAMPLES        # sem overlap nos negativos


def decode(path, sr=F.SR, max_seconds=MAX_SECONDS):
    """MP3/qualquer -> np.float64 mono em [-1,1] via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-ac", "1", "-ar", str(sr), "-t", str(max_seconds),
           "-f", "f32le", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float64)


def windows_from(audio, hop, max_win):
    """Gera janelas com energia suficiente (pula silêncio)."""
    out = []
    n = len(audio)
    i = 0
    while i + F.WINDOW_SAMPLES <= n and len(out) < max_win:
        w = audio[i:i + F.WINDOW_SAMPLES]
        if np.sqrt(np.mean(w * w)) >= RMS_SILENCE:
            out.append(w)
        i += hop
    return out


def process_folder(folder, label, hop, max_win):
    files = sorted(glob.glob(os.path.join(folder, "*.mp3")))
    X = []
    dropped = 0
    print(f"  {folder}: {len(files)} arquivos")
    for k, path in enumerate(files):
        audio = decode(path)
        if audio is None or len(audio) < F.WINDOW_SAMPLES:
            continue
        for w in windows_from(audio, hop, max_win):
            feat = F.extract(w)
            # limpeza de positivos: exige energia real na banda do bem-te-vi
            if label == 1 and feat[BAND_RATIO_IDX] < POS_MIN_BAND_RATIO:
                dropped += 1
                continue
            X.append(feat)
        if (k + 1) % 25 == 0:
            print(f"    processados {k+1}/{len(files)} ({len(X)} janelas)")
    if label == 1:
        print(f"    (positivos descartados por baixa energia na banda: {dropped})")
    return X, [label] * len(X)


def main():
    pos_dir = os.path.join(DATA, "pos")
    neg_dir = os.path.join(DATA, "neg")
    if not (os.path.isdir(pos_dir) and os.path.isdir(neg_dir)):
        raise SystemExit("data/pos e data/neg nao encontrados. "
                         "Rode fetch_xenocanto.py primeiro.")

    print("== Extraindo features dos POSITIVOS (bem-te-vi) ==")
    Xp, yp = process_folder(pos_dir, 1, HOP_POS, MAX_WIN_POS)
    print("== Extraindo features dos NEGATIVOS ==")
    Xn, yn = process_folder(neg_dir, 0, HOP_NEG, MAX_WIN_NEG)

    X = np.array(Xp + Xn, dtype=np.float64)
    y = np.array(yp + yn, dtype=np.int64)
    if len(y) == 0:
        raise SystemExit("Nenhuma janela extraida. Verifique os audios/ffmpeg.")
    print(f"== Total: {len(y)} janelas (pos={int(y.sum())}, neg={int((1-y).sum())}) ==")

    np.savez(OUT, X=X, y=y)
    print(f"== Salvo -> {OUT} ==")
    print("Agora rode: python train_model.py real")


if __name__ == "__main__":
    main()
