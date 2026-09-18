"""
gen_sim_clip.py — Extrai um trecho REAL de bem-te-vi para o modo SIM do Wokwi.

O modelo agora é treinado em áudio real, então o sinal sintético não é
reconhecido. Aqui pegamos, entre as gravações positivas baixadas, a janela
(~0.78 s) que o modelo REAL melhor pontua, quantizamos para int16 e gravamos
em include/sim_clip.h. O firmware reproduz esse clipe em WOKWI_SIM, garantindo
que o LED acenda ao segurar o botão — com o mesmo modelo usado no hardware.
"""
import os
import glob
import numpy as np
import features as F
from build_real_dataset import decode

HERE = os.path.dirname(os.path.abspath(__file__))
POS = os.path.join(HERE, "..", "data", "pos")
OUT = os.path.join(HERE, "..", "include", "sim_clip.h")

d = np.load(os.path.join(HERE, "params.npz"))
mean, scale = d["mean"], d["scale"]
W0, b0, W1, b1, W2, b2 = d["W0"], d["b0"], d["W1"], d["b1"], d["W2"], d["b2"]


def prob(x):
    z = (x - mean) / scale
    h0 = np.maximum(z @ W0 + b0, 0)
    h1 = np.maximum(h0 @ W1 + b1, 0)
    return float(1 / (1 + np.exp(-(h1 @ W2 + b2)[0])))


def main():
    files = sorted(glob.glob(os.path.join(POS, "*.mp3")))[:60]
    best_p, best_w = -1.0, None
    for path in files:
        audio = decode(path)
        if audio is None or len(audio) < F.WINDOW_SAMPLES:
            continue
        hop = F.WINDOW_SAMPLES // 2
        for i in range(0, len(audio) - F.WINDOW_SAMPLES + 1, hop):
            w = audio[i:i + F.WINDOW_SAMPLES]
            if np.sqrt(np.mean(w * w)) < 0.02:
                continue
            p = prob(F.extract(w))
            if p > best_p:
                best_p, best_w = p, w.copy()
    if best_w is None:
        raise SystemExit("Nenhuma janela positiva encontrada em data/pos.")

    # normaliza para ~0.7 de pico e quantiza para int16
    peak = max(np.max(np.abs(best_w)), 1e-9)
    clip = np.clip(best_w / peak * 0.7, -1, 1)
    q = np.round(clip * 32767).astype(np.int16)

    # verifica que a versao quantizada ainda e reconhecida
    p_q = prob(F.extract(q.astype(np.float64) / 32768.0))
    print(f"melhor janela: p={best_p:.3f}  |  apos int16: p={p_q:.3f}  ({len(q)} amostras)")

    def fmt(a):
        return ",\n    ".join(", ".join(str(int(v)) for v in a[i:i + 16])
                              for i in range(0, len(a), 16))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("/*\n * sim_clip.h — GERADO por model/gen_sim_clip.py (NAO editar)\n"
                " * Trecho real de bem-te-vi (~0.78 s, 16 kHz, int16) para o modo\n"
                f" * WOKWI_SIM. Prob no modelo real: {p_q:.3f}.\n */\n")
        f.write("#ifndef SIM_CLIP_H\n#define SIM_CLIP_H\n#include <stdint.h>\n\n")
        f.write(f"#define SIM_CLIP_LEN {len(q)}\n\n")
        f.write(f"static const int16_t SIM_CLIP[SIM_CLIP_LEN] = {{\n    {fmt(q)}\n}};\n\n")
        f.write("#endif // SIM_CLIP_H\n")
    print(f"salvo -> {OUT}")


if __name__ == "__main__":
    main()
