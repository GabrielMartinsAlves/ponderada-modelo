"""
verify_c_port.py — Confere se a porta em C (audio_features.h) reproduz o
extrator de referência (features.py).

Reimplementa em Python o MESMO algoritmo do C (FFT radix-2 iterativa, mesmas
fórmulas de feature) e compara, janela a janela, com features.py (numpy).
Se a diferença máxima for pequena (~1e-4 relativo), a porta está correta.

Isto não compila o C (não há toolchain local), mas valida a lógica linha a
linha — que é onde moram os bugs de portabilidade.
"""
import numpy as np
import features as F
from generate_dataset import build_dataset

N_FFT = F.N_FFT

# ---- tabelas idênticas às de af_init() ----
hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / (N_FFT - 1))
bin_freqs = np.arange(F.N_BINS) * F.SR / N_FFT

logn = int(np.log2(N_FFT))
bitrev = np.array([int(f"{i:0{logn}b}"[::-1], 2) for i in range(N_FFT)])
tw_re = np.cos(-2 * np.pi * np.arange(N_FFT // 2) / N_FFT)
tw_im = np.sin(-2 * np.pi * np.arange(N_FFT // 2) / N_FFT)


def c_fft(re, im):
    """Radix-2 iterativa, exatamente como af_fft() no C."""
    re = re.copy(); im = im.copy()
    for i in range(N_FFT):
        j = bitrev[i]
        if j > i:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    length = 2
    while length <= N_FFT:
        half = length >> 1
        step = N_FFT // length
        for i in range(0, N_FFT, length):
            k = 0
            for j in range(half):
                wr, wi = tw_re[k], tw_im[k]
                a, b = i + j, i + j + half
                xr = re[b] * wr - im[b] * wi
                xi = re[b] * wi + im[b] * wr
                re[b] = re[a] - xr; im[b] = im[a] - xi
                re[a] += xr; im[a] += xi
                k += step
        length <<= 1
    return re, im


def c_frame_features(frame):
    frame = np.asarray(frame, dtype=np.float64)
    sumsq = np.sum(frame * frame)
    signs = np.sign(frame)
    zc = int(np.sum(signs[1:] != signs[:-1]))
    rms = np.sqrt(sumsq / N_FFT + F.EPS)
    zcr = zc / (N_FFT - 1)

    re = frame * hann
    im = np.zeros(N_FFT)
    re, im = c_fft(re, im)
    power = re[:F.N_BINS] ** 2 + im[:F.N_BINS] ** 2
    psum = F.EPS + power.sum()

    centroid = np.sum(bin_freqs * power) / psum
    band = power[(bin_freqs >= F.BAND_LO) & (bin_freqs <= F.BAND_HI)].sum()
    band_ratio = band / psum

    thr = 0.85 * (psum - F.EPS)
    cum = 0.0; roll_idx = F.N_BINS - 1
    for k in range(F.N_BINS):
        cum += power[k]
        if cum >= thr:
            roll_idx = k; break
    rolloff = bin_freqs[roll_idx]

    bw = np.sqrt(np.sum(power * (bin_freqs - centroid) ** 2) / psum)

    logmel = np.log((F.MEL_FB @ power) + F.EPS)
    mfcc = F.DCT_BASIS @ logmel

    return np.concatenate([[rms, centroid, rolloff, bw, zcr, band_ratio], mfcc])


def c_extract(window):
    window = np.asarray(window, dtype=np.float64)
    feats = np.array([c_frame_features(window[i * F.HOP:i * F.HOP + N_FFT])
                      for i in range(F.N_FRAMES)])
    return np.concatenate([feats.mean(axis=0), feats.std(axis=0)])


def main():
    Xw, _ = build_dataset(n_pos=6, n_neg=6, seed=7)
    max_abs = 0.0
    max_rel = 0.0
    for w in Xw:
        a = F.extract(w)       # referência numpy
        b = c_extract(w)       # mirror do C
        d = np.abs(a - b)
        denom = np.maximum(np.abs(a), 1e-6)
        max_abs = max(max_abs, float(d.max()))
        max_rel = max(max_rel, float((d / denom).max()))
    print(f"janelas testadas : {len(Xw)}")
    print(f"erro abs. maximo : {max_abs:.3e}")
    print(f"erro rel. maximo : {max_rel:.3e}")
    ok = max_rel < 1e-3
    print("RESULTADO        :", "OK (porta C valida)" if ok else "DIVERGENCIA!")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
