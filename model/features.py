"""
features.py — Extração de features acústicas (espelho EXATO do firmware C).

Este arquivo define o "contrato" de features usado tanto no treino (Python)
quanto na inferência no ESP32 (audio_features.h). Qualquer alteração aqui
DEVE ser replicada no header C, senão o modelo pré-treinado não bate.

Pipeline (idêntico em Python e C):
  SR = 16 kHz, quadro = 512 amostras, hop = 256 (50% overlap), janela de Hann.
  Por quadro (14 features): RMS, Centroide Espectral, Rolloff(85%),
  Largura de Banda, ZCR, Razão de energia na banda 1.5-4 kHz e 8 MFCCs.
  Janela de ~0.78 s = 48 quadros; agrega por média e desvio-padrão -> 28 features.
"""
import numpy as np

# ---------------------------------------------------------------------------
# ESPECIFICAÇÃO DE FEATURES  (mantenha sincronizado com audio_features.h)
# ---------------------------------------------------------------------------
SR = 16000
N_FFT = 512
HOP = 256
N_FRAMES = 48
WINDOW_SAMPLES = (N_FRAMES - 1) * HOP + N_FFT  # 12544  (~0.784 s)
N_MEL = 20
N_MFCC = 8
FMIN = 300.0
FMAX = 7500.0
BAND_LO = 1500.0   # banda característica do bem-te-vi
BAND_HI = 4000.0
EPS = 1e-10

N_PER_FRAME = 6 + N_MFCC     # 14 features por quadro
N_FEATURES = 2 * N_PER_FRAME  # 28 (média + desvio de cada uma)

# Nomes (para relatório / depuração)
FRAME_FEATURE_NAMES = (
    ["rms", "centroid", "rolloff85", "bandwidth", "zcr", "band_ratio"]
    + [f"mfcc{i}" for i in range(N_MFCC)]
)
FEATURE_NAMES = (
    [f"{n}_mean" for n in FRAME_FEATURE_NAMES]
    + [f"{n}_std" for n in FRAME_FEATURE_NAMES]
)


# ---------------------------------------------------------------------------
# Constantes derivadas (calculadas uma vez)
# ---------------------------------------------------------------------------
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank():
    n_bins = N_FFT // 2 + 1
    m_lo, m_hi = _hz_to_mel(FMIN), _hz_to_mel(FMAX)
    m_pts = np.linspace(m_lo, m_hi, N_MEL + 2)
    f_pts = _mel_to_hz(m_pts)
    bin_freqs = np.arange(n_bins) * SR / N_FFT
    fb = np.zeros((N_MEL, n_bins), dtype=np.float64)
    for m in range(1, N_MEL + 1):
        f_left, f_c, f_right = f_pts[m - 1], f_pts[m], f_pts[m + 1]
        for k in range(n_bins):
            f = bin_freqs[k]
            if f_left <= f <= f_c and f_c > f_left:
                fb[m - 1, k] = (f - f_left) / (f_c - f_left)
            elif f_c < f <= f_right and f_right > f_c:
                fb[m - 1, k] = (f_right - f) / (f_right - f_c)
    return fb


N_BINS = N_FFT // 2 + 1
MEL_FB = _mel_filterbank()
# Janela de Hann "periódica" idêntica ao C: 0.5 - 0.5*cos(2*pi*n/(N-1))
HANN = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / (N_FFT - 1))
BIN_FREQS = np.arange(N_BINS) * SR / N_FFT

# Base DCT-II (N_MFCC x N_MEL), mesma fórmula usada no C
_n = np.arange(N_MFCC)[:, None]
_m = np.arange(N_MEL)[None, :]
DCT_BASIS = np.cos(np.pi * _n * (2.0 * _m + 1.0) / (2.0 * N_MEL))


def frame_features(frame):
    """14 features de um único quadro de N_FFT amostras (float em [-1,1])."""
    frame = np.asarray(frame, dtype=np.float64)
    rms = np.sqrt(np.mean(frame * frame) + EPS)

    signs = np.sign(frame)
    zcr = np.mean(np.abs(np.diff(signs)) > 0.0)  # fração de cruzamentos por zero

    w = frame * HANN
    spec = np.fft.rfft(w, N_FFT)
    power = spec.real ** 2 + spec.imag ** 2
    psum = power.sum() + EPS

    centroid = float(np.sum(BIN_FREQS * power) / psum)

    cum = np.cumsum(power)
    thr = 0.85 * cum[-1]
    idx = int(np.searchsorted(cum, thr))
    rolloff = float(BIN_FREQS[min(idx, N_BINS - 1)])

    bandwidth = float(np.sqrt(np.sum(power * (BIN_FREQS - centroid) ** 2) / psum))

    band_mask = (BIN_FREQS >= BAND_LO) & (BIN_FREQS <= BAND_HI)
    band_ratio = float(power[band_mask].sum() / psum)

    mel = MEL_FB @ power
    logmel = np.log(mel + EPS)
    mfcc = DCT_BASIS @ logmel

    out = np.empty(N_PER_FRAME, dtype=np.float64)
    out[0] = rms
    out[1] = centroid
    out[2] = rolloff
    out[3] = bandwidth
    out[4] = zcr
    out[5] = band_ratio
    out[6:] = mfcc
    return out


def extract(window):
    """Vetor de 28 features (média+desvio) de uma janela de WINDOW_SAMPLES."""
    window = np.asarray(window, dtype=np.float64)
    if window.shape[0] < WINDOW_SAMPLES:
        window = np.pad(window, (0, WINDOW_SAMPLES - window.shape[0]))
    feats = np.empty((N_FRAMES, N_PER_FRAME), dtype=np.float64)
    for i in range(N_FRAMES):
        s = i * HOP
        feats[i] = frame_features(window[s:s + N_FFT])
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    return np.concatenate([mean, std])
