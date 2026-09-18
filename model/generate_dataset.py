"""
generate_dataset.py — Gera janelas de áudio sintéticas rotuladas.

Positivos (label=1): cantos de BEM-TE-VI (Pitangus sulphuratus).
    O canto é o clássico assobio de três notas "bem-te-VI", estridente,
    com energia concentrada em ~1.5-4 kHz e a nota final ascendente.
Negativos (label=0): ruído branco/rosa, silêncio, outros assobios em
    frequências diferentes, "claps"/impactos de banda larga e formantes
    tipo voz humana. Serve para o modelo aprender a rejeitar o que NÃO é
    bem-te-vi.

Não depende de nenhum dataset externo: tudo é sintetizado, então o projeto
roda "out-of-the-box". Se você tiver gravações reais (.wav 16 kHz mono),
pode substituir/aumentar chamando load_real_wavs().
"""
import numpy as np
from features import SR, WINDOW_SAMPLES

RNG_DEFAULT = 1234


def _envelope(n, attack, release):
    """Envelope de amplitude simples (ataque/sustain/decaimento)."""
    env = np.ones(n)
    a = max(1, int(attack * n))
    r = max(1, int(release * n))
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.linspace(1, 0, r)
    return env


def _note(dur, f_start, f_end, sr=SR, harmonics=(1.0, 0.4, 0.15)):
    """Uma nota assobiada com glissando linear de f_start->f_end + harmônicos."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = np.linspace(f_start, f_end, n)
    phase = 2 * np.pi * np.cumsum(f) / sr
    sig = np.zeros(n)
    for h, amp in enumerate(harmonics, start=1):
        sig += amp * np.sin(h * phase)
    sig *= _envelope(n, 0.15, 0.25)
    return sig


def make_bemtevi(rng):
    """Sintetiza um canto de bem-te-vi dentro de uma janela WINDOW_SAMPLES."""
    win = np.zeros(WINDOW_SAMPLES)
    # variação aleatória de tom/ritmo entre indivíduos
    base = rng.uniform(0.9, 1.15)
    gap = rng.uniform(0.03, 0.08)
    # nota 1 "bem" — curta, levemente descendente
    n1 = _note(rng.uniform(0.10, 0.16), base * 3400, base * 2700)
    # nota 2 "te" — curtíssima, médio-agudo
    n2 = _note(rng.uniform(0.06, 0.10), base * 2900, base * 2600)
    # nota 3 "vi" — mais longa e ASCENDENTE (marca registrada)
    n3 = _note(rng.uniform(0.18, 0.28), base * 2600, base * 3900)

    pos = int(rng.uniform(0.05, 0.15) * WINDOW_SAMPLES)  # início dentro da janela
    for note in (n1, n2, n3):
        end = pos + len(note)
        if end > WINDOW_SAMPLES:
            note = note[: WINDOW_SAMPLES - pos]
            end = WINDOW_SAMPLES
        win[pos:end] += note[: end - pos]
        pos = end + int(gap * SR)
        if pos >= WINDOW_SAMPLES:
            break

    win /= (np.max(np.abs(win)) + 1e-9)
    win *= rng.uniform(0.35, 0.95)                   # volume variável
    # SNR variável — às vezes o canto fica bem imerso em ruído ambiente
    win += rng.uniform(0.02, 0.18) * _pink_noise(WINDOW_SAMPLES, rng)
    win += rng.uniform(0.01, 0.08) * rng.standard_normal(WINDOW_SAMPLES)
    return np.clip(win, -1.0, 1.0)


def _pink_noise(n, rng):
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.arange(spec.shape[0])
    freqs[0] = 1
    spec /= np.sqrt(freqs)
    out = np.fft.irfft(spec, n)
    return out / (np.max(np.abs(out)) + 1e-9)


def make_negative(rng):
    """Um exemplo negativo de tipo aleatório."""
    kind = rng.integers(0, 7)
    n = WINDOW_SAMPLES
    if kind == 0:                                   # ruído branco
        win = 0.3 * rng.standard_normal(n)
    elif kind == 1:                                 # ruído rosa (vento/ambiente)
        win = 0.6 * _pink_noise(n, rng)
    elif kind == 2:                                 # silêncio quase total
        win = 0.01 * rng.standard_normal(n)
    elif kind == 3:                                 # assobio em frequência "errada"
        f = rng.uniform(500, 1200)                  # grave -> não é bem-te-vi
        win = _note(rng.uniform(0.3, 0.6), f, f * rng.uniform(0.9, 1.1))
        win = np.pad(win, (0, max(0, n - len(win))))[:n]
        win += 0.05 * rng.standard_normal(n)
    elif kind == 4:                                 # tom ESTÁVEL dentro da banda (confunde)
        f = rng.uniform(2000, 3800)                 # mesma banda, mas sem o padrão 3-notas
        win = _note(rng.uniform(0.4, 0.7), f, f)
        win = np.pad(win, (0, max(0, n - len(win))))[:n]
        win += 0.06 * rng.standard_normal(n)
    elif kind == 5:                                 # sequência aleatória de notas (outro pássaro)
        win = np.zeros(n)
        pos = int(rng.uniform(0.02, 0.1) * n)
        for _ in range(rng.integers(2, 5)):
            f0 = rng.uniform(1800, 4500)
            note = _note(rng.uniform(0.05, 0.15), f0, f0 * rng.uniform(0.8, 1.2))
            end = min(pos + len(note), n)
            win[pos:end] += note[:end - pos]
            pos = end + int(rng.uniform(0.02, 0.12) * SR)
            if pos >= n:
                break
        win += 0.05 * rng.standard_normal(n)
    else:                                           # impacto/clap de banda larga
        win = np.zeros(n)
        for _ in range(rng.integers(1, 4)):
            p = rng.integers(0, n - 400)
            burst = rng.standard_normal(400) * np.exp(-np.linspace(0, 6, 400))
            win[p:p + 400] += burst
        win += 0.05 * rng.standard_normal(n)
    win /= (np.max(np.abs(win)) + 1e-9)
    win *= rng.uniform(0.3, 0.95)
    return np.clip(win, -1.0, 1.0)


def build_dataset(n_pos=500, n_neg=500, seed=RNG_DEFAULT):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_pos):
        X.append(make_bemtevi(rng))
        y.append(1)
    for _ in range(n_neg):
        X.append(make_negative(rng))
        y.append(0)
    return np.array(X), np.array(y, dtype=np.int64)
