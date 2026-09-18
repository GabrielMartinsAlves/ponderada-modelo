/*
 * Extracao de features acusticas no ESP32, espelho exato de model/features.py.
 * Mudar qualquer parametro aqui exige refletir em features.py e re-treinar,
 * senao os pesos de model_params.h deixam de valer.
 *
 * Por janela: 48 quadros de 512 amostras (50% overlap). Por quadro sao 14
 * features (RMS, centroide, rolloff85, largura de banda, ZCR, razao de energia
 * 1.5-4 kHz e 8 MFCCs), agregadas por media e desvio em 28 features.
 */
#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#include <math.h>
#include <string.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define AF_SR            16000
#define AF_N_FFT         512
#define AF_HOP           256
#define AF_N_FRAMES      48
#define AF_WINDOW_SAMPLES ((AF_N_FRAMES - 1) * AF_HOP + AF_N_FFT)  // 12544
#define AF_N_BINS        (AF_N_FFT / 2 + 1)                        // 257
#define AF_N_MEL         20
#define AF_N_MFCC        8
#define AF_FMIN          300.0f
#define AF_FMAX          7500.0f
#define AF_BAND_LO       1500.0f
#define AF_BAND_HI       4000.0f
#define AF_EPS           1e-10f
#define AF_N_PER_FRAME   (6 + AF_N_MFCC)   // 14
#define AF_N_FEATURES    (2 * AF_N_PER_FRAME) // 28

// Tabelas pre-computadas em af_init() e buffers de trabalho da FFT.
static float af_hann[AF_N_FFT];
static float af_bin_freqs[AF_N_BINS];
static float af_mel_fb[AF_N_MEL][AF_N_BINS];
static float af_dct[AF_N_MFCC][AF_N_MEL];
static float af_tw_re[AF_N_FFT / 2];
static float af_tw_im[AF_N_FFT / 2];
static uint16_t af_bitrev[AF_N_FFT];
static float af_re[AF_N_FFT];
static float af_im[AF_N_FFT];

static inline float af_hz_to_mel(float f) { return 2595.0f * log10f(1.0f + f / 700.0f); }
static inline float af_mel_to_hz(float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); }

static void af_init(void) {
    for (int i = 0; i < AF_N_FFT; i++)
        af_hann[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (AF_N_FFT - 1));

    for (int k = 0; k < AF_N_BINS; k++)
        af_bin_freqs[k] = (float)k * AF_SR / AF_N_FFT;

    // filtros mel triangulares
    float m_lo = af_hz_to_mel(AF_FMIN), m_hi = af_hz_to_mel(AF_FMAX);
    float f_pts[AF_N_MEL + 2];
    for (int i = 0; i < AF_N_MEL + 2; i++) {
        float mel = m_lo + (m_hi - m_lo) * i / (AF_N_MEL + 1);
        f_pts[i] = af_mel_to_hz(mel);
    }
    memset(af_mel_fb, 0, sizeof(af_mel_fb));
    for (int m = 1; m <= AF_N_MEL; m++) {
        float fl = f_pts[m - 1], fc = f_pts[m], fr = f_pts[m + 1];
        for (int k = 0; k < AF_N_BINS; k++) {
            float f = af_bin_freqs[k];
            if (f >= fl && f <= fc && fc > fl) af_mel_fb[m - 1][k] = (f - fl) / (fc - fl);
            else if (f > fc && f <= fr && fr > fc) af_mel_fb[m - 1][k] = (fr - f) / (fr - fc);
        }
    }

    // base DCT-II
    for (int n = 0; n < AF_N_MFCC; n++)
        for (int m = 0; m < AF_N_MEL; m++)
            af_dct[n][m] = cosf((float)M_PI * n * (2 * m + 1) / (2.0f * AF_N_MEL));

    // twiddles + bit reversal (512 pontos)
    for (int i = 0; i < AF_N_FFT / 2; i++) {
        af_tw_re[i] = cosf(-2.0f * (float)M_PI * i / AF_N_FFT);
        af_tw_im[i] = sinf(-2.0f * (float)M_PI * i / AF_N_FFT);
    }
    int logn = 0;
    while ((1 << logn) < AF_N_FFT) logn++;
    for (int i = 0; i < AF_N_FFT; i++) {
        int r = 0, x = i;
        for (int b = 0; b < logn; b++) { r = (r << 1) | (x & 1); x >>= 1; }
        af_bitrev[i] = (uint16_t)r;
    }
}

// FFT radix-2 in-place sobre af_re/af_im (comprimento AF_N_FFT)
static void af_fft(void) {
    for (int i = 0; i < AF_N_FFT; i++) {
        int j = af_bitrev[i];
        if (j > i) {
            float tr = af_re[i]; af_re[i] = af_re[j]; af_re[j] = tr;
            float ti = af_im[i]; af_im[i] = af_im[j]; af_im[j] = ti;
        }
    }
    for (int len = 2; len <= AF_N_FFT; len <<= 1) {
        int half = len >> 1;
        int step = AF_N_FFT / len;
        for (int i = 0; i < AF_N_FFT; i += len) {
            int k = 0;
            for (int j = 0; j < half; j++) {
                float wr = af_tw_re[k], wi = af_tw_im[k];
                int a = i + j, b = i + j + half;
                float xr = af_re[b] * wr - af_im[b] * wi;
                float xi = af_re[b] * wi + af_im[b] * wr;
                af_re[b] = af_re[a] - xr; af_im[b] = af_im[a] - xi;
                af_re[a] += xr;           af_im[a] += xi;
                k += step;
            }
        }
    }
}

// 14 features de um quadro (frame[AF_N_FFT] em float) -> out[AF_N_PER_FRAME]
static void af_frame_features(const float *frame, float *out) {
    // RMS e ZCR no sinal cru
    float sumsq = 0.0f;
    int zc = 0;
    float prev = frame[0];
    for (int i = 0; i < AF_N_FFT; i++) {
        sumsq += frame[i] * frame[i];
        if (i > 0) {
            float s = frame[i], p = prev;
            if ((s > 0 && p < 0) || (s < 0 && p > 0) ||
                (s == 0.0f) != (p == 0.0f)) zc++;
            prev = s;
        }
    }
    float rms = sqrtf(sumsq / AF_N_FFT + AF_EPS);
    float zcr = (float)zc / (AF_N_FFT - 1);

    // janela + FFT
    for (int i = 0; i < AF_N_FFT; i++) { af_re[i] = frame[i] * af_hann[i]; af_im[i] = 0.0f; }
    af_fft();

    // espectro de potência
    float power[AF_N_BINS];
    float psum = AF_EPS;
    for (int k = 0; k < AF_N_BINS; k++) {
        power[k] = af_re[k] * af_re[k] + af_im[k] * af_im[k];
        psum += power[k];
    }

    float centroid = 0.0f, band = 0.0f;
    for (int k = 0; k < AF_N_BINS; k++) {
        centroid += af_bin_freqs[k] * power[k];
        if (af_bin_freqs[k] >= AF_BAND_LO && af_bin_freqs[k] <= AF_BAND_HI) band += power[k];
    }
    centroid /= psum;
    float band_ratio = band / psum;

    // rolloff 85%
    float thr = 0.85f * (psum - AF_EPS);
    float cum = 0.0f; int roll_idx = AF_N_BINS - 1;
    for (int k = 0; k < AF_N_BINS; k++) {
        cum += power[k];
        if (cum >= thr) { roll_idx = k; break; }
    }
    float rolloff = af_bin_freqs[roll_idx];

    // largura de banda
    float bw = 0.0f;
    for (int k = 0; k < AF_N_BINS; k++) {
        float d = af_bin_freqs[k] - centroid;
        bw += power[k] * d * d;
    }
    bw = sqrtf(bw / psum);

    // mel -> log -> MFCC (DCT-II)
    float logmel[AF_N_MEL];
    for (int m = 0; m < AF_N_MEL; m++) {
        float e = 0.0f;
        for (int k = 0; k < AF_N_BINS; k++) e += af_mel_fb[m][k] * power[k];
        logmel[m] = logf(e + AF_EPS);
    }
    out[0] = rms; out[1] = centroid; out[2] = rolloff;
    out[3] = bw;  out[4] = zcr;      out[5] = band_ratio;
    for (int n = 0; n < AF_N_MFCC; n++) {
        float c = 0.0f;
        for (int m = 0; m < AF_N_MEL; m++) c += af_dct[n][m] * logmel[m];
        out[6 + n] = c;
    }
}

// Extrai o vetor de 28 features de uma janela [AF_WINDOW_SAMPLES] -> out[28]
static void af_extract(const float *window, float *out) {
    float sum[AF_N_PER_FRAME] = {0};
    float sumsq[AF_N_PER_FRAME] = {0};
    float ff[AF_N_PER_FRAME];
    for (int fr = 0; fr < AF_N_FRAMES; fr++) {
        af_frame_features(window + fr * AF_HOP, ff);
        for (int j = 0; j < AF_N_PER_FRAME; j++) { sum[j] += ff[j]; sumsq[j] += ff[j] * ff[j]; }
    }
    for (int j = 0; j < AF_N_PER_FRAME; j++) {
        float mean = sum[j] / AF_N_FRAMES;
        float var = sumsq[j] / AF_N_FRAMES - mean * mean;
        if (var < 0.0f) var = 0.0f;
        out[j] = mean;                       // médias: índices 0..13
        out[AF_N_PER_FRAME + j] = sqrtf(var); // desvios: índices 14..27
    }
}

#endif // AUDIO_FEATURES_H
