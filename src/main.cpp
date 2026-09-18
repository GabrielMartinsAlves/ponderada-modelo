/*
 * Detector acustico de bem-te-vi (Pitangus sulphuratus).
 * ESP32 + INMP441 (I2S) + LED + buzzer, com FreeRTOS.
 * Pipeline em tarefas concorrentes: captura -> features -> deteccao -> alerta.
 */
#include <Arduino.h>
#include "esp_timer.h"
#include "audio_features.h"
#include "model_params.h"

#ifndef WOKWI_SIM
#define WOKWI_SIM 1
#endif

#if !WOKWI_SIM
#include "driver/i2s.h"
#endif

// INMP441 (I2S)
#define I2S_SCK_PIN   14
#define I2S_WS_PIN    15
#define I2S_SD_PIN    32
// Atuadores
#define LED_PIN        2
#define BUZZER_PIN    26
// Botoes de simulacao (somente no modo Wokwi, sem microfone)
#define BTN_BEMTEVI    4
#define BTN_NOISE      5

#define BUZZER_CH      0

// O arduino-esp32 3.x removeu ledcSetup/ledcAttachPin e passou a endereçar o
// buzzer pelo pino. O core 2.x usa canal. Este alias cobre as duas versoes.
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  #define BUZZER_TARGET BUZZER_PIN
#else
  #define BUZZER_TARGET BUZZER_CH
#endif

#define NUM_BUFFERS    2
#define WIN            AF_WINDOW_SAMPLES

static float *audioBuf[NUM_BUFFERS];

typedef struct {
    float feats[AF_N_FEATURES];
    int64_t t_cap_start;
    int64_t t_cap_end;
    int64_t t_feat_end;
} FeatureMsg;

static QueueHandle_t freeQueue;      // indices de buffers livres
static QueueHandle_t readyQueue;     // indices de buffers cheios
static QueueHandle_t featureQueue;   // features + timestamps
static SemaphoreHandle_t serialMutex;  // exclusao mutua no Serial
static SemaphoreHandle_t alertSem;     // semaforo binario: sinaliza deteccao -> alertTask

static int64_t bufCapStart[NUM_BUFFERS];
static int64_t bufCapEnd[NUM_BUFFERS];
static volatile bool g_lastAnomaly = false;

static void logline(const String &s) {
    if (xSemaphoreTake(serialMutex, portMAX_DELAY) == pdTRUE) {
        Serial.println(s);
        xSemaphoreGive(serialMutex);
    }
}

#if WOKWI_SIM
#include "sim_clip.h"   // trecho real de bem-te-vi embutido para o modo Wokwi

static uint32_t rng_state = 0x1234abcd;
static inline float frand() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((int32_t)rng_state) / 2147483648.0f;
}

static void synth_bemtevi(float *buf) {
    for (int i = 0; i < WIN; i++) {
        float s = (i < SIM_CLIP_LEN) ? (SIM_CLIP[i] / 32768.0f) : 0.0f;
        buf[i] = s + 0.02f * frand();
    }
}

static void synth_noise(float *buf) {
    for (int i = 0; i < WIN; i++) buf[i] = 0.4f * frand();
}
#endif

// z=(x-mean)/scale; h0=relu(z*W0+B0); h1=relu(h0*W1+B1); p=sigmoid(h1*W2+B2)
static float run_model(const float *feats) {
    float z[MODEL_IN];
    for (int i = 0; i < MODEL_IN; i++)
        z[i] = (feats[i] - MODEL_MEAN[i]) / MODEL_SCALE[i];

    float h0[MODEL_H0];
    for (int j = 0; j < MODEL_H0; j++) {
        float acc = MODEL_B0[j];
        for (int i = 0; i < MODEL_IN; i++) acc += z[i] * MODEL_W0[i * MODEL_H0 + j];
        h0[j] = acc > 0.0f ? acc : 0.0f;
    }

    float h1[MODEL_H1];
    for (int j = 0; j < MODEL_H1; j++) {
        float acc = MODEL_B1[j];
        for (int i = 0; i < MODEL_H0; i++) acc += h0[i] * MODEL_W1[i * MODEL_H1 + j];
        h1[j] = acc > 0.0f ? acc : 0.0f;
    }

    float out = MODEL_B2;
    for (int i = 0; i < MODEL_H1; i++) out += h1[i] * MODEL_W2[i];
    return 1.0f / (1.0f + expf(-out));
}

// Task 1, alta prioridade. Enche um buffer de janela com audio.
static void captureTask(void *arg) {
    (void)arg;
#if !WOKWI_SIM
    static int32_t raw[512];
#endif
    int idx;
    for (;;) {
        if (xQueueReceive(freeQueue, &idx, portMAX_DELAY) != pdTRUE) continue;
        bufCapStart[idx] = esp_timer_get_time();
        float *buf = audioBuf[idx];

#if WOKWI_SIM
        if (digitalRead(BTN_BEMTEVI) == LOW) synth_bemtevi(buf);
        else                                 synth_noise(buf);
        vTaskDelay(pdMS_TO_TICKS((WIN * 1000) / AF_SR));
#else
        int got = 0;
        while (got < WIN) {
            size_t nbytes = 0;
            int want = WIN - got; if (want > 512) want = 512;
            i2s_read(I2S_NUM_0, raw, want * sizeof(int32_t), &nbytes, portMAX_DELAY);
            int n = nbytes / sizeof(int32_t);
            for (int i = 0; i < n; i++)
                buf[got + i] = (float)(raw[i] >> 8) / 8388608.0f;  // 24 bits -> [-1,1]
            got += n;
        }
#endif
        bufCapEnd[idx] = esp_timer_get_time();
        xQueueSend(readyQueue, &idx, portMAX_DELAY);
    }
}

// Task 2, prioridade media. Extrai as 28 features da janela.
static void featureTask(void *arg) {
    (void)arg;
    int idx;
    FeatureMsg msg;
    for (;;) {
        if (xQueueReceive(readyQueue, &idx, portMAX_DELAY) != pdTRUE) continue;
        msg.t_cap_start = bufCapStart[idx];
        msg.t_cap_end   = bufCapEnd[idx];

        af_extract(audioBuf[idx], msg.feats);

        xQueueSend(freeQueue, &idx, portMAX_DELAY);   // libera o buffer para a captura
        msg.t_feat_end = esp_timer_get_time();
        xQueueSend(featureQueue, &msg, portMAX_DELAY);
    }
}

// Task 3, prioridade baixa. Roda o modelo e sinaliza a atuacao pelo semaforo.
static void detectTask(void *arg) {
    (void)arg;
    FeatureMsg msg;
    uint32_t n = 0, hits = 0;
    for (;;) {
        if (xQueueReceive(featureQueue, &msg, portMAX_DELAY) != pdTRUE) continue;
        float p = run_model(msg.feats);
        int64_t t_det_end = esp_timer_get_time();
        bool anomaly = p > MODEL_THRESHOLD;

        g_lastAnomaly = anomaly;
        xSemaphoreGive(alertSem);
        n++; if (anomaly) hits++;

        float lat_cap  = (msg.t_cap_end  - msg.t_cap_start) / 1000.0f;
        float lat_feat = (msg.t_feat_end - msg.t_cap_end)   / 1000.0f;
        float lat_det  = (t_det_end      - msg.t_feat_end)  / 1000.0f;
        float lat_e2e  = (t_det_end      - msg.t_cap_start) / 1000.0f;

        char line[200];
        snprintf(line, sizeof(line),
                 "#%lu %s  p=%.3f | lat[ms] captura=%.1f features=%.2f deteccao=%.3f e2e=%.1f | detec=%lu",
                 (unsigned long)n, anomaly ? "BEM-TE-VI!" : "-        ",
                 p, lat_cap, lat_feat, lat_det, lat_e2e, (unsigned long)hits);
        logline(String(line));
    }
}

// Task 4, atuacao. Espera o semaforo binario e aciona LED e buzzer.
static void alertTask(void *arg) {
    (void)arg;
    for (;;) {
        if (xSemaphoreTake(alertSem, portMAX_DELAY) != pdTRUE) continue;
        bool anomaly = g_lastAnomaly;
        digitalWrite(LED_PIN, anomaly ? HIGH : LOW);
        if (anomaly) {
            ledcWriteTone(BUZZER_TARGET, 3000);
            vTaskDelay(pdMS_TO_TICKS(150));
            ledcWriteTone(BUZZER_TARGET, 0);
        } else {
            ledcWriteTone(BUZZER_TARGET, 0);
        }
    }
}

#if !WOKWI_SIM
static void i2s_setup() {
    i2s_config_t cfg = {};
    cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
    cfg.sample_rate = AF_SR;
    cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
    cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;   // INMP441 com L/R no GND
    cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    cfg.dma_buf_count = 8;
    cfg.dma_buf_len = 256;
    cfg.use_apll = false;

    i2s_pin_config_t pins = {};
    pins.bck_io_num = I2S_SCK_PIN;
    pins.ws_io_num = I2S_WS_PIN;
    pins.data_out_num = I2S_PIN_NO_CHANGE;
    pins.data_in_num = I2S_SD_PIN;

    i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pins);
    i2s_zero_dma_buffer(I2S_NUM_0);
}
#endif

void setup() {
    Serial.begin(115200);
    delay(300);

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    pinMode(BTN_BEMTEVI, INPUT_PULLUP);
    pinMode(BTN_NOISE, INPUT_PULLUP);
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcAttach(BUZZER_PIN, 3000, 8);
#else
    ledcSetup(BUZZER_CH, 3000, 8);
    ledcAttachPin(BUZZER_PIN, BUZZER_CH);
#endif
    ledcWriteTone(BUZZER_TARGET, 0);

    af_init();

    for (int i = 0; i < NUM_BUFFERS; i++)
        audioBuf[i] = (float *)malloc(sizeof(float) * WIN);

    serialMutex = xSemaphoreCreateMutex();
    alertSem    = xSemaphoreCreateBinary();
    freeQueue    = xQueueCreate(NUM_BUFFERS, sizeof(int));
    readyQueue   = xQueueCreate(NUM_BUFFERS, sizeof(int));
    featureQueue = xQueueCreate(4, sizeof(FeatureMsg));

    for (int i = 0; i < NUM_BUFFERS; i++) xQueueSend(freeQueue, &i, 0);

#if !WOKWI_SIM
    i2s_setup();
#endif

    Serial.println();
    Serial.println("=== Detector Acustico de BEM-TE-VI (FreeRTOS) ===");
    Serial.printf("Modo: %s | janela=%d amostras (%.0f ms) | features=%d\n",
                  WOKWI_SIM ? "SIMULACAO (botoes GPIO4/GPIO5)" : "HARDWARE (INMP441 I2S)",
                  WIN, WIN * 1000.0f / AF_SR, AF_N_FEATURES);
    Serial.printf("Threshold=%.2f  (p>threshold => bem-te-vi)\n", MODEL_THRESHOLD);
    Serial.println("Tarefas: captura(P3) -> features(P2) -> deteccao(P1) -> alerta(P1)");
    Serial.println("--------------------------------------------------");

    xTaskCreatePinnedToCore(captureTask, "capture", 4096, NULL, 3, NULL, 1);
    xTaskCreatePinnedToCore(featureTask, "feature", 8192, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(detectTask,  "detect",  4096, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(alertTask,   "alert",   2048, NULL, 1, NULL, 0);
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}
