# Detector de Anomalias Acústicas, Bem-te-vi 🐦

Sistema embarcado que detecta em tempo real o canto do bem-te-vi
(*Pitangus sulphuratus*) usando um ESP32 com microfone I2S INMP441. A
arquitetura roda sobre FreeRTOS com quatro tarefas concorrentes, faz a extração
de features acústicas no próprio dispositivo (edge computing) e usa um modelo
pré-treinado exportado em `.onnx`.

Anomalia escolhida para a ponderada, o canto do bem-te-vi.

---

## 1. Por que bem-te-vi

O bem-te-vi é uma ave territorial e muito vocal, com um canto marcante de três
notas ("bem-te-VI") e energia concentrada entre 1.5 e 4 kHz. Detectar esse canto
no próprio dispositivo, sem gravar nem transmitir áudio, serve para monitorar
biodiversidade urbana e rural com baixo custo e baixo consumo, estimar a presença
e a atividade da espécie ao longo do dia e preservar a privacidade, já que só o
resultado sai do ESP32.

---

## 2. Arquitetura RTOS

O trabalho é dividido em quatro tarefas com prioridades diferentes. As três
primeiras formam o pipeline e a quarta cuida da atuação.

| Tarefa | Prioridade | Core | Papel |
|--------|-----------|------|-------|
| `captureTask` | 3 (alta)  | 1 | Lê o INMP441 via I2S e enche um buffer de janela (~0.78 s) |
| `featureTask` | 2 (média) | 0 | Calcula as 28 features (RMS, centroide, rolloff, largura de banda, ZCR, MFCCs) |
| `detectTask`  | 1 (baixa) | 0 | Roda o MLP (28→32→16→1) e sinaliza a atuação pelo semáforo |
| `alertTask`   | 1         | 0 | Espera o semáforo binário e aciona LED e buzzer |

O diagrama completo, com o SVG, está em
[`docs/rtos_diagram.md`](docs/rtos_diagram.md) e
[`docs/rtos_diagram.svg`](docs/rtos_diagram.svg).

Sincronização usada (os três tipos que a atividade pede).

- **Filas.** `freeQueue` e `readyQueue` formam um pool de buffers no padrão
  produtor consumidor. A captura só escreve num buffer que retirou da `freeQueue`
  e ele só volta a ficar livre depois de processado, então a captura nunca
  sobrescreve um buffer que ainda está sendo lido. A `featureQueue` desacopla a
  extração de features da detecção.
- **Semáforo binário.** `alertSem` sinaliza um evento entre tarefas. A
  `detectTask` faz `xSemaphoreGive` ao decidir e a `alertTask` fica bloqueada em
  `xSemaphoreTake` até esse sinal, o que separa a decisão da atuação e tira o
  atraso do bipe de dentro do laço de detecção.
- **Mutex.** `serialMutex` dá exclusão mútua no `Serial`, evitando que logs de
  tarefas concorrentes se embaralhem.

A latência de cada etapa é medida com `esp_timer_get_time()`.

---

## 3. Pipeline de features (idêntico em Python e C)

A taxa de amostragem é 16 kHz, o quadro tem 512 amostras com passo de 256 (50% de
sobreposição) e janela de Hann. Cada quadro gera 14 features (RMS, centroide
espectral, rolloff de 85%, largura de banda, ZCR, razão de energia na banda de
1.5 a 4 kHz e 8 MFCCs). A janela de 48 quadros (~0.78 s) agrega tudo por média e
desvio, resultando em 28 features.

O extrator em C ([`include/audio_features.h`](include/audio_features.h)) é um
espelho exato do Python ([`model/features.py`](model/features.py)). A equivalência
é validada por [`model/verify_c_port.py`](model/verify_c_port.py), que reproduz o
algoritmo do C (inclusive a FFT radix-2) e compara com o numpy.

```
erro relativo maximo = 7.3e-15  (porta C valida, precisao de maquina)
```

---

## 4. Modelo de detecção

O modelo é um MLP `28 → 32 → 16 → 1` com ReLU nas camadas ocultas e sigmoide na
saída. Ele é pequeno o bastante para rodar no ESP32 (só multiplicações de matriz
e ReLU) e bem mais discriminativo que um modelo linear. A padronização
`z = (x - média) / desvio` fica embutida tanto no `.onnx` quanto no header C.

- Modelo em ONNX, [`model/bemtevi_detector.onnx`](model/bemtevi_detector.onnx),
  com o grafo `Sub, Div, Gemm, Relu, Gemm, Relu, Gemm, Sigmoid`, verificado com
  `onnxruntime`.
- Pesos para o firmware em [`include/model_params.h`](include/model_params.h).
- A porta para o firmware (`run_model()` no `main.cpp`) é validada por
  [`model/verify_mlp_port.py`](model/verify_mlp_port.py). O forward em loops
  bate com o numpy e o ONNX a aproximadamente 4.5e-08.

> **Esforço além do pedido.** A atividade permitia usar um modelo pré-treinado
> qualquer. Nós fomos atrás de gravações reais no xeno-canto e treinamos o nosso
> próprio modelo especificamente para o bem-te-vi, o que era opcional. Também
> existe um modo sintético (`python train_model.py`) para quem não quiser baixar
> áudio.

Métricas no conjunto de teste com dados reais do xeno-canto
(`model/metrics.txt`).

| Métrica | Valor |
|---------|-------|
| Accuracy  | 0.8635 |
| Precision | 0.8562 |
| Recall    | 0.7975 |
| F1        | 0.8258 |

O treino usou gravações reais de campo. Os positivos são de *Pitangus
sulphuratus* no Brasil (cerca de 180 gravações que viraram 1284 janelas após a
limpeza) e os negativos são uma amostra diversa de cerca de 200 outras espécies
brasileiras (cerca de 399 gravações que viraram 1879 janelas). Separar uma espécie
de centenas é um problema realista, por isso as métricas são honestas e bem
diferentes do 1.0 que o dataset sintético produzia. Dá para ajustar via
`MODEL_THRESHOLD` e com mais dados.

---

## 5. Latência e performance

O script `model/test_performance.py` avalia a qualidade num conjunto separado de
dados reais e cronometra o pipeline. Resultado típico no PC, que serve de
referência já que o ESP32 é mais lento porém folgado.

```
qualidade (teste real): accuracy=0.905  precision=0.895  recall=0.864  f1=0.880
extracao de features  : media ~3.9 ms   (FFT e MFCC dos 48 quadros)
deteccao (MLP)        : media ~0.02 ms
duracao da janela     : 784 ms de audio  ->  fator tempo-real ~200x
```

No ESP32, a etapa de captura tem uma latência inerente de cerca de 784 ms, que é
o tempo de gravar a janela. A extração de features custa poucos milissegundos e a
detecção fica abaixo de 1 ms, então sobra bastante folga e o sistema opera com
conforto em tempo real. Os números exatos por etapa são impressos pela
`detectTask` a cada janela.

---

## 5.1 Validação em hardware real 🎥

O sistema foi gravado no ESP32 físico com o microfone INMP441, reproduzindo sons
externos perto do microfone. As latências medidas no próprio ESP32 foram captura
784 ms, features cerca de 44 ms, detecção cerca de 0.18 ms e ponta a ponta cerca
de 829 ms, o que dá em torno de 18 vezes de folga sobre o tempo real.

Vídeos da demonstração no Google Drive (os arquivos são grandes e ficam fora do
git).

- Bem-te-vi, o sistema detecta. https://drive.google.com/file/d/135TFwSusxztVXhZIZ1w-JomyL0mMaf1N/view?usp=sharing
- Calopsita, o sistema rejeita. https://drive.google.com/file/d/133DDDBkuu48qhWjsvi3Av1iq3vE3iRRp/view?usp=sharing

### Teste 1, bem-te-vi real (verdadeiro positivo)
Durante o canto a probabilidade fica acima do threshold e o alerta dispara. Nas
pausas do canto ela cai abaixo de 0.5, exatamente como esperado, porque o `p`
acompanha a estrutura de três notas.

```
#91 BEM-TE-VI!  p=0.754   #99  BEM-TE-VI!  p=0.931
#92 BEM-TE-VI!  p=0.875   #100 BEM-TE-VI!  p=0.709
#93 BEM-TE-VI!  p=0.770   #101 BEM-TE-VI!  p=0.562
#94 BEM-TE-VI!  p=0.797   #102 -          p=0.432   (pausa do canto)
#95 BEM-TE-VI!  p=0.618   #103 BEM-TE-VI!  p=0.764
#97 -          p=0.381    (pausa)          detec acumulado 44
```

### Teste 2, calopsita, *Nymphicus hollandicus* (verdadeiro negativo)
Tocando o canto de outra ave o modelo rejeita corretamente. A probabilidade fica
quase sempre entre 0.00 e 0.05 e o contador de detecções não incrementa (`detec`
fica fixo em 86). Aparecem falsos positivos ocasionais e de baixa confiança (por
exemplo `p=0.368` e `p=0.234`), mas ficam abaixo do threshold de 0.5 e por isso
não geram alarme.

```
#326 - p=0.368   #329 - p=0.024   #332 - p=0.038   #338 - p=0.004
#327 - p=0.030   #330 - p=0.036   #333 - p=0.033   #339 - p=0.008
#328 - p=0.234   #331 - p=0.033   #337 - p=0.017   #342 - p=0.020
                                            detec permanece em 86 (nenhum alarme)
```

### O que os testes mostram
- Generalização. O modelo foi treinado no xeno-canto e reconheceu um áudio
  diferente captado por um microfone físico, ou seja, não decorou o treino.
- Separação clara. O bem-te-vi puxa o `p` para cima (0.56 a 1.0) e a calopsita
  para baixo (na maior parte abaixo de 0.05).
- Alerta. Neste teste o buzzer não estava conectado, mas o LED piscou certinho a
  cada detecção, o que comprova o caminho de atuação.
- Robustez. Os falsos positivos são raros e de baixa confiança. Subir o
  `MODEL_THRESHOLD` para algo como 0.6 ou 0.7 elimina esses casos ao custo de um
  pouco de recall.

---

## 6. Como rodar

### 6.1 Treinar e gerar os artefatos (Python)
```bash
cd model
pip install -r requirements.txt
python train_model.py       # gera .onnx, ../include/model_params.h e metrics.txt
python verify_c_port.py     # confere a paridade C x Python
python test_performance.py  # mede qualidade e latencia
```

Para treinar com dados reais do xeno-canto (opcional, foi o que fizemos).
```bash
cd model
set XC_KEY=sua_chave_da_api                    # Windows. No Linux/Mac use export
python fetch_xenocanto.py --pos 180 --neg 400  # baixa bem-te-vi e outras aves BR
python build_real_dataset.py                   # MP3 vira 28 features via ffmpeg
python train_model.py real                     # re-treina e re-exporta os artefatos
python gen_sim_clip.py                         # gera o clipe real para o modo Wokwi
```
Isso pede uma API key gratuita do xeno-canto (fica na sua conta) e o ffmpeg no
PATH. Como o extrator em C é igual ao do Python, o novo `model_params.h` já
funciona no firmware sem tocar no `main.cpp`.

### 6.2 Gravar no ESP32 pelo Arduino IDE

Este é o caminho principal, porque é o que a maioria usa.

**Passo 1, instalar o suporte ao ESP32 (uma vez só).** Em Arquivo, Preferências,
no campo de URLs adicionais de gerenciadores de placas, cole
`https://espressif.github.io/arduino-esp32/package_esp32_index.json`. Depois abra
Ferramentas, Placa, Gerenciador de Placas, procure por esp32 e instale o pacote
da Espressif. Se aparecer erro no `driver/i2s.h`, instale a versão 2.0.17, que o
código também suporta.

**Passo 2, montar a sketch.** O Arduino IDE exige que o arquivo principal seja
`.ino` e fique numa pasta com o mesmo nome, e que os headers estejam na mesma
pasta. Crie uma pasta `bemtevi_detector` com quatro arquivos.

| Arquivo na pasta | Copie o conteúdo de |
|---|---|
| `bemtevi_detector.ino` | `src/main.cpp` |
| `audio_features.h` | `include/audio_features.h` |
| `model_params.h` | `include/model_params.h` |
| `sim_clip.h` | `include/sim_clip.h` |

**Passo 3, escolher o modo.** Por padrão o firmware sobe em modo simulação
(`WOKWI_SIM` vale 1), que funciona num ESP32 físico com os dois botões, sem
microfone. Para usar o microfone INMP441 de verdade, adicione uma linha no topo
do `.ino`, antes de qualquer `#include`.

```cpp
#define WOKWI_SIM 0
```

**Passo 4, placa e porta.** Ligue o ESP32 no USB. Em Ferramentas, Placa,
selecione ESP32 Dev Module (ou o nome exato da sua placa). Em Ferramentas, Porta,
escolha a COM que apareceu. Se nenhuma porta aparecer, instale o driver USB serial
CP2102 ou CH340, conforme o chip da placa.

**Passo 5, gravar.** Clique em Upload. Se travar em `Connecting......`, segure o
botão BOOT da placa enquanto isso aparece e solte quando começar a gravar.

**Passo 6, ver a saída.** Abra o Monitor Serial em 115200 baud. Vão aparecer as
linhas de detecção e de latência.

Ligações do microfone real (para o modo `WOKWI_SIM=0`). GPIO14 no SCK, GPIO15 no
WS, GPIO32 no SD, 3V3 no VDD, GND no GND e L/R no GND. LED no GPIO2 via resistor
de 220 Ω e buzzer no GPIO26. O detalhe das portas está em
[`docs/rtos_diagram.md`](docs/rtos_diagram.md).

### 6.3 Alternativa por linha de comando (PlatformIO)
```bash
pio run -t upload             # WOKWI_SIM=1 por padrao
pio device monitor -b 115200
```
Para o microfone real, mude `platformio.ini` para `-D WOKWI_SIM=0`.

### 6.4 Testar no Wokwi (virtual, sem microfone)
Importe [`diagram.json`](diagram.json). Como o Wokwi não injeta áudio real no I2S,
o firmware roda com `WOKWI_SIM=1` e reproduz um trecho real de bem-te-vi embutido
em `include/sim_clip.h`, de forma que o mesmo modelo usado no hardware reconhece o
sinal. Segure o botão de bem-te-vi (GPIO4) e o LED (GPIO2) acende com o buzzer
tocando. Segure o botão de ruído (GPIO5) e não há alerta. No wokwi.com o projeto
precisa dos quatro arquivos citados no passo 2.

---

## 7. Estrutura do repositório

```
platformio.ini              build ESP32 (Arduino, arduino-esp32 2.0.x)
wokwi.toml                  aponta o firmware para o simulador
diagram.json                circuito Wokwi (ESP32, LED, buzzer, botoes)
include/
  audio_features.h          extracao de features em C (FFT, MFCC) e af_extract()
  model_params.h            pesos do MLP (gerado por train_model.py)
  sim_clip.h                clipe real de bem-te-vi para o modo Wokwi (gerado)
src/
  main.cpp                  FreeRTOS com 4 tarefas, filas, mutex, semaforo, I2S
model/
  features.py               extrator de referencia (espelho do C)
  generate_dataset.py       sintese de cantos e negativos (fallback)
  fetch_xenocanto.py        baixa audio real do xeno-canto (API v3)
  build_real_dataset.py     MP3 vira janelas e 28 features (ffmpeg)
  train_model.py            treina o MLP (real ou sintetico) e exporta o header C
  export_onnx.py            exporta bemtevi_detector.onnx
  gen_sim_clip.py           gera o sim_clip.h a partir de um positivo real
  verify_c_port.py          valida o extrator C contra o Python
  verify_mlp_port.py        valida run_model() contra o ONNX e o numpy
  test_performance.py       qualidade e latencia
  bemtevi_detector.onnx     entregavel, modelo em ONNX
  metrics.txt               metricas de teste
docs/
  rtos_diagram.md           pinout e diagrama de tarefas
  rtos_diagram.svg          diagrama de tarefas em imagem
```

---

## 8. Entregáveis da ponderada

- [x] Código-fonte do firmware e do treino neste repositório
- [x] Captura contínua via ESP32 e INMP441 por I2S
- [x] Quatro tarefas RTOS sincronizadas com filas, mutex e semáforo binário
- [x] Modelo pré-treinado em `.onnx` (`model/bemtevi_detector.onnx`)
- [x] Diagrama de tarefas RTOS em mermaid e SVG (`docs/rtos_diagram.md` e `.svg`)
- [x] Latência medida e documentada por etapa (na `detectTask` e na seção 5)
- [x] Alerta por LED e buzzer
- [x] Conflitos de concorrência resolvidos (pool de buffers, mutex, semáforo)
- [x] Script de teste que simula anomalias e mede performance
- [x] Relatório técnico (este README e a pasta `docs`)
- [x] Validação em hardware real com vídeos da demonstração (bem-te-vi e calopsita)
