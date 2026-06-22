# 🤟 Tradutor LIBRAS-SC

Sistema de reconhecimento de **LIBRAS** (Língua Brasileira de Sinais) que usa visão computacional
para detectar sinais a partir da webcam ou de vídeos e os traduz para **português fluído** com a
ajuda de um modelo de linguagem local (Ollama).

O pipeline rastreia as **mãos** com o MediaPipe Hand Landmarker, classifica a sequência de
movimentos com uma rede neural **LSTM** e converte as "glossas" detectadas (palavras-sinal) em
frases naturais através de um modelo Gemma customizado.

```
Webcam/Vídeo ──▶ MediaPipe (126 coords) ──▶ LSTM ──▶ Glossas ──▶ Ollama/Gemma ──▶ Português
```

---

## ✨ Como funciona

Cada sinal é representado por uma **sequência de 30 frames**, e cada frame contém as coordenadas
`(x, y, z)` de **2 mãos × 21 pontos = 126 valores**. Quando uma mão não é detectada, os valores
ficam zerados para manter o formato fixo esperado pela rede.

A LSTM recebe um tensor `(30, 126)` e devolve a probabilidade de cada sinal. Quando a confiança
ultrapassa o *threshold* (0.85), a glossa é adicionada à frase. Ao final, o conjunto de glossas é
enviado ao Ollama, que devolve a tradução em português.

---

## 📂 Estrutura do projeto

```
traduto-libras/
├── coletor_dados.py        # Coleta sinais pela webcam (grava sequências .npy)
├── processador_videos.py   # Extrai coordenadas de vídeos .mp4 → .npy
├── augment_total.py        # Data augmentation (10× variações por amostra)
├── treinal_modelo.py       # Treina a LSTM → modelo_libras.h5 / .keras
├── tradutor_final.py       # Tradutor em tempo real (webcam + OpenCV + Ollama)
├── Modelfile               # Definição do modelo Gemma customizado (Ollama)
│
├── videos_baixados/        # Vídeos-fonte por sinal (entrada do processador)
│   ├── oi/ gostar/ laranja/ melancia/ ...
├── DATA/                   # Dataset de coordenadas (.npy), uma pasta por sinal
│   ├── OI/ GOSTAR/ OBRIGADO/ ...
│
├── hand_landmarker.task    # Modelo MediaPipe (baixado automaticamente)
├── modelo_libras.h5        # Pesos da LSTM treinada (legado Keras)
└── modelo_libras.keras     # Pesos da LSTM treinada (formato novo)
```

> ⚠️ Arquivos de dados (`*.npy`), modelos (`*.h5`, `*.task`) e o `venv/` são ignorados pelo Git
> (veja `.gitignore`). O repositório versiona apenas o código e alguns vídeos-fonte.

---

## 🔧 Pré-requisitos

- **Python 3.12**
- **Webcam** (para coleta e tradução em tempo real)
- **[Ollama](https://ollama.com)** instalado e rodando localmente (para a etapa de tradução)

### Dependências Python

Ainda não há um `requirements.txt`. Instale manualmente:

```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install opencv-python mediapipe numpy tensorflow scikit-learn requests
```

### Modelo de tradução (Ollama)

O tradutor usa um modelo Gemma customizado definido no `Modelfile`. Crie-o uma vez:

```bash
ollama create tradutor-sc -f Modelfile
```

> O `Modelfile` parte de `gemma3:4b`. O Ollama baixará o modelo base automaticamente na primeira vez.

---

## 🚀 Uso

### 1. Obter dados de treino

Você pode coletar sinais de duas formas:

**a) Pela webcam** — grava sequências de 30 frames ao vivo:

```bash
python coletor_dados.py
# 1/2/3: trocar de sinal  •  S: gravar 30 frames  •  Q: sair
```

**b) A partir de vídeos** — coloque arquivos `.mp4` em `videos_baixados/<SINAL>/` e extraia:

```bash
python processador_videos.py
# Gera os .npy correspondentes em DATA/<SINAL>/
```

### 2. Aumentar o dataset (opcional, recomendado)

Cria 10 variações (escala, translação e jitter) de cada amostra para robustez:

```bash
python augment_total.py
```

### 3. Treinar o modelo

Treina a LSTM com os `.npy` em `DATA/` e salva os pesos:

```bash
python treinal_modelo.py
# → modelo_libras.h5
```

### 4. Traduzir em tempo real

Inicie o Ollama, depois rode o tradutor:

```bash
python tradutor_final.py
# Espaço: traduzir as glossas com o Gemma
# C: limpar a frase  •  Q: sair
```

---

## 🧠 Detalhes técnicos

| Componente        | Configuração                                              |
|-------------------|-----------------------------------------------------------|
| Detecção de mãos  | MediaPipe Hand Landmarker (`num_hands=2`)                 |
| Entrada da rede   | `(30 frames, 126 coords)`                                 |
| Arquitetura       | 3× LSTM (64→128→64) + Dropout + Dense (64→32→softmax)     |
| Treino            | Adam, `categorical_crossentropy`, 300 épocas, batch 8     |
| Threshold         | 0.85 de confiança para aceitar uma glossa                 |
| Tradução          | Ollama `tradutor-sc` (Gemma 3 4B), `temperature=0`        |

---

## ⚠️ Inconsistências conhecidas

Este é um projeto em evolução e os scripts ainda não compartilham uma configuração central.
Antes de treinar/inferir, alinhe a lista de sinais entre os arquivos:

- A lista de sinais (`ACTIONS` / `SIGNS`) está **duplicada e divergente** entre os scripts:
  - `tradutor_final.py` → `["OI", "GOSTAR", "MELANCIA"]`
  - `treinal_modelo.py` / `processador_videos.py` / `augment_total.py` → `["OI", "GOSTAR", "LARANJA", "MELANCIA"]`
  - `coletor_dados.py` → `["OI", "GOSTAR", "MANDIOCA"]` (e captura **apenas 1 mão / 63 coords**)
- A pasta `DATA/` contém **~20 sinais**, bem mais do que os scripts referenciam.
- A ordem de `ACTIONS` no tradutor **deve ser idêntica** à usada no treino, ou as predições
  apontarão para o sinal errado.
- `coletor_dados.py` grava 63 coordenadas (1 mão), incompatível com as 126 (2 mãos) esperadas
  pelo treino — use `processador_videos.py` ou ajuste o coletor antes de misturar as fontes.

> Há também uma versão web experimental (FastAPI + Vite/React) cujos artefatos de build estão em
> `backend/` e `frontend/dist/`, mas o código-fonte não está versionado neste repositório.

---

## 🗺️ Próximos passos

- [ ] Centralizar a lista de sinais e os hiperparâmetros em um único `config.py` / `labels.json`
- [ ] Adicionar `requirements.txt`
- [ ] Unificar a captura para 2 mãos (126 coords) em todas as fontes
- [ ] Usar um conjunto de validação separado no treino (hoje a validação reusa o teste)
- [ ] Versionar a versão web (frontend + backend) ou removê-la do repositório
