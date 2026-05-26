# Tradutor LIBRAS

Tradução de LIBRAS em tempo real no browser.  
Câmera → MediaPipe (landmarks) → LSTM → glossas → Ollama/Gemma3 → português fluído.

## Pré-requisitos

- Python 3.11+
- Node.js 20+
- [Ollama](https://ollama.com) com o modelo `tradutor-sc` (opcional — há fallback)

## Instalação

```bash
# Backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # ajuste se necessário

# Frontend
cd frontend
npm install
```

## Rodando

**Terminal 1 — backend:**
```bash
source venv/bin/activate
uvicorn backend.app:app --reload
# API disponível em http://localhost:8000
# Healthcheck: curl localhost:8000/health
```

**Terminal 2 — frontend (dev):**
```bash
cd frontend
npm run dev
# Abre http://localhost:5173
```

**Ollama (opcional):**
```bash
ollama serve
ollama create tradutor-sc -f Modelfile   # só na primeira vez
```

Se o Ollama não estiver rodando, o backend usa fallback: glossas concatenadas em minúsculas.

## Pipeline de dados

1. **Baixar vídeos** — `python scripts/baixar_videos.py`  
   Usa `yt-dlp` (`pip install yt-dlp`). Salva em `videos_baixados/<SINAL>/`.

2. **Extrair landmarks** — `python scripts/processador_videos.py`  
   Detecta automaticamente o segmento com mão em movimento e reamostras para 30 frames.  
   Saída: `DATA/<SINAL>/ext_*.npy` (shape `(30, 126)`).

3. **Aumentar dataset** — `python scripts/augment_total.py`  
   Gera 10× variações por arquivo (escala, translação, jitter).  
   Saída: `DATA/<SINAL>/aug_*.npy`.

4. **Treinar modelo** — `python scripts/treinal_modelo.py`  
   Split 80/10/10 (train/val/test). Salva `modelo_libras.keras` + `labels.json`.  
   Imprime matriz de confusão e accuracy por classe ao final.

5. **Testar localmente (CLI)** — `python scripts/tradutor_cli.py`  
   Interface OpenCV com webcam. Útil para debug sem subir o backend web.

6. **Coletar sinais novos** — `python scripts/coletor_dados.py`  
   Webcam interativa. Teclas: `S` gravar, `A`/`D` navegar sinais, `Q` sair.

## Adicionar um sinal novo

1. Edite `config.py` — adicione o nome em `SIGNS`.
2. Baixe vídeos ou colete via `coletor_dados.py`.
3. Rode o pipeline: `processador_videos.py` → `augment_total.py` → `treinal_modelo.py`.
4. Reinicie o backend — ele carrega `labels.json` automaticamente.

## Docker (backend)

```bash
docker build -t traduto-libras-backend .
docker run -p 8000:8000 --env-file .env traduto-libras-backend
```

O frontend pode ser servido separadamente (`npm run build` → qualquer servidor estático)  
ou via `docker-compose` (veja TODO abaixo).

## Estrutura

```
├── config.py               # fonte única de verdade (sinais, paths, threshold)
├── backend/                # FastAPI: app.py, inference.py, mediapipe_runner.py,
│   │                       #          ollama_client.py, session.py
├── frontend/               # Vite + React + TypeScript + Tailwind
├── scripts/                # pipeline de dados e CLI
│   ├── baixar_videos.py
│   ├── processador_videos.py
│   ├── augment_total.py
│   ├── treinal_modelo.py
│   ├── coletor_dados.py
│   └── tradutor_cli.py
├── DATA/                   # coordenadas .npy por sinal (gitignored)
├── videos_baixados/        # vídeos MP4 brutos (gitignored)
├── Modelfile               # configuração Ollama
├── requirements.txt
└── Dockerfile
```

## TODO

- [ ] `docker-compose.yml` com backend + ollama + frontend estático
- [ ] Normalização de coords relativa à palma (robustez a posição/escala)
- [ ] Frames binários no WebSocket (reduz overhead do base64)
- [ ] Transformer / TCN no lugar de LSTM
- [ ] Múltiplos usuários simultâneos com fila de inferência
- [ ] Deploy em produção (Fly.io, Render, etc.)
