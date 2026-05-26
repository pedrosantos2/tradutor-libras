# ── Estágio único — runtime do backend ──────────────────────────────────────
FROM python:3.11-slim

# Dependências de sistema para OpenCV e MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependências Python antes de copiar o código (melhor cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código e artefatos necessários
COPY config.py .
COPY backend/ backend/
COPY hand_landmarker.task .
# Se já tiver modelo e labels treinados:
COPY modelo_libras.keras . 2>/dev/null || true
COPY labels.json          . 2>/dev/null || true

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
