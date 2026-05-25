import asyncio
import base64
import logging
import os

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import config as cfg
from backend.inference import ACTIONS, n_classes, predict
from backend.mediapipe_runner import detect
from backend.ollama_client import chamar_ollama
from backend.session import SessionState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MIN_HAND_FRAMES = 20
MAX_NO_HAND_FRAMES = 10

app = FastAPI(title="Tradutor LIBRAS")

_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_classes": n_classes, "actions": ACTIONS}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = SessionState()
    logger.info("Cliente conectado: %s", websocket.client)

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "frame")

            if action == "translate":
                frase = chamar_ollama(state.glossas)
                state.glossas = []
                state.last_prediction = ""
                await websocket.send_json({"type": "translation", "text": frase})
                continue

            if action == "clear":
                state.glossas = []
                state.last_prediction = ""
                await websocket.send_json({"type": "clear"})
                continue

            # --- PROCESSAR FRAME ---
            frame_b64 = data.get("frame", "")
            img_bytes = base64.b64decode(frame_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            coords, landmarks_ui = detect(frame)
            hand_detected = bool(coords.any())

            if hand_detected:
                state.no_hand_streak = 0
            else:
                state.no_hand_streak += 1
                if state.no_hand_streak > MAX_NO_HAND_FRAMES:
                    state.sequence = []
                    state.hand_flags = []
                    state.last_prediction = ""

            state.sequence.append(coords)
            state.sequence = state.sequence[-cfg.FRAME_WINDOW:]
            state.hand_flags.append(hand_detected)
            state.hand_flags = state.hand_flags[-cfg.FRAME_WINDOW:]

            current_sign = ""
            confidence = 0.0

            if len(state.sequence) == cfg.FRAME_WINDOW and sum(state.hand_flags) >= MIN_HAND_FRAMES:
                loop = asyncio.get_event_loop()
                current_sign, confidence = await loop.run_in_executor(
                    None, lambda: predict(state.sequence)
                )
                if current_sign and current_sign != state.last_prediction:
                    state.glossas.append(current_sign)
                    state.last_prediction = current_sign

            await websocket.send_json({
                "type": "frame_result",
                "landmarks": landmarks_ui,
                "current_sign": current_sign,
                "confidence": round(confidence, 3),
                "glossas": state.glossas,
            })

    except WebSocketDisconnect:
        logger.info("Cliente desconectado: %s", websocket.client)
