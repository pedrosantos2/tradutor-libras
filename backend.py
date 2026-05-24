import asyncio
import base64
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

# --- CONFIGURAÇÕES ---
ACTIONS = [
    "OI", "TCHAU", "EU", "NOME", "OBRIGADO", "SIM", "NAO",
    "POR_FAVOR", "DESCULPA", "BEM", "GOSTAR", "AJUDA",
    "ENTENDER", "NAO_ENTENDER", "REPETIR", "PRAZER", "AMIGO", "SURDO",
]
FRAME_WINDOW = 30
COORD_SIZE = 126
THRESHOLD = 0.85
MODEL_PATH = Path("hand_landmarker.task")

# --- CARREGA MODELO ---
model_lstm = load_model("modelo_libras.h5")
# Usa o número de classes que o modelo foi treinado, não a lista acima
n_classes = model_lstm.output_shape[-1]
ACTIONS = ACTIONS[:n_classes]

# --- SETUP MEDIAPIPE ---
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
mp_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)
hand_landmarker = vision.HandLandmarker.create_from_options(mp_options)

print(f"Modelo carregado: {n_classes} classes → {ACTIONS}")

# --- OLLAMA ---
def chamar_ollama(glossas: list) -> str:
    if not glossas:
        return ""
    prompt = f"Converta estas glossas de LIBRAS para português fluído: {glossas}"
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "tradutor-sc", "prompt": prompt, "stream": False},
            timeout=30,
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"Erro Ollama: {e}"


# --- FASTAPI ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


MIN_HAND_FRAMES = 20    # de 30, quantos precisam ter mão para predizer
MAX_NO_HAND_FRAMES = 10 # frames sem mão antes de resetar a sequência


class SessionState:
    def __init__(self):
        self.sequence: list = []
        self.hand_flags: list = []  # True/False por frame: mão detectada?
        self.no_hand_streak: int = 0
        self.glossas: list = []
        self.last_prediction: str = ""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = SessionState()

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "frame")

            # --- TRADUZIR ---
            if action == "translate":
                frase = chamar_ollama(state.glossas)
                state.glossas = []
                state.last_prediction = ""
                await websocket.send_json({"type": "translation", "text": frase})
                continue

            # --- LIMPAR ---
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

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = hand_landmarker.detect(mp_image)

            coords = np.zeros(COORD_SIZE)
            landmarks_ui = []
            hand_detected = bool(results.hand_landmarks)

            if hand_detected:
                left_pts = [0.0] * 63
                right_pts = [0.0] * 63
                for hand_lms, handedness_list in zip(results.hand_landmarks, results.handedness):
                    label = handedness_list[0].category_name  # "Left" or "Right"
                    pts = []
                    for lm in hand_lms:
                        pts.extend([lm.x, lm.y, lm.z])
                        landmarks_ui.append({"x": lm.x, "y": lm.y})
                    if label == "Left":
                        left_pts = pts
                    else:
                        right_pts = pts
                # Left → slot 0 (0-62), Right → slot 1 (63-125), always consistent
                coords[:] = left_pts + right_pts
                state.no_hand_streak = 0
            else:
                state.no_hand_streak += 1
                if state.no_hand_streak > MAX_NO_HAND_FRAMES:
                    state.sequence = []
                    state.hand_flags = []
                    state.last_prediction = ""

            state.sequence.append(coords)
            state.sequence = state.sequence[-FRAME_WINDOW:]
            state.hand_flags.append(hand_detected)
            state.hand_flags = state.hand_flags[-FRAME_WINDOW:]

            current_sign = ""
            confidence = 0.0

            hand_frame_count = sum(state.hand_flags)
            if len(state.sequence) == FRAME_WINDOW and hand_frame_count >= MIN_HAND_FRAMES:
                input_data = np.expand_dims(state.sequence, axis=0)
                loop = asyncio.get_event_loop()
                res = await loop.run_in_executor(
                    None, lambda: model_lstm.predict(input_data, verbose=0)[0]
                )
                idx = int(np.argmax(res))
                confidence = float(res[idx])

                if confidence > THRESHOLD:
                    current_sign = ACTIONS[idx]
                    if current_sign != state.last_prediction:
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
        pass
