import cv2
import mediapipe as mp
import numpy as np
import importlib
import os
import ssl
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURAÇÕES DE DIRETÓRIOS ---
SIGNS = ["OI", "GOSTAR", "MANDIOCA"]
DATA_PATH = Path("DATA")
for sign in SIGNS:
    (DATA_PATH / sign).mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# --- FUNÇÕES DE SUPORTE ---
def ensure_model_file() -> str:
    if not MODEL_PATH.exists():
        print("Baixando modelo...")
        try:
            with urlopen(MODEL_URL, timeout=30) as response:
                MODEL_PATH.write_bytes(response.read())
        except Exception:
            # Fallback simples para erro de SSL comum no Mac
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            with urlopen(MODEL_URL, timeout=30) as response:
                MODEL_PATH.write_bytes(response.read())
    return str(MODEL_PATH)

# --- SETUP MEDIAPIPE ---
base_options = python.BaseOptions(model_asset_path=ensure_model_file())
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1, # Focando em uma mão para simplificar o dataset inicial
    min_hand_detection_confidence=0.7,
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# --- VARIÁVEIS DE ESTADO ---
cap = cv2.VideoCapture(0)
current_sign_idx = 0
is_recording = False
frame_counter = 0
sequence_data = []

print("\n--- COMANDOS ---")
print("1, 2, 3: Alternar entre OI, GOSTAR, MANDIOCA")
print("S: Gravar sequência de 30 frames")
print("Q: Sair\n")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    image_h, image_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = hand_landmarker.detect(mp_image)

    # Extração de coordenadas (Mesmo se não detectar, precisamos de zeros para manter o shape da rede neural)
    current_frame_landmarks = np.zeros(63) # 21 pontos * 3 (x,y,z)

    if results.hand_landmarks:
        hand_points = results.hand_landmarks[0] # Pega a primeira mão
        landmarks = []
        for lm in hand_points:
            landmarks.extend([lm.x, lm.y, lm.z])
            # Desenha na tela para feedback
            x_px, y_px = int(lm.x * image_w), int(lm.y * image_h)
            cv2.circle(image, (x_px, y_px), 3, (0, 255, 0), -1)
        current_frame_landmarks = np.array(landmarks)

    # --- LÓGICA DE GRAVAÇÃO ---
    if is_recording:
        sequence_data.append(current_frame_landmarks)
        frame_counter += 1
        
        # Feedback visual de gravação
        cv2.putText(image, f"GRAVANDO: {frame_counter}/30", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if frame_counter >= 30:
            timestamp = int(time.time())
            file_name = DATA_PATH / SIGNS[current_sign_idx] / f"seq_{timestamp}.npy"
            np.save(file_name, np.array(sequence_data))
            print(f"✅ Salvo: {file_name}")
            
            is_recording = False
            frame_counter = 0
            sequence_data = []

    # Interface na tela
    cv2.putText(image, f"SINAL ATUAL: {SIGNS[current_sign_idx]}", (50, image_h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Coletor LIBRAS-SC', image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('1'): current_sign_idx = 0
    elif key == ord('2'): current_sign_idx = 1
    elif key == ord('3'): current_sign_idx = 2
    elif key == ord('s') and not is_recording:
        print(f"🔴 Iniciando gravação de {SIGNS[current_sign_idx]}...")
        is_recording = True

cap.release()
hand_landmarker.close()
cv2.destroyAllWindows()