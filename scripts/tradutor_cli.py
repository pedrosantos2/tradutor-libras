import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

import config as cfg

# --- CARREGA MODELO ---
_keras = Path(cfg.MODEL_PATH)
_h5 = Path(cfg.LEGACY_MODEL_PATH)
if _keras.exists():
    model_lstm = load_model(str(_keras))
elif _h5.exists():
    model_lstm = load_model(str(_h5))
else:
    raise FileNotFoundError("Nenhum modelo encontrado (modelo_libras.keras ou .h5)")

n_classes = model_lstm.output_shape[-1]
ACTIONS = cfg.load_labels(n_classes)

MODEL_PATH = Path(cfg.HAND_LANDMARKER_MODEL)
COORD_SIZE = cfg.COORD_SIZE
FRAME_WINDOW = cfg.FRAME_WINDOW
threshold = cfg.THRESHOLD
# Desvio padrão mínimo das coordenadas ao longo dos frames (detecta movimento real)
MOVEMENT_THRESHOLD = 0.008
# Quantas predições consecutivas do mesmo sinal são necessárias para confirmar
REQUIRED_CONFIRMATIONS = 4
# Frames de cooldown após confirmar um sinal (evita repetição imediata)
COOLDOWN_FRAMES = 20

# --- FUNÇÃO OLLAMA ---
def chamar_gemma(glossas):
    if not glossas: return ""
    prompt = f"Converta estas glossas de LIBRAS para português fluído: {glossas}"
    try:
        url = "http://localhost:11434/api/generate"
        payload = {"model": "tradutor-sc", "prompt": prompt, "stream": False}
        response = requests.post(url, json=payload)
        return response.json().get('response', "Erro na resposta").strip()
    except Exception as e:
        return f"Erro: {e}"

# --- SETUP MEDIAPIPE ---
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# --- VARIÁVEIS DE ESTADO ---
cap = cv2.VideoCapture(0)
sequence = []
glossas_detectadas = []
last_prediction = ""
candidate_sign = ""
consecutive_count = 0
cooldown_counter = 0

traducao_final_tela = "Aguardando sinais..."

print("\n🚀 SISTEMA RODANDO - LIBRAS-SC")
print("Espaço: Traduzir (Gemma) | C: Limpar Tela | Q: Sair\n")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    image_h, image_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = hand_landmarker.detect(mp_image)

    current_coords = np.zeros(COORD_SIZE)

    if results.hand_landmarks:
        left_pts = [0.0] * 63
        right_pts = [0.0] * 63
        for hand_lms, handedness_list in zip(results.hand_landmarks, results.handedness):
            label = handedness_list[0].category_name  # "Left" or "Right"
            pts = []
            for lm in hand_lms:
                pts.extend([lm.x, lm.y, lm.z])
                x_px, y_px = int(lm.x * image_w), int(lm.y * image_h)
                cv2.circle(image, (x_px, y_px), 3, (0, 255, 0), -1)
            if label == "Left":
                left_pts = pts
            else:
                right_pts = pts
        # Left → slot 0 (0-62), Right → slot 1 (63-125), always consistent
        current_coords[:] = left_pts + right_pts

    sequence.append(current_coords)
    sequence = sequence[-FRAME_WINDOW:]

    if cooldown_counter > 0:
        cooldown_counter -= 1

    if len(sequence) == FRAME_WINDOW:
        seq_array = np.array(sequence)

        # Só prediz se há movimento real nas mãos (filtra mão parada / sem sinal)
        movement = np.std(seq_array[:, :63], axis=0).mean()  # só coordenadas da mão 1
        has_movement = movement > MOVEMENT_THRESHOLD

        if has_movement and cooldown_counter == 0:
            input_data = np.expand_dims(seq_array, axis=0)
            res = model_lstm.predict(input_data, verbose=0)[0]
            idx = np.argmax(res)

            if res[idx] > threshold:
                pred_atual = ACTIONS[idx]
                if pred_atual == candidate_sign:
                    consecutive_count += 1
                else:
                    candidate_sign = pred_atual
                    consecutive_count = 1

                if consecutive_count >= REQUIRED_CONFIRMATIONS and candidate_sign != last_prediction:
                    glossas_detectadas.append(candidate_sign)
                    last_prediction = candidate_sign
                    consecutive_count = 0
                    cooldown_counter = COOLDOWN_FRAMES
            else:
                consecutive_count = 0
                candidate_sign = ""
        elif not has_movement:
            consecutive_count = 0
            candidate_sign = ""
            last_prediction = ""  # reset para aceitar o mesmo sinal após pausa

    # ==========================================
    # INTERFACE VISUAL (HUD)
    # ==========================================
    
    # 1. Barra Inferior (As Glossas Detectadas)
    cv2.rectangle(image, (0, image_h-40), (image_w, image_h), (30, 30, 30), -1)
    texto_glossas = " > ".join(glossas_detectadas)
    cv2.putText(image, f"Sinais: {texto_glossas}", (15, image_h-12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 2. Barra Superior (A Tradução do Gemma)
    cv2.rectangle(image, (0, 0), (image_w, 50), (160, 40, 40), -1) # Fundo Azul Escuro
    cv2.putText(image, f"Gemma: {traducao_final_tela}", (15, 33), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Tradutor Final LIBRAS-SC', image)

    # --- CONTROLES DO TECLADO ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('c'): # Nova tecla para limpar a tela
        glossas_detectadas = []
        last_prediction = ""
        traducao_final_tela = "Aguardando sinais..."
    elif key == ord(' '):
        # Quando apertar espaço, avisa na tela que está processando
        traducao_final_tela = "Processando IA... aguarde."
        cv2.putText(image, f"Gemma: {traducao_final_tela}", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Tradutor Final LIBRAS-SC', image)
        cv2.waitKey(1) # Força atualização da tela
        
        # Chama a IA
        frase = chamar_gemma(glossas_detectadas)
        traducao_final_tela = frase # Atualiza a tela com a frase pronta
        
        # Limpa o buffer de sinais para a próxima frase
        glossas_detectadas = []
        last_prediction = ""

cap.release()
hand_landmarker.close()
cv2.destroyAllWindows()