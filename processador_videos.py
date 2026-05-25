import os
import ssl
from pathlib import Path
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import COORD_SIZE, FRAME_WINDOW, SIGNS

# --- CONFIGURAÇÕES ---
INPUT_BASE_FOLDER = Path("videos_baixados")
OUTPUT_BASE_FOLDER = Path("DATA")
FRAME_COUNT = FRAME_WINDOW

MODEL_PATH = Path("hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# --- GARANTIR O MODELO ---
if not MODEL_PATH.exists():
    print("Baixando modelo hand_landmarker.task...")
    context = ssl._create_unverified_context()
    with urlopen(MODEL_URL, timeout=30, context=context) as response:
        MODEL_PATH.write_bytes(response.read())

# --- SETUP MEDIAPIPE TASKS ---
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5 # Menor para aceitar vídeos de internet
)
detector = vision.HandLandmarker.create_from_options(options)

def extrair_coords_do_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    lista_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Processamento
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect(mp_image)
        
        coords = np.zeros(COORD_SIZE)
        if results.hand_landmarks:
            left_pts = [0.0] * 63
            right_pts = [0.0] * 63
            for hand_lms, handedness_list in zip(results.hand_landmarks, results.handedness):
                label = handedness_list[0].category_name
                pts = []
                for lm in hand_lms:
                    pts.extend([lm.x, lm.y, lm.z])
                if label == "Left":
                    left_pts = pts
                else:
                    right_pts = pts
            coords[:] = left_pts + right_pts
        
        lista_frames.append(coords)
    
    cap.release()
    
    # --- NORMALIZAÇÃO INTELIGENTE (O CORTE DO MEIO) ---
    total_frames = len(lista_frames)
    
    if total_frames == 0:
        return None

    if total_frames > FRAME_COUNT:
        # Acha o meio exato do vídeo e pega 15 frames pra trás e 15 pra frente
        meio = total_frames // 2
        inicio = max(0, meio - (FRAME_COUNT // 2))
        fim = inicio + FRAME_COUNT
        
        recorte = lista_frames[inicio:fim]
        
        # Só por segurança, se faltar algum frame no final da matemática, ele repete
        while len(recorte) < FRAME_COUNT:
            recorte.append(recorte[-1])
            
        return np.array(recorte)
    else:
        # Se o vídeo for super curto (menos de 30 frames), repete o último frame até dar 30
        while len(lista_frames) < FRAME_COUNT:
            lista_frames.append(lista_frames[-1])
        return np.array(lista_frames)

# --- LOOP PRINCIPAL PELAS PASTAS ---
print("\n🔄 Iniciando processamento de vídeos...\n")

for sign in SIGNS:
    input_dir = INPUT_BASE_FOLDER / sign
    output_dir = OUTPUT_BASE_FOLDER / sign
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"⚠️ Pasta não encontrada: {input_dir}")
        continue
        
    videos = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    print(f"📁 Processando {len(videos)} vídeos de: {sign}")
    
    for v_name in videos:
        v_path = input_dir / v_name
        try:
            dados = extrair_coords_do_video(v_path)
            if dados is not None:
                save_path = output_dir / f"ext_{v_name}.npy"
                np.save(save_path, dados)
        except Exception as e:
            print(f"❌ Erro ao processar {v_name}: {e}")

detector.close()
print("\n✅ Processamento concluído! O 'miolo' da ação foi extraído com sucesso.")