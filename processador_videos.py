import os
import ssl
from pathlib import Path
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import COORD_SIZE, FRAME_WINDOW, MOVEMENT_FLOOR, SIGNS

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

_DETECTION_WINDOW = 5  # tamanho da janela para detectar segmento ativo


def _find_active_segment(frames: np.ndarray) -> tuple[int, int] | None:
    """
    Detecta o trecho do vídeo onde há mão em movimento.
    Retorna (inicio, fim) inclusivos, ou None se não encontrar.
    """
    n = len(frames)
    if n < _DETECTION_WINDOW:
        return None

    hand_present = frames.any(axis=1)  # True onde alguma coord != 0

    # norma L2 da diferença entre frames consecutivos (movimento)
    movement = np.zeros(n)
    movement[1:] = np.linalg.norm(np.diff(frames, axis=0), axis=1)

    def is_active(i: int) -> bool:
        seg = slice(i, min(i + _DETECTION_WINDOW, n))
        return (hand_present[seg].sum() >= _DETECTION_WINDOW - 1 and
                movement[seg].mean() > MOVEMENT_FLOOR)

    # Varredura forward: início do segmento
    start = None
    for i in range(n - _DETECTION_WINDOW + 1):
        if is_active(i):
            start = i
            break

    if start is None:
        return None

    # Varredura backward: fim do segmento
    end = start
    for i in range(n - _DETECTION_WINDOW, start - 1, -1):
        if is_active(i):
            end = min(i + _DETECTION_WINDOW - 1, n - 1)
            break

    if end <= start:
        return None

    return start, end


def _resample(frames: np.ndarray, target: int) -> np.ndarray:
    """Reamostrar uniformemente para exatamente `target` frames."""
    idxs = np.linspace(0, len(frames) - 1, target).astype(int)
    return frames[idxs]


def extrair_coords_do_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    lista_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

    if not lista_frames:
        return None

    frames_arr = np.array(lista_frames)  # (n, 126)

    segment = _find_active_segment(frames_arr)

    if segment is not None:
        start, end = segment
        recorte = frames_arr[start:end + 1]
    else:
        # Fallback: corte geométrico do meio
        n = len(frames_arr)
        meio = n // 2
        inicio = max(0, meio - (FRAME_COUNT // 2))
        recorte = frames_arr[inicio:inicio + FRAME_COUNT]
        if len(recorte) < 2:
            return None
        print(f"    ⚠️  segmento ativo não encontrado, usando corte do meio")

    return _resample(recorte, FRAME_COUNT)

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
print("\n✅ Processamento concluído!")