import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config as cfg

logger = logging.getLogger(__name__)


def _build_landmarker() -> vision.HandLandmarker:
    model_path = Path(cfg.HAND_LANDMARKER_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo MediaPipe não encontrado: {model_path}")
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.HandLandmarker.create_from_options(options)


hand_landmarker = _build_landmarker()
logger.info("MediaPipe HandLandmarker carregado")


def detect(frame_bgr: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """
    Processa um frame BGR e retorna (coords, landmarks_ui).
    - coords: shape (126,) — zeros se mão não detectada
    - landmarks_ui: lista de {"x": float, "y": float} para overlay no frontend
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = hand_landmarker.detect(mp_image)

    coords = np.zeros(cfg.COORD_SIZE)
    landmarks_ui: list[dict] = []

    if results.hand_landmarks:
        left_pts = [0.0] * 63
        right_pts = [0.0] * 63
        for hand_lms, handedness_list in zip(results.hand_landmarks, results.handedness):
            label = handedness_list[0].category_name
            pts: list[float] = []
            for lm in hand_lms:
                pts.extend([lm.x, lm.y, lm.z])
                landmarks_ui.append({"x": lm.x, "y": lm.y})
            if label == "Left":
                left_pts = pts
            else:
                right_pts = pts
        # Left → slot 0-62, Right → slot 63-125 (sempre consistente)
        coords[:] = left_pts + right_pts

    return coords, landmarks_ui
