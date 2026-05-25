import json
import os
from pathlib import Path

SIGNS = [
    "OI", "TCHAU", "EU", "NOME", "OBRIGADO", "SIM", "NAO",
    "POR_FAVOR", "DESCULPA", "BEM", "GOSTAR", "AJUDA",
    "ENTENDER", "NAO_ENTENDER", "REPETIR", "PRAZER", "AMIGO", "SURDO",
    "MELANCIA", "LARANJA",
]

FRAME_WINDOW = 30
COORD_SIZE = 126
HAND_LANDMARKER_MODEL = "hand_landmarker.task"
MODEL_PATH = "modelo_libras.keras"
LEGACY_MODEL_PATH = "modelo_libras.h5"
LABELS_PATH = "labels.json"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tradutor-sc")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
THRESHOLD = float(os.getenv("THRESHOLD", "0.85"))
MOVEMENT_FLOOR = 0.005


def load_labels(n_classes: int | None = None) -> list[str]:
    """Carrega labels do labels.json se existir; senão usa SIGNS com slice opcional."""
    p = Path(LABELS_PATH)
    if p.exists():
        with open(p) as f:
            labels = json.load(f)
        return labels[:n_classes] if n_classes else labels
    return SIGNS[:n_classes] if n_classes else SIGNS
