import logging
from pathlib import Path

import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


def _load_model():
    from tensorflow.keras.models import load_model as tf_load

    keras_path = Path(cfg.MODEL_PATH)
    h5_path = Path(cfg.LEGACY_MODEL_PATH)
    if keras_path.exists():
        logger.info("Carregando modelo: %s", keras_path)
        return tf_load(str(keras_path))
    if h5_path.exists():
        logger.info("Carregando modelo legado: %s", h5_path)
        return tf_load(str(h5_path))
    raise FileNotFoundError("Nenhum modelo encontrado (modelo_libras.keras ou .h5)")


model = _load_model()
n_classes: int = model.output_shape[-1]
ACTIONS: list[str] = cfg.load_labels(n_classes)

logger.info("Modelo carregado: %d classes → %s", n_classes, ACTIONS)


def predict(sequence: list) -> tuple[str, float]:
    """
    Recebe uma lista de FRAME_WINDOW arrays (126,) e retorna (sinal, confiança).
    Retorna ("", confidence) se abaixo do threshold.
    """
    input_data = np.expand_dims(sequence, axis=0)
    res = model.predict(input_data, verbose=0)[0]
    idx = int(np.argmax(res))
    confidence = float(res[idx])
    if confidence > cfg.THRESHOLD:
        return ACTIONS[idx], confidence
    return "", confidence
