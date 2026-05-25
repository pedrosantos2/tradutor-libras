import logging

import requests

import config as cfg

logger = logging.getLogger(__name__)


def chamar_ollama(glossas: list) -> str:
    if not glossas:
        return ""
    prompt = f"Converta estas glossas de LIBRAS para português fluído: {glossas}"
    try:
        resp = requests.post(
            cfg.OLLAMA_URL,
            json={"model": cfg.OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        return resp.json().get("response", "").strip()
    except Exception as exc:
        logger.warning("Ollama indisponível (%s) — usando fallback", exc)
        return " ".join(glossas).lower().capitalize() + "."
