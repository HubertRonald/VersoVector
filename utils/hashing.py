from __future__ import annotations

__all__ = ["make_short_hash"]

import hashlib


def make_short_hash(value: str, n_chars: int = 12) -> str:
    """
    Genera un hash corto y estable a partir de un string.

    Se usa para construir identificadores técnicos reproducibles
    sin guardar rutas locales ni depender solo de title/poet.
    """
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:n_chars]
