"""TTS package facade.

Exposes a simple factory create_tts(settings) that returns the appropriate
generator implementation based on settings.tts.engine:
 - 'edge'  -> EdgeTTSGenerator (edge-tts)
 - 'piper' -> PiperTTSGenerator (local Piper models)
"""

from __future__ import annotations

import logging

from ..config import Settings

logger = logging.getLogger(__name__)

# Import implementations
try:
    from .piper_generator import PiperTTSGenerator  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency path
    PiperTTSGenerator = None  # type: ignore
    logger.debug(f"Piper generator not available: {e}")

try:
    from .generator import TTSGenerator as EdgeTTSGenerator  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency path
    EdgeTTSGenerator = None  # type: ignore
    logger.debug(f"Edge generator not available: {e}")


def create_tts(settings: Settings):
    """Create a TTS generator based on settings.tts.engine or alias default_tts.

    Precedence:
      1. settings.tts.engine (if set)
      2. settings.tts.default_tts (alias)
      3. default 'edge'
    """
    engine = getattr(settings.tts, "engine", None)
    if not engine:
        engine = getattr(settings.tts, "default_tts", None)
    engine = (engine or "edge").lower()
    if engine == "edge":
        if EdgeTTSGenerator is not None:
            return EdgeTTSGenerator(settings)
        logger.warning("Edge TTS unavailable; falling back to Piper.")
        if PiperTTSGenerator is not None:
            return PiperTTSGenerator(settings)
        raise RuntimeError("No TTS engine available: edge and piper unavailable")
    # Piper as default fallback for unknown engines
    if PiperTTSGenerator is not None:
        return PiperTTSGenerator(settings)
    if EdgeTTSGenerator is not None:
        logger.warning("Piper TTS unavailable; using Edge instead.")
        return EdgeTTSGenerator(settings)
    raise RuntimeError("No TTS engine available: edge and piper unavailable")


__all__ = ["create_tts"]


__all__ = [
    "PiperTTSGenerator",
    "EdgeTTSGenerator",
    "create_tts",
]


# Attempt to import Edge TTS generator (edge-tts based)
try:
    from .generator import TTSGenerator as EdgeTTSGenerator  # type: ignore
except Exception:  # edge_tts may not be installed in this environment
    EdgeTTSGenerator = None  # type: ignore


def create_tts(settings: Settings):
    """Factory to create a TTS generator based on settings.tts.engine.

    - 'edge': use EdgeTTSGenerator (edge-tts)
    - 'piper': use PiperTTSGenerator (local Piper models)
    """
    engine = getattr(settings.tts, "engine", "edge").lower()
    if engine == "edge":
        if EdgeTTSGenerator is None:
            logger.warning("Edge TTS not available; falling back to Piper TTS")
            return PiperTTSGenerator(settings)
        return EdgeTTSGenerator(settings)
    # default to Piper
    return PiperTTSGenerator(settings)


__all__ = [
    "PiperTTSGenerator",
    "EdgeTTSGenerator",
    "create_tts",
]
