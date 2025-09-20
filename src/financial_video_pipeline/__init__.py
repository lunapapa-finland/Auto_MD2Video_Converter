"""Financial Video Pipeline - Convert financial content to personalized videos."""

__version__ = "0.1.0"

from .pipeline import Pipeline
from .config import Settings, ConfigManager

__all__ = ["Pipeline", "Settings", "ConfigManager"]