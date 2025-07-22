"""
PhotoSight: Intelligent RAW photo processing pipeline

A comprehensive system for automatically processing and curating RAW photos
with scene-aware analysis, AI-powered curation, and non-destructive workflow.
"""

__version__ = "0.1.0"
__author__ = "Sam Scarrow"
__email__ = "sam@example.com"

# Core imports for easy access
from .config import load_config
# from .processing.scene_classifier import SceneClassifier
# from .processing.raw_processor import RawPostProcessor, ProcessingRecipe
# from .analysis.improved_blur_detection import ImprovedBlurDetector
# from .analysis.ai.curator import AICurator

__all__ = [
    "load_config",
]
