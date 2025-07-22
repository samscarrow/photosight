"""
RAW processing modules for PhotoSight

Includes scene-aware processing, geometry correction, and non-destructive workflow.
"""

from .scene_classifier import SceneClassifier
from .raw_processor import RawPostProcessor, ProcessingRecipe

__all__ = [
    "SceneClassifier",
    "RawPostProcessor", 
    "ProcessingRecipe",
]