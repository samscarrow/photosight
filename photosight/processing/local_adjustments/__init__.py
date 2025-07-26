"""
Local Adjustments Framework for PhotoSight

Provides non-destructive, layer-based local adjustments with sophisticated masking.
"""

from .models import AdjustmentLayer, MaskType, MaskData, LocalAdjustmentConfig
from .mask_generator import MaskGenerator
from .adjustment_processor import LocalAdjustmentProcessor

__all__ = [
    'AdjustmentLayer',
    'MaskType', 
    'MaskData',
    'LocalAdjustmentConfig',
    'MaskGenerator',
    'LocalAdjustmentProcessor'
]