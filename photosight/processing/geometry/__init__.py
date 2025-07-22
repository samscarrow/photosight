"""
Geometry processing modules for PhotoSight

Includes horizon detection, automatic straightening, and intelligent cropping.
"""

from .horizon_detector import HorizonDetector
from .auto_straighten import AutoStraightener
from .smart_crop import SmartCropper, CropSuggestion

__all__ = [
    "HorizonDetector",
    "AutoStraightener",
    "SmartCropper",
    "CropSuggestion",
]