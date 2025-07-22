"""
Geometry processing modules for PhotoSight

Includes horizon detection, automatic straightening, and intelligent cropping.
"""

from .horizon_detector import HorizonDetector
from .auto_straighten import AutoStraightener

__all__ = [
    "HorizonDetector",
    "AutoStraightener",
]