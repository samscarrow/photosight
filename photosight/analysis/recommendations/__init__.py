"""
PhotoSight Recommendations Module

Provides intelligent gear recommendations, shooting pattern analysis,
and quality improvement suggestions based on photo metadata analysis.
"""

from .gear_advisor import GearAdvisor
from .shooting_patterns import ShootingPatternAnalyzer
from .quality_trends import QualityTrendAnalyzer

__all__ = [
    'GearAdvisor',
    'ShootingPatternAnalyzer',
    'QualityTrendAnalyzer'
]