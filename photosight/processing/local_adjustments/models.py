"""
Data models for the Local Adjustments framework.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum
import numpy as np


class MaskType(Enum):
    """Available mask generation strategies."""
    RADIAL = "radial"
    LINEAR = "linear"
    LUMINOSITY = "luminosity"
    COLOR_RANGE = "color"
    MANUAL = "manual"  # For future brush/path support


@dataclass
class RadialMaskData:
    """Parameters for radial gradient masks."""
    center: Tuple[float, float]  # Normalized coordinates (0-1)
    radius: Tuple[float, float]  # (rx, ry) for elliptical shapes, normalized
    feather: float = 0.5  # 0-1, controls gradient softness
    invert: bool = False  # True to affect outside the circle


@dataclass
class LinearMaskData:
    """Parameters for linear gradient masks."""
    start_point: Tuple[float, float]  # Normalized coordinates (0-1)
    end_point: Tuple[float, float]    # Normalized coordinates (0-1)
    feather: float = 0.5  # 0-1, controls gradient length
    invert: bool = False  # True to invert the gradient direction


@dataclass
class LuminosityMaskData:
    """Parameters for luminosity-based masks."""
    range: Tuple[float, float]  # (min, max) luminance values (0-1)
    feather: float = 0.1  # Smoothness at range boundaries
    invert: bool = False  # True to select outside the range


@dataclass
class ColorRangeMaskData:
    """Parameters for color range masks."""
    target_hue: float  # 0-360 degrees
    hue_range: float = 30.0  # Tolerance in degrees
    saturation_range: Tuple[float, float] = (0.0, 1.0)  # Min/max saturation
    value_range: Tuple[float, float] = (0.0, 1.0)  # Min/max value
    feather: float = 0.1  # Edge softness
    invert: bool = False


@dataclass
class MaskData:
    """Container for mask generation parameters."""
    type: MaskType
    parameters: Dict[str, Any]  # Type-specific parameters
    
    # Optional mask combination
    blend_mode: str = "normal"  # normal, multiply, screen, etc.
    opacity: float = 1.0  # 0-1, overall mask strength
    
    @classmethod
    def radial(cls, center: Tuple[float, float], radius: Tuple[float, float], 
               feather: float = 0.5, invert: bool = False) -> 'MaskData':
        """Create radial mask data."""
        return cls(
            type=MaskType.RADIAL,
            parameters={
                'center': center,
                'radius': radius,
                'feather': feather,
                'invert': invert
            }
        )
    
    @classmethod
    def linear(cls, start: Tuple[float, float], end: Tuple[float, float],
               feather: float = 0.5, invert: bool = False) -> 'MaskData':
        """Create linear gradient mask data."""
        return cls(
            type=MaskType.LINEAR,
            parameters={
                'start_point': start,
                'end_point': end,
                'feather': feather,
                'invert': invert
            }
        )
    
    @classmethod
    def luminosity(cls, min_lum: float, max_lum: float, 
                   feather: float = 0.1, invert: bool = False) -> 'MaskData':
        """Create luminosity range mask data."""
        return cls(
            type=MaskType.LUMINOSITY,
            parameters={
                'range': (min_lum, max_lum),
                'feather': feather,
                'invert': invert
            }
        )
    
    @classmethod
    def color_range(cls, target_hue: float, hue_range: float = 30.0,
                    saturation_range: Tuple[float, float] = (0.0, 1.0),
                    value_range: Tuple[float, float] = (0.0, 1.0),
                    feather: float = 0.1, invert: bool = False) -> 'MaskData':
        """Create color range mask data."""
        return cls(
            type=MaskType.COLOR_RANGE,
            parameters={
                'target_hue': target_hue,
                'hue_range': hue_range,
                'saturation_range': saturation_range,
                'value_range': value_range,
                'feather': feather,
                'invert': invert
            }
        )


@dataclass
class AdjustmentLayer:
    """A single adjustment layer with mask and adjustments."""
    name: str
    enabled: bool = True
    mask_data: MaskData = None
    adjustments: Dict[str, float] = field(default_factory=dict)
    
    # Layer properties
    blend_mode: str = "normal"  # How this layer blends with previous
    opacity: float = 1.0  # Overall layer opacity (0-1)
    
    # Optional mask refinements
    mask_blur: float = 0.0  # Gaussian blur radius for mask softening
    mask_contrast: float = 1.0  # Adjust mask contrast
    
    def get_adjustment_value(self, key: str, default: float = 0.0) -> float:
        """Get adjustment value with default."""
        return self.adjustments.get(key, default)
    
    def has_adjustments(self) -> bool:
        """Check if layer has any non-zero adjustments."""
        return any(abs(v) > 0.001 for v in self.adjustments.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'name': self.name,
            'enabled': self.enabled,
            'adjustments': self.adjustments,
            'blend_mode': self.blend_mode,
            'opacity': self.opacity,
            'mask_blur': self.mask_blur,
            'mask_contrast': self.mask_contrast
        }
        
        if self.mask_data:
            result['mask_data'] = {
                'type': self.mask_data.type.value,
                'parameters': self.mask_data.parameters,
                'blend_mode': self.mask_data.blend_mode,
                'opacity': self.mask_data.opacity
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdjustmentLayer':
        """Create from dictionary (JSON deserialization)."""
        mask_data = None
        if 'mask_data' in data and data['mask_data']:
            mask_dict = data['mask_data']
            mask_data = MaskData(
                type=MaskType(mask_dict['type']),
                parameters=mask_dict['parameters'],
                blend_mode=mask_dict.get('blend_mode', 'normal'),
                opacity=mask_dict.get('opacity', 1.0)
            )
        
        return cls(
            name=data['name'],
            enabled=data.get('enabled', True),
            mask_data=mask_data,
            adjustments=data.get('adjustments', {}),
            blend_mode=data.get('blend_mode', 'normal'),
            opacity=data.get('opacity', 1.0),
            mask_blur=data.get('mask_blur', 0.0),
            mask_contrast=data.get('mask_contrast', 1.0)
        )
    
    def __post_init__(self):
        """Validate adjustment values."""
        # Common adjustment keys and their valid ranges
        valid_ranges = {
            'exposure': (-5.0, 5.0),
            'contrast': (-100.0, 100.0),
            'highlights': (-100.0, 100.0),
            'shadows': (-100.0, 100.0),
            'whites': (-100.0, 100.0),
            'blacks': (-100.0, 100.0),
            'clarity': (-100.0, 100.0),
            'vibrance': (-100.0, 100.0),
            'saturation': (-100.0, 100.0),
            'temperature': (-100.0, 100.0),
            'tint': (-100.0, 100.0),
            'sharpness': (0.0, 300.0),
            'noise_reduction': (0.0, 100.0),
            'dehaze': (-100.0, 100.0)
        }
        
        for key, value in self.adjustments.items():
            if key in valid_ranges:
                min_val, max_val = valid_ranges[key]
                if not min_val <= value <= max_val:
                    raise ValueError(
                        f"Adjustment '{key}' value {value} out of range [{min_val}, {max_val}]"
                    )


@dataclass
class LocalAdjustmentConfig:
    """Configuration for local adjustment processing."""
    # Performance settings
    mask_resolution_factor: float = 1.0  # Reduce mask resolution for speed
    enable_gpu: bool = False  # Use GPU for mask generation if available
    cache_masks: bool = True  # Cache generated masks
    
    # Quality settings
    mask_precision: np.dtype = np.float32  # float16 for speed, float32 for quality
    blend_precision: np.dtype = np.float32
    
    # Debug settings
    save_masks: bool = False  # Save mask images for debugging
    mask_save_path: Optional[str] = None