"""
Data models for noise reduction system.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum
import numpy as np


class NoiseType(Enum):
    """Types of noise that can be reduced."""
    LUMINANCE = "luminance"          # Grain-like noise in brightness
    CHROMINANCE = "chrominance"      # Color noise (red/green speckles)
    HOT_PIXEL = "hot_pixel"          # Stuck/hot pixels
    PATTERN = "pattern"              # Fixed pattern noise
    BANDING = "banding"              # Horizontal/vertical banding
    ALL = "all"                      # All noise types


class EdgePreservationMethod(Enum):
    """Methods for preserving edges during noise reduction."""
    BILATERAL = "bilateral"          # Bilateral filtering
    GUIDED = "guided"               # Guided filter
    NLM = "nlm"                     # Non-local means
    ADAPTIVE = "adaptive"           # Adaptive based on content
    WAVELET = "wavelet"            # Wavelet-based


@dataclass
class NoiseProfile:
    """
    Camera and ISO-specific noise characteristics.
    
    Stores noise parameters for different ISO values to enable
    adaptive noise reduction based on shooting conditions.
    """
    camera_model: str
    iso_value: int
    
    # Noise characteristics
    luminance_noise_level: float      # 0-100, typical grain amount
    chrominance_noise_level: float    # 0-100, color noise amount
    
    # Noise frequency characteristics  
    noise_grain_size: float = 1.0     # Relative grain size (0.5-3.0)
    noise_pattern: str = "random"     # random, horizontal, vertical, diagonal
    
    # Color noise specifics
    red_channel_factor: float = 1.0   # Red channel noise multiplier
    blue_channel_factor: float = 1.2  # Blue typically noisier
    
    # High ISO specific issues
    has_banding: bool = False         # Banding at high ISO
    has_pattern_noise: bool = False   # Fixed pattern noise
    hot_pixel_threshold: float = 0.0  # ISO where hot pixels appear
    
    # Optimal reduction parameters (pre-tuned)
    suggested_luminance_reduction: float = 0.0    # 0-100
    suggested_chrominance_reduction: float = 0.0  # 0-100
    suggested_detail_preservation: float = 50.0   # 0-100
    
    def get_auto_settings(self) -> 'NoiseReductionSettings':
        """Generate automatic noise reduction settings for this profile."""
        return NoiseReductionSettings(
            luminance_amount=self.suggested_luminance_reduction,
            chrominance_amount=self.suggested_chrominance_reduction,
            detail_preservation=self.suggested_detail_preservation,
            edge_preservation_method=EdgePreservationMethod.ADAPTIVE,
            iso_adaptive=True
        )


@dataclass
class NoiseReductionSettings:
    """
    User-configurable noise reduction parameters.
    
    These settings control how aggressively noise is reduced
    and how much detail is preserved.
    """
    # Main controls
    luminance_amount: float = 0.0      # 0-100, luminance NR strength
    chrominance_amount: float = 50.0   # 0-100, color NR strength
    detail_preservation: float = 50.0   # 0-100, edge/detail protection
    
    # Advanced controls
    edge_preservation_method: EdgePreservationMethod = EdgePreservationMethod.BILATERAL
    contrast_preservation: float = 80.0  # 0-100, preserve local contrast
    
    # Noise-specific settings
    hot_pixel_removal: bool = True      # Remove hot/stuck pixels
    pattern_suppression: bool = True    # Suppress pattern noise
    banding_reduction: bool = False     # Reduce banding (careful!)
    
    # Processing options
    iso_adaptive: bool = True           # Use ISO-specific profiles
    preserve_grain: bool = False        # Artistic grain preservation
    grain_size_bias: float = 1.0       # 0.5-2.0, affects what's considered noise
    
    # Quality vs speed
    quality_mode: str = "high"          # "fast", "balanced", "high"
    multi_scale: bool = True            # Multi-scale processing
    
    # Region-specific (integrates with local adjustments)
    mask_enabled: bool = False          # Apply only to masked areas
    mask_blend_mode: str = "normal"     # How to blend with non-masked
    
    def scale_by_factor(self, factor: float) -> 'NoiseReductionSettings':
        """Scale noise reduction amounts by a factor."""
        return NoiseReductionSettings(
            luminance_amount=self.luminance_amount * factor,
            chrominance_amount=self.chrominance_amount * factor,
            detail_preservation=self.detail_preservation,
            edge_preservation_method=self.edge_preservation_method,
            contrast_preservation=self.contrast_preservation,
            hot_pixel_removal=self.hot_pixel_removal,
            pattern_suppression=self.pattern_suppression,
            banding_reduction=self.banding_reduction,
            iso_adaptive=self.iso_adaptive,
            preserve_grain=self.preserve_grain,
            grain_size_bias=self.grain_size_bias,
            quality_mode=self.quality_mode,
            multi_scale=self.multi_scale,
            mask_enabled=self.mask_enabled,
            mask_blend_mode=self.mask_blend_mode
        )


@dataclass
class NoiseAnalysis:
    """Results from analyzing image noise characteristics."""
    estimated_luminance_noise: float   # 0-100 scale
    estimated_chrominance_noise: float # 0-100 scale
    
    # Spatial characteristics
    dominant_noise_frequency: str      # "fine", "medium", "coarse"
    noise_uniformity: float           # 0-1, how uniform the noise is
    
    # Problem areas
    hot_pixel_count: int = 0
    pattern_noise_detected: bool = False
    banding_detected: bool = False
    
    # Regional analysis
    noise_map: Optional[np.ndarray] = None  # Spatial noise distribution
    edge_map: Optional[np.ndarray] = None   # Areas needing protection
    
    # Recommendations
    suggested_settings: Optional[NoiseReductionSettings] = None


@dataclass
class FrequencyBand:
    """Represents a frequency band for multi-scale processing."""
    scale: int                    # Scale level (1=finest detail)
    coefficient: np.ndarray       # Wavelet/frequency coefficients
    noise_threshold: float        # Threshold for this scale
    detail_weight: float = 1.0    # Preservation weight


@dataclass 
class NoiseReductionConfig:
    """Configuration for noise reduction processing."""
    # Algorithm selection
    luminance_algorithm: str = "nlm"        # nlm, bilateral, wavelet
    chrominance_algorithm: str = "bilateral" # bilateral, median, guided
    
    # Performance settings
    tile_size: int = 256            # Process in tiles for memory
    overlap: int = 32               # Tile overlap for seamless blending
    num_scales: int = 4             # Number of frequency scales
    
    # Quality settings
    search_window: int = 21         # NLM search window
    patch_size: int = 7             # NLM patch size
    bilateral_sigma_space: float = 10.0
    bilateral_sigma_color: float = 0.1
    
    # Edge detection
    edge_threshold: float = 0.1     # Gradient threshold for edges
    edge_dilation: int = 2          # Dilate edge mask
    
    # Hot pixel detection
    hot_pixel_dev_threshold: float = 5.0  # Std devs for hot pixel
    hot_pixel_kernel_size: int = 5        # Detection kernel
    
    # Pattern noise
    pattern_fft_threshold: float = 0.8    # FFT peak for patterns
    pattern_suppression_factor: float = 0.5