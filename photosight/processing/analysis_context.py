"""
Analysis Context for PhotoSight.

Provides a unified, standardized data pipeline with lazy computation and caching
for image analysis operations. Eliminates redundant processing and ensures
consistent data types across all analyzers.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import logging
from functools import cached_property

logger = logging.getLogger(__name__)


class AnalysisContext:
    """
    Unified context for image analysis with lazy computation and caching.
    
    This class serves as the single source of truth for image data during
    analysis, providing standardized representations and eliminating redundant
    computations across multiple analyzers.
    
    Key principles:
    - Standard internal format: float32 arrays with values in [0.0, 1.0] range
    - Lazy computation: Derived representations computed only when requested
    - Caching: Expensive operations cached for subsequent requests
    - Type safety: Each analyzer gets exactly the data type it needs
    """
    
    def __init__(self, image_array: np.ndarray, photo_path: Optional[str] = None):
        """
        Initialize analysis context with source image.
        
        Args:
            image_array: Source image array (any dtype, will be standardized)
            photo_path: Optional path to the source image file
        """
        # Standardize to float32 [0.0, 1.0] range - our single source of truth
        self._standard_image = self._standardize_image(image_array)
        self._shape = self._standard_image.shape
        self._height, self._width = self._shape[:2]
        self._channels = self._shape[2] if len(self._shape) == 3 else 1
        self._photo_path = photo_path
        
        # Analysis results from external analyzers
        self._emotional_impact_score = None  # Will be set by Vision LLM if available
        
        # Cache for derived representations
        self._cache: Dict[str, Any] = {}
        
        logger.debug(f"Initialized AnalysisContext: {self._width}x{self._height}, {self._channels} channels, path: {photo_path}")
    
    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._width
    
    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._height
    
    @property
    def channels(self) -> int:
        """Number of color channels."""
        return self._channels
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Image shape tuple."""
        return self._shape
    
    @property
    def photo_path(self) -> Optional[str]:
        """Path to the source image file."""
        return self._photo_path
    
    @property
    def emotional_impact_score(self) -> Optional[float]:
        """
        Emotional impact score from Vision LLM analysis.
        
        Returns None if not yet analyzed, otherwise a score between 0.0 and 1.0.
        """
        return self._emotional_impact_score
    
    @emotional_impact_score.setter
    def emotional_impact_score(self, value: Optional[float]):
        """
        Set emotional impact score from Vision LLM analysis.
        
        Args:
            value: Score between 0.0 and 1.0, or None
        """
        if value is not None:
            # Ensure value is in valid range
            self._emotional_impact_score = max(0.0, min(1.0, float(value)))
        else:
            self._emotional_impact_score = None
    
    @property
    def standard_image(self) -> np.ndarray:
        """
        Standard internal representation: float32 array with values in [0.0, 1.0].
        
        This is the single source of truth for image data. All other representations
        are derived from this standard format.
        """
        return self._standard_image
    
    @cached_property
    def uint8_image(self) -> np.ndarray:
        """
        Image as uint8 array with values in [0, 255] range.
        
        Commonly needed for OpenCV functions and display operations.
        """
        return (self._standard_image * 255.0).astype(np.uint8)
    
    @cached_property
    def grayscale(self) -> np.ndarray:
        """
        Grayscale version as float32 array with values in [0.0, 1.0].
        
        Uses OpenCV's standard RGB to grayscale conversion weights.
        """
        if self._channels == 1:
            return self._standard_image.squeeze()
        else:
            # Convert RGB to grayscale using standard weights
            return cv2.cvtColor(self._standard_image, cv2.COLOR_RGB2GRAY)
    
    @cached_property
    def grayscale_uint8(self) -> np.ndarray:
        """Grayscale version as uint8 array with values in [0, 255]."""
        return (self.grayscale * 255.0).astype(np.uint8)
    
    @cached_property
    def lab_image(self) -> np.ndarray:
        """
        Image in LAB color space as float32 array.
        
        LAB is perceptually uniform and useful for color analysis.
        Values are in OpenCV's LAB range: L[0,100], a[-127,127], b[-127,127]
        """
        if self._channels == 1:
            # Convert grayscale to 3-channel first
            rgb_for_lab = cv2.cvtColor(self.grayscale_uint8, cv2.COLOR_GRAY2RGB)
        else:
            rgb_for_lab = self.uint8_image
        
        lab = cv2.cvtColor(rgb_for_lab, cv2.COLOR_RGB2LAB)
        return lab.astype(np.float32)
    
    @cached_property
    def hsv_image(self) -> np.ndarray:
        """
        Image in HSV color space as float32 array.
        
        HSV is intuitive for color-based analysis.
        Values are in OpenCV's HSV range: H[0,180], S[0,255], V[0,255]
        """
        if self._channels == 1:
            # Convert grayscale to 3-channel first
            rgb_for_hsv = cv2.cvtColor(self.grayscale_uint8, cv2.COLOR_GRAY2RGB)
        else:
            rgb_for_hsv = self.uint8_image
        
        hsv = cv2.cvtColor(rgb_for_hsv, cv2.COLOR_RGB2HSV)
        return hsv.astype(np.float32)
    
    def get_edge_map(self, low_threshold: float = 50.0, high_threshold: float = 150.0) -> np.ndarray:
        """
        Compute edge map using Canny edge detection.
        
        Args:
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            
        Returns:
            Binary edge map as float32 array with values in [0.0, 1.0]
        """
        cache_key = f"edges_{low_threshold}_{high_threshold}"
        
        if cache_key not in self._cache:
            edges = cv2.Canny(self.grayscale_uint8, low_threshold, high_threshold)
            # Convert to float32 [0.0, 1.0] range
            self._cache[cache_key] = edges.astype(np.float32) / 255.0
        
        return self._cache[cache_key]
    
    def get_histogram(self, bins: int = 256, channel: Optional[int] = None) -> np.ndarray:
        """
        Compute intensity histogram.
        
        Args:
            bins: Number of histogram bins
            channel: Specific channel to analyze (None for all channels)
            
        Returns:
            Normalized histogram as float32 array
        """
        cache_key = f"hist_{bins}_{channel}"
        
        if cache_key not in self._cache:
            if channel is None:
                # All channels combined
                hist_data = self.grayscale
            else:
                # Specific channel
                if self._channels == 1:
                    hist_data = self.grayscale
                else:
                    hist_data = self._standard_image[:, :, channel]
            
            # Compute histogram
            hist, _ = np.histogram(hist_data.flatten(), bins=bins, range=(0.0, 1.0))
            
            # Normalize
            hist_normalized = hist.astype(np.float32)
            if hist_normalized.sum() > 0:
                hist_normalized = hist_normalized / hist_normalized.sum()
            
            self._cache[cache_key] = hist_normalized
        
        return self._cache[cache_key]
    
    def get_gradient_magnitude(self) -> np.ndarray:
        """
        Compute gradient magnitude using Sobel filters.
        
        Returns:
            Gradient magnitude as float32 array with values in [0.0, 1.0+]
        """
        cache_key = "gradient_magnitude"
        
        if cache_key not in self._cache:
            gray = self.grayscale_uint8
            
            # Compute gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute magnitude
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize to [0.0, 1.0] range
            if magnitude.max() > 0:
                magnitude = magnitude / magnitude.max()
            
            self._cache[cache_key] = magnitude.astype(np.float32)
        
        return self._cache[cache_key]
    
    def get_luminance(self) -> np.ndarray:
        """
        Compute perceptual luminance using standard weights.
        
        Returns:
            Luminance as float32 array with values in [0.0, 1.0]
        """
        cache_key = "luminance"
        
        if cache_key not in self._cache:
            if self._channels == 1:
                luminance = self.grayscale
            else:
                # Use ITU-R BT.709 standard weights for RGB to luminance
                weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
                luminance = np.dot(self._standard_image, weights)
            
            self._cache[cache_key] = luminance
        
        return self._cache[cache_key]
    
    def get_color_moments(self) -> Dict[str, np.ndarray]:
        """
        Compute color moments (mean, variance, skewness) for each channel.
        
        Returns:
            Dictionary with 'mean', 'variance', 'skewness' arrays
        """
        cache_key = "color_moments"
        
        if cache_key not in self._cache:
            if self._channels == 1:
                data = self.grayscale.reshape(-1, 1)
            else:
                data = self._standard_image.reshape(-1, self._channels)
            
            # Use int64 to prevent overflow in calculations
            data_int64 = (data * 255).astype(np.int64)
            
            mean = np.mean(data_int64, axis=0).astype(np.float32) / 255.0
            variance = np.var(data_int64, axis=0).astype(np.float32) / (255.0**2)
            
            # Compute skewness
            centered = data_int64 - np.mean(data_int64, axis=0)
            skewness = np.mean(centered**3, axis=0).astype(np.float32) / (np.var(data_int64, axis=0)**1.5 + 1e-8)
            skewness = skewness / (255.0**3)
            
            self._cache[cache_key] = {
                'mean': mean,
                'variance': variance,
                'skewness': skewness
            }
        
        return self._cache[cache_key]
    
    def get_for_kmeans(self, n_colors: int = 5) -> Tuple[np.ndarray, int]:
        """
        Prepare image data for K-means clustering.
        
        Args:
            n_colors: Number of color clusters
            
        Returns:
            Tuple of (reshaped_data, total_pixels) ready for cv2.kmeans
        """
        cache_key = f"kmeans_data_{n_colors}"
        
        if cache_key not in self._cache:
            if self._channels == 1:
                # For grayscale, create 3-channel data
                data = np.stack([self.grayscale] * 3, axis=2)
            else:
                data = self._standard_image
            
            # Reshape to 2D array of pixels: [N_pixels, 3_channels]
            pixels = data.reshape(-1, 3)
            
            # Convert to float32 [0, 255] range as expected by cv2.kmeans
            pixels_kmeans = (pixels * 255.0).astype(np.float32)
            
            self._cache[cache_key] = (pixels_kmeans, len(pixels_kmeans))
        
        return self._cache[cache_key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data for debugging."""
        return {
            'cached_items': list(self._cache.keys()),
            'cache_size': len(self._cache),
            'image_size': f"{self._width}x{self._height}",
            'channels': self._channels
        }
    
    def _standardize_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Convert input image to standard internal format.
        
        Args:
            image_array: Input image array (any dtype)
            
        Returns:
            Standardized float32 array with values in [0.0, 1.0]
        """
        # Convert to float32
        if image_array.dtype == np.uint8:
            # Convert from [0, 255] to [0.0, 1.0]
            standardized = image_array.astype(np.float32) / 255.0
        elif image_array.dtype == np.uint16:
            # Convert from [0, 65535] to [0.0, 1.0]
            standardized = image_array.astype(np.float32) / 65535.0
        elif image_array.dtype in [np.float32, np.float64]:
            standardized = image_array.astype(np.float32)
            # Ensure values are in [0.0, 1.0] range
            if standardized.max() > 1.0:
                standardized = standardized / standardized.max()
        else:
            # Fallback: normalize to [0.0, 1.0] range
            standardized = image_array.astype(np.float32)
            if standardized.max() > standardized.min():
                standardized = (standardized - standardized.min()) / (standardized.max() - standardized.min())
        
        # Ensure values are clipped to valid range
        standardized = np.clip(standardized, 0.0, 1.0)
        
        return standardized