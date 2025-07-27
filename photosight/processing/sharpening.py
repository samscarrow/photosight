"""
Sharpening pipeline for PhotoSight.

Implements three-stage sharpening:
1. Input sharpening - Deconvolution to restore detail lost in capture
2. Creative sharpening - Local contrast enhancement for artistic effect
3. Output sharpening - Final sharpening optimized for display medium
"""

import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from typing import Tuple, Optional, Literal
import cv2
from dataclasses import dataclass


@dataclass
class InputSharpeningSettings:
    """Settings for input sharpening (deconvolution)."""
    enabled: bool = True
    radius: float = 0.8  # Radius of point spread function
    amount: float = 50.0  # Strength (0-100)
    iterations: int = 10  # Richardson-Lucy iterations
    threshold: float = 0.0  # Noise threshold


@dataclass
class CreativeSharpeningSettings:
    """Settings for creative sharpening (local contrast)."""
    enabled: bool = True
    radius: float = 1.5  # Radius for unsharp mask
    amount: float = 100.0  # Strength (0-300)
    threshold: float = 2.0  # Minimum edge strength
    clarity_amount: float = 20.0  # Local contrast enhancement (0-100)
    clarity_radius: float = 50.0  # Radius for clarity effect
    mask_edges: bool = True  # Apply only to edges


@dataclass
class OutputSharpeningSettings:
    """Settings for output sharpening."""
    enabled: bool = True
    medium: Literal["screen", "print", "web"] = "screen"
    amount: float = 50.0  # Strength (0-100)
    radius: float = 0.5  # Radius based on output size
    print_dpi: Optional[int] = 300  # DPI for print output


@dataclass
class SharpeningSettings:
    """Complete sharpening settings for all stages."""
    input_sharpening: InputSharpeningSettings = InputSharpeningSettings()
    creative_sharpening: CreativeSharpeningSettings = CreativeSharpeningSettings()
    output_sharpening: OutputSharpeningSettings = OutputSharpeningSettings()


class SharpeningPipeline:
    """Three-stage sharpening pipeline."""
    
    def __init__(self, settings: SharpeningSettings):
        self.settings = settings
    
    def apply_input_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Apply input sharpening using Richardson-Lucy deconvolution.
        
        This compensates for the inherent softness in digital capture
        and lens diffraction.
        """
        if not self.settings.input_sharpening.enabled:
            return image
        
        settings = self.settings.input_sharpening
        
        # Create point spread function (PSF)
        psf = self._create_gaussian_psf(settings.radius)
        
        # Apply Richardson-Lucy deconvolution
        result = self._richardson_lucy_deconvolution(
            image, psf, 
            iterations=settings.iterations,
            threshold=settings.threshold
        )
        
        # Blend with original based on amount
        alpha = settings.amount / 100.0
        return alpha * result + (1 - alpha) * image
    
    def apply_creative_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Apply creative sharpening for artistic enhancement.
        
        Combines unsharp masking with local contrast enhancement (clarity).
        """
        if not self.settings.creative_sharpening.enabled:
            return image
        
        settings = self.settings.creative_sharpening
        
        # Apply clarity (local contrast enhancement) first
        if settings.clarity_amount > 0:
            image = self._apply_clarity(
                image,
                amount=settings.clarity_amount,
                radius=settings.clarity_radius
            )
        
        # Create edge mask if requested
        if settings.mask_edges:
            edge_mask = self._create_edge_mask(image, settings.threshold)
        else:
            edge_mask = np.ones_like(image[:, :, 0])
        
        # Apply unsharp masking
        result = self._unsharp_mask(
            image,
            radius=settings.radius,
            amount=settings.amount,
            threshold=settings.threshold
        )
        
        # Apply edge mask
        for c in range(image.shape[2]):
            result[:, :, c] = (
                result[:, :, c] * edge_mask + 
                image[:, :, c] * (1 - edge_mask)
            )
        
        return result
    
    def apply_output_sharpening(self, image: np.ndarray, 
                              output_size: Tuple[int, int]) -> np.ndarray:
        """
        Apply output sharpening optimized for display medium.
        
        Parameters are automatically adjusted based on output size
        and display medium.
        """
        if not self.settings.output_sharpening.enabled:
            return image
        
        settings = self.settings.output_sharpening
        
        # Calculate optimal radius based on output size and medium
        radius = self._calculate_output_radius(
            output_size, 
            settings.medium,
            settings.print_dpi
        )
        
        # Adjust amount based on medium
        amount = self._adjust_amount_for_medium(
            settings.amount,
            settings.medium
        )
        
        # Apply targeted sharpening
        return self._unsharp_mask(
            image,
            radius=radius,
            amount=amount,
            threshold=0.0  # No threshold for output sharpening
        )
    
    def _create_gaussian_psf(self, radius: float) -> np.ndarray:
        """Create Gaussian point spread function."""
        size = int(radius * 4) | 1  # Ensure odd size
        psf = np.zeros((size, size))
        center = size // 2
        
        sigma = radius / 2.355  # FWHM to sigma conversion
        
        for i in range(size):
            for j in range(size):
                dist_sq = (i - center) ** 2 + (j - center) ** 2
                psf[i, j] = np.exp(-dist_sq / (2 * sigma ** 2))
        
        return psf / psf.sum()
    
    def _richardson_lucy_deconvolution(self, image: np.ndarray, psf: np.ndarray,
                                     iterations: int = 10, 
                                     threshold: float = 0.0) -> np.ndarray:
        """
        Richardson-Lucy deconvolution algorithm.
        
        Iteratively deconvolves the image to restore sharpness.
        """
        # Initialize with input image
        estimate = image.copy()
        psf_flipped = np.flip(psf)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-12
        
        for _ in range(iterations):
            # Forward convolution
            convolved = ndimage.convolve(estimate, psf, mode='reflect')
            
            # Calculate ratio
            ratio = image / (convolved + eps)
            
            # Apply threshold to reduce noise amplification
            if threshold > 0:
                mask = np.abs(ratio - 1.0) > threshold / 100.0
                ratio = np.where(mask, ratio, 1.0)
            
            # Backward convolution and update
            correction = ndimage.convolve(ratio, psf_flipped, mode='reflect')
            estimate *= correction
            
            # Clamp values
            estimate = np.clip(estimate, 0, 1)
        
        return estimate
    
    def _apply_clarity(self, image: np.ndarray, amount: float, 
                      radius: float) -> np.ndarray:
        """Apply clarity (local contrast enhancement)."""
        # Create low-frequency and high-frequency components
        sigma_low = radius
        sigma_high = radius / 3
        
        # Multi-scale approach
        low_freq = cv2.GaussianBlur(image, (0, 0), sigma_low)
        mid_freq = cv2.GaussianBlur(image, (0, 0), sigma_high)
        
        # Calculate local contrast mask
        local_contrast = mid_freq - low_freq
        
        # Apply S-curve to contrast mask for more natural look
        contrast_enhanced = self._apply_s_curve(local_contrast, strength=0.5)
        
        # Blend with original
        alpha = amount / 100.0
        result = image + alpha * contrast_enhanced
        
        return np.clip(result, 0, 1)
    
    def _unsharp_mask(self, image: np.ndarray, radius: float, 
                     amount: float, threshold: float) -> np.ndarray:
        """Apply unsharp masking."""
        # Create blurred version
        sigma = radius
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Calculate mask
        mask = image - blurred
        
        # Apply threshold
        if threshold > 0:
            mask_magnitude = np.sqrt(np.sum(mask ** 2, axis=2))
            threshold_mask = mask_magnitude > (threshold / 255.0)
            threshold_mask = threshold_mask[:, :, np.newaxis]
            mask = mask * threshold_mask
        
        # Apply sharpening
        sharpened = image + (amount / 100.0) * mask
        
        return np.clip(sharpened, 0, 1)
    
    def _create_edge_mask(self, image: np.ndarray, threshold: float) -> np.ndarray:
        """Create edge mask for selective sharpening."""
        # Convert to grayscale
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Detect edges using Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        
        # Normalize and threshold
        edges = edges / edges.max()
        edge_mask = edges > (threshold / 100.0)
        
        # Dilate mask slightly
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1)
        
        # Smooth the mask
        edge_mask = cv2.GaussianBlur(edge_mask.astype(float), (5, 5), 1.0)
        
        return edge_mask
    
    def _calculate_output_radius(self, output_size: Tuple[int, int], 
                               medium: str, print_dpi: Optional[int]) -> float:
        """Calculate optimal sharpening radius for output."""
        width, height = output_size
        diagonal_pixels = np.sqrt(width ** 2 + height ** 2)
        
        if medium == "screen":
            # For screen viewing, base on pixel density
            base_radius = 0.5
            size_factor = diagonal_pixels / 2000.0  # Normalize to ~2000px diagonal
            return base_radius * np.sqrt(size_factor)
        
        elif medium == "print":
            # For print, consider DPI
            if print_dpi:
                # Calculate viewing distance factor
                viewing_distance_inches = 12.0  # Typical viewing distance
                angular_resolution = 1 / 60.0  # 1 arc minute
                
                optimal_radius = (print_dpi * angular_resolution * 
                                viewing_distance_inches / 60.0)
                return min(optimal_radius, 2.0)
            return 1.0
        
        elif medium == "web":
            # For web, assume lower resolution viewing
            return 0.3 + (diagonal_pixels / 4000.0)
        
        return 0.5
    
    def _adjust_amount_for_medium(self, base_amount: float, medium: str) -> float:
        """Adjust sharpening amount based on output medium."""
        multipliers = {
            "screen": 1.0,
            "print": 1.5,  # Print needs more sharpening
            "web": 0.7     # Web needs less (will be compressed)
        }
        return base_amount * multipliers.get(medium, 1.0)
    
    def _apply_s_curve(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Apply S-curve for contrast enhancement."""
        # Simple S-curve using tanh
        normalized = (image - 0.5) * strength * 2
        return 0.5 + 0.5 * np.tanh(normalized)


def create_default_sharpening_settings(style: str = "standard") -> SharpeningSettings:
    """Create default sharpening settings for different styles."""
    if style == "standard":
        return SharpeningSettings()
    
    elif style == "portrait":
        # Softer sharpening for portraits
        return SharpeningSettings(
            input_sharpening=InputSharpeningSettings(
                amount=30.0,
                radius=0.6
            ),
            creative_sharpening=CreativeSharpeningSettings(
                amount=60.0,
                radius=1.2,
                clarity_amount=10.0,
                mask_edges=True
            ),
            output_sharpening=OutputSharpeningSettings(
                amount=30.0
            )
        )
    
    elif style == "landscape":
        # Stronger sharpening for landscapes
        return SharpeningSettings(
            input_sharpening=InputSharpeningSettings(
                amount=70.0,
                radius=1.0,
                iterations=15
            ),
            creative_sharpening=CreativeSharpeningSettings(
                amount=150.0,
                radius=1.8,
                clarity_amount=40.0,
                clarity_radius=75.0
            ),
            output_sharpening=OutputSharpeningSettings(
                amount=60.0
            )
        )
    
    else:
        return SharpeningSettings()