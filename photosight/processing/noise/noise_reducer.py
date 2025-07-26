"""
Advanced noise reduction implementation with edge preservation.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from scipy.ndimage import median_filter, gaussian_filter
from scipy.fft import fft2, ifft2, fftshift
import logging

from .models import (
    NoiseReductionSettings, NoiseProfile, NoiseAnalysis, 
    EdgePreservationMethod, NoiseReductionConfig, FrequencyBand
)
from .profiles import get_noise_profile, get_generic_profile

logger = logging.getLogger(__name__)


class NoiseReducer:
    """
    Advanced noise reduction processor with multiple algorithms.
    
    Supports:
    - Luminance noise reduction with edge preservation
    - Chrominance noise reduction
    - Hot pixel removal
    - Pattern noise suppression
    - ISO-adaptive processing
    """
    
    def __init__(self, config: Optional[NoiseReductionConfig] = None):
        """Initialize noise reducer with configuration."""
        self.config = config or NoiseReductionConfig()
        self._edge_cache = {}
        
    def reduce_noise(self, image: np.ndarray, settings: NoiseReductionSettings,
                    iso: Optional[int] = None, 
                    camera_model: Optional[str] = None,
                    mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply comprehensive noise reduction to image.
        
        Args:
            image: Input image (float32, 0-1 range, RGB)
            settings: Noise reduction settings
            iso: ISO value for adaptive processing
            camera_model: Camera model for specific profiles
            mask: Optional mask for local noise reduction
            
        Returns:
            Denoised image
        """
        # Ensure proper format
        image = self._prepare_image(image)
        height, width = image.shape[:2]
        
        # Get noise profile if ISO-adaptive
        profile = None
        if settings.iso_adaptive and iso is not None:
            if camera_model:
                profile = get_noise_profile(camera_model, iso)
            if profile is None:
                profile = get_generic_profile(iso)
                
            # Auto-adjust settings based on profile
            if profile and settings.luminance_amount == 0 and settings.chrominance_amount == 0:
                auto_settings = profile.get_auto_settings()
                settings = auto_settings
        
        # Analyze image noise if needed
        if settings.quality_mode == "high" or profile is None:
            noise_analysis = self._analyze_noise(image)
            logger.debug(f"Noise analysis: Lum={noise_analysis.estimated_luminance_noise:.1f}, "
                        f"Chr={noise_analysis.estimated_chrominance_noise:.1f}")
        
        # Apply pre-processing for problem pixels
        if settings.hot_pixel_removal:
            image = self._remove_hot_pixels(image, iso)
        
        # Main noise reduction based on quality mode
        if settings.quality_mode == "fast":
            result = self._fast_noise_reduction(image, settings)
        elif settings.quality_mode == "balanced":
            result = self._balanced_noise_reduction(image, settings, profile)
        else:  # high quality
            result = self._high_quality_noise_reduction(image, settings, profile)
        
        # Pattern noise suppression
        if settings.pattern_suppression and profile and profile.has_pattern_noise:
            result = self._suppress_pattern_noise(result)
        
        # Banding reduction (careful!)
        if settings.banding_reduction and profile and profile.has_banding:
            result = self._reduce_banding(result)
        
        # Apply mask if provided
        if mask is not None and settings.mask_enabled:
            result = self._apply_mask(image, result, mask, settings.mask_blend_mode)
        
        return np.clip(result, 0, 1)
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for processing."""
        if image.dtype != np.float32:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            else:
                image = image.astype(np.float32)
        return np.clip(image, 0, 1)
    
    def _fast_noise_reduction(self, image: np.ndarray, 
                            settings: NoiseReductionSettings) -> np.ndarray:
        """Fast noise reduction using bilateral filtering."""
        result = image.copy()
        
        # Luminance noise reduction
        if settings.luminance_amount > 0:
            # Convert to LAB for luminance processing
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0].astype(np.float32) / 255.0
            
            # Bilateral filter on luminance
            sigma_space = self.config.bilateral_sigma_space * (settings.luminance_amount / 50)
            sigma_color = self.config.bilateral_sigma_color * (settings.detail_preservation / 100)
            
            l_filtered = cv2.bilateralFilter(
                (l_channel * 255).astype(np.uint8),
                d=9,
                sigmaColor=sigma_color * 255,
                sigmaSpace=sigma_space
            ).astype(np.float32) / 255.0
            
            # Blend based on amount
            blend_factor = settings.luminance_amount / 100.0
            l_channel = l_channel * (1 - blend_factor) + l_filtered * blend_factor
            
            lab[:, :, 0] = (l_channel * 255).astype(np.uint8)
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        # Chrominance noise reduction
        if settings.chrominance_amount > 0:
            result = self._reduce_chrominance_fast(result, settings.chrominance_amount)
        
        return result
    
    def _balanced_noise_reduction(self, image: np.ndarray,
                                settings: NoiseReductionSettings,
                                profile: Optional[NoiseProfile]) -> np.ndarray:
        """Balanced quality/speed noise reduction."""
        # Use guided filter for edge preservation
        if settings.edge_preservation_method == EdgePreservationMethod.GUIDED:
            return self._guided_filter_denoise(image, settings, profile)
        else:
            return self._adaptive_denoise(image, settings, profile)
    
    def _high_quality_noise_reduction(self, image: np.ndarray,
                                    settings: NoiseReductionSettings,
                                    profile: Optional[NoiseProfile]) -> np.ndarray:
        """High quality noise reduction with multiple algorithms."""
        result = image.copy()
        
        # Multi-scale processing if enabled
        if settings.multi_scale:
            result = self._multiscale_denoise(result, settings, profile)
        else:
            # Single scale high-quality
            if settings.luminance_amount > 0:
                if settings.edge_preservation_method == EdgePreservationMethod.NLM:
                    result = self._nlm_denoise(result, settings, profile)
                elif settings.edge_preservation_method == EdgePreservationMethod.WAVELET:
                    result = self._wavelet_denoise(result, settings, profile)
                else:
                    result = self._adaptive_denoise(result, settings, profile)
        
        # High-quality chrominance reduction
        if settings.chrominance_amount > 0:
            result = self._reduce_chrominance_advanced(result, settings, profile)
        
        return result
    
    def _guided_filter_denoise(self, image: np.ndarray,
                              settings: NoiseReductionSettings,
                              profile: Optional[NoiseProfile]) -> np.ndarray:
        """Guided filter-based denoising."""
        # Convert to LAB
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        
        if settings.luminance_amount > 0:
            # Use image itself as guide for structure preservation
            radius = int(5 + settings.luminance_amount / 20)
            eps = 0.01 * (1 - settings.detail_preservation / 100)
            
            lab[:, :, 0] = cv2.ximgproc.guidedFilter(
                guide=lab[:, :, 0],
                src=lab[:, :, 0],
                radius=radius,
                eps=eps
            )
        
        # Convert back
        lab = (lab * 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        return result
    
    def _nlm_denoise(self, image: np.ndarray,
                    settings: NoiseReductionSettings,
                    profile: Optional[NoiseProfile]) -> np.ndarray:
        """Non-local means denoising."""
        # NLM parameters based on settings
        h = settings.luminance_amount / 100.0 * 0.1  # Filter strength
        template_window = self.config.patch_size
        search_window = self.config.search_window
        
        # Apply to each channel
        result = np.zeros_like(image)
        for i in range(3):
            # Adjust h based on channel noise (blue typically noisier)
            channel_h = h
            if i == 2 and profile:  # Blue channel
                channel_h *= profile.blue_channel_factor
                
            result[:, :, i] = cv2.fastNlMeansDenoising(
                (image[:, :, i] * 255).astype(np.uint8),
                None,
                h=channel_h * 255,
                templateWindowSize=template_window,
                searchWindowSize=search_window
            ).astype(np.float32) / 255.0
        
        # Preserve edges
        if settings.detail_preservation > 50:
            edges = self._detect_edges(image)
            blend_factor = (settings.detail_preservation - 50) / 50.0
            result = result * (1 - edges * blend_factor) + image * (edges * blend_factor)
        
        return result
    
    def _wavelet_denoise(self, image: np.ndarray,
                        settings: NoiseReductionSettings,
                        profile: Optional[NoiseProfile]) -> np.ndarray:
        """Wavelet-based denoising with edge preservation."""
        import pywt
        
        result = np.zeros_like(image)
        
        for channel in range(3):
            # Wavelet decomposition
            coeffs = pywt.wavedec2(image[:, :, channel], 'db4', level=4)
            
            # Threshold coefficients based on noise level
            threshold = (settings.luminance_amount / 100.0) * 0.1
            if profile:
                threshold *= (profile.luminance_noise_level / 50.0)
            
            # Soft thresholding
            coeffs_thresh = list(coeffs)
            for i in range(1, len(coeffs)):
                # Apply different thresholds to different scales
                scale_threshold = threshold * (1.0 - i * 0.2)  # Less aggressive at finer scales
                
                coeffs_thresh[i] = tuple([
                    pywt.threshold(c, scale_threshold, mode='soft') 
                    for c in coeffs[i]
                ])
            
            # Reconstruct
            result[:, :, channel] = pywt.waverec2(coeffs_thresh, 'db4')
        
        return np.clip(result, 0, 1)
    
    def _multiscale_denoise(self, image: np.ndarray,
                          settings: NoiseReductionSettings,
                          profile: Optional[NoiseProfile]) -> np.ndarray:
        """Multi-scale noise reduction using pyramid decomposition."""
        # Build Gaussian pyramid
        pyramid = [image]
        current = image
        
        for i in range(self.config.num_scales - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        
        # Process each scale
        denoised_pyramid = []
        for i, scale_image in enumerate(pyramid):
            # Adjust settings for scale
            scale_settings = settings.scale_by_factor(1.0 - i * 0.2)
            
            # Denoise this scale
            if i == 0:  # Finest scale - preserve more detail
                denoised = self._guided_filter_denoise(scale_image, scale_settings, profile)
            else:
                denoised = self._nlm_denoise(scale_image, scale_settings, profile)
            
            denoised_pyramid.append(denoised)
        
        # Reconstruct from pyramid
        result = denoised_pyramid[-1]
        for i in range(len(denoised_pyramid) - 2, -1, -1):
            result = cv2.pyrUp(result)
            # Ensure dimensions match
            h, w = denoised_pyramid[i].shape[:2]
            result = cv2.resize(result, (w, h))
            
            # Blend with finer scale
            blend_factor = 0.5 + i * 0.1  # More weight to finer scales
            result = result * (1 - blend_factor) + denoised_pyramid[i] * blend_factor
        
        return result
    
    def _reduce_chrominance_fast(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Fast chrominance noise reduction."""
        # Convert to YCbCr
        ycbcr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        
        # Apply median filter to color channels
        kernel_size = 3 + int(amount / 33)  # 3, 5, or 7
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        ycbcr[:, :, 1] = cv2.medianBlur(ycbcr[:, :, 1], kernel_size)
        ycbcr[:, :, 2] = cv2.medianBlur(ycbcr[:, :, 2], kernel_size)
        
        # Convert back
        result = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
        
        # Blend based on amount
        blend_factor = amount / 100.0
        return image * (1 - blend_factor) + result * blend_factor
    
    def _reduce_chrominance_advanced(self, image: np.ndarray,
                                   settings: NoiseReductionSettings,
                                   profile: Optional[NoiseProfile]) -> np.ndarray:
        """Advanced chrominance noise reduction with color preservation."""
        # Convert to LAB
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Process a and b channels
        for channel in [1, 2]:
            # Adaptive kernel size based on noise grain
            if profile and profile.noise_grain_size > 1.0:
                kernel_size = 5 + int(profile.noise_grain_size)
            else:
                kernel_size = 5
                
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Edge-aware filtering
            edges = self._detect_edges(lab[:, :, 0] / 255.0)
            
            # Strong filtering in smooth areas, weak in edges
            filtered = cv2.bilateralFilter(
                lab[:, :, channel].astype(np.uint8),
                d=kernel_size,
                sigmaColor=75,
                sigmaSpace=75
            ).astype(np.float32)
            
            # Blend based on edges and amount
            blend = (settings.chrominance_amount / 100.0) * (1 - edges)
            lab[:, :, channel] = lab[:, :, channel] * (1 - blend) + filtered * blend
        
        # Convert back
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        return result
    
    def _detect_edges(self, luminance: np.ndarray) -> np.ndarray:
        """Detect edges for preservation."""
        # Check cache
        cache_key = id(luminance)
        if cache_key in self._edge_cache:
            return self._edge_cache[cache_key]
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize and threshold
        edges = edges / (edges.max() + 1e-8)
        edges = np.where(edges > self.config.edge_threshold, edges, 0)
        
        # Dilate edges slightly
        if self.config.edge_dilation > 0:
            kernel = np.ones((self.config.edge_dilation*2+1, self.config.edge_dilation*2+1))
            edges = cv2.dilate(edges, kernel)
        
        # Smooth edge mask
        edges = gaussian_filter(edges, sigma=1.0)
        edges = np.clip(edges, 0, 1)
        
        # Cache result
        self._edge_cache[cache_key] = edges
        
        return edges
    
    def _remove_hot_pixels(self, image: np.ndarray, iso: Optional[int]) -> np.ndarray:
        """Remove hot/stuck pixels."""
        result = image.copy()
        
        # Hot pixel detection threshold
        threshold = self.config.hot_pixel_dev_threshold
        if iso and iso > 6400:
            threshold *= 0.8  # More aggressive at high ISO
        
        for channel in range(3):
            # Median filter for comparison
            median = median_filter(image[:, :, channel], size=self.config.hot_pixel_kernel_size)
            
            # Find pixels that deviate significantly from local median
            diff = np.abs(image[:, :, channel] - median)
            local_std = np.std(diff)
            
            # Replace hot pixels with median value
            hot_pixels = diff > (threshold * local_std)
            result[:, :, channel][hot_pixels] = median[hot_pixels]
        
        return result
    
    def _suppress_pattern_noise(self, image: np.ndarray) -> np.ndarray:
        """Suppress fixed pattern noise using frequency domain."""
        result = image.copy()
        
        for channel in range(3):
            # FFT
            f_transform = fft2(image[:, :, channel])
            f_shift = fftshift(f_transform)
            
            # Magnitude spectrum
            magnitude = np.abs(f_shift)
            
            # Find peaks (patterns)
            mean_mag = np.mean(magnitude)
            threshold = mean_mag * self.config.pattern_fft_threshold
            
            # Create mask to suppress patterns
            rows, cols = image.shape[:2]
            crow, ccol = rows // 2, cols // 2
            
            # Don't suppress DC component and low frequencies
            mask = np.ones_like(magnitude)
            pattern_mask = magnitude > threshold
            
            # Preserve center (low frequencies)
            pattern_mask[crow-10:crow+10, ccol-10:ccol+10] = False
            
            # Suppress patterns
            mask[pattern_mask] *= self.config.pattern_suppression_factor
            
            # Apply mask and inverse FFT
            f_shift_filtered = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            result[:, :, channel] = np.real(ifft2(f_ishift))
        
        return np.clip(result, 0, 1)
    
    def _reduce_banding(self, image: np.ndarray) -> np.ndarray:
        """Reduce horizontal/vertical banding artifacts."""
        # This is tricky and can harm the image if not careful
        # Simple approach: slight blur in the direction of banding
        
        # Detect dominant banding direction (simplified)
        # In practice, this would need more sophisticated analysis
        
        # Apply very slight directional blur
        kernel_h = np.array([[1, 2, 1]]) / 4.0  # Horizontal
        kernel_v = kernel_h.T  # Vertical
        
        # Very gentle application
        blurred_h = cv2.filter2D(image, -1, kernel_h)
        blurred_v = cv2.filter2D(image, -1, kernel_v)
        
        # Blend slightly (very conservative)
        blend_factor = 0.1  # Only 10% blend to avoid destroying detail
        result = image * (1 - blend_factor) + blurred_h * (blend_factor * 0.5) + blurred_v * (blend_factor * 0.5)
        
        return result
    
    def _adaptive_denoise(self, image: np.ndarray,
                         settings: NoiseReductionSettings,
                         profile: Optional[NoiseProfile]) -> np.ndarray:
        """Adaptive denoising based on local image statistics."""
        # Analyze local statistics
        window_size = 15
        
        # Local mean and variance
        mean = cv2.blur(image, (window_size, window_size))
        mean_sq = cv2.blur(image ** 2, (window_size, window_size))
        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0)
        
        # Adaptive filter strength based on local variance
        # Low variance = more filtering, high variance = less filtering
        noise_var = (settings.luminance_amount / 100.0) ** 2 * 0.01
        filter_strength = noise_var / (variance + noise_var)
        
        # Apply adaptive filtering
        filtered = cv2.bilateralFilter(
            (image * 255).astype(np.uint8),
            d=9,
            sigmaColor=75,
            sigmaSpace=75
        ).astype(np.float32) / 255.0
        
        # Adaptive blending
        result = image * (1 - filter_strength) + filtered * filter_strength
        
        return result
    
    def _analyze_noise(self, image: np.ndarray) -> NoiseAnalysis:
        """Analyze image noise characteristics."""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # High-pass filter to isolate noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        
        # Estimate noise levels
        noise_std = np.std(noise)
        luminance_noise = noise_std * 1000  # Scale to 0-100 range
        
        # Chrominance noise estimation
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        a_noise = np.std(lab[:, :, 1]) / 255.0 * 200
        b_noise = np.std(lab[:, :, 2]) / 255.0 * 200
        chrominance_noise = (a_noise + b_noise) / 2
        
        # Noise frequency analysis
        noise_fft = np.abs(fft2(noise))
        high_freq = np.sum(noise_fft[noise_fft.shape[0]//4:, :])
        low_freq = np.sum(noise_fft[:noise_fft.shape[0]//4, :])
        
        if high_freq > low_freq * 2:
            dominant_freq = "fine"
        elif low_freq > high_freq * 2:
            dominant_freq = "coarse"
        else:
            dominant_freq = "medium"
        
        return NoiseAnalysis(
            estimated_luminance_noise=min(luminance_noise, 100),
            estimated_chrominance_noise=min(chrominance_noise, 100),
            dominant_noise_frequency=dominant_freq,
            noise_uniformity=1.0 - (np.std(noise.std(axis=1)) / (noise_std + 1e-8))
        )
    
    def _apply_mask(self, original: np.ndarray, denoised: np.ndarray,
                   mask: np.ndarray, blend_mode: str) -> np.ndarray:
        """Apply denoising through mask."""
        # Ensure mask is 3-channel
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        if blend_mode == "normal":
            return original * (1 - mask) + denoised * mask
        else:
            # Could implement other blend modes if needed
            return original * (1 - mask) + denoised * mask
    
    def clear_cache(self):
        """Clear edge detection cache."""
        self._edge_cache.clear()