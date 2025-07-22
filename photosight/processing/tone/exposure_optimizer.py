"""
Exposure optimization module for PhotoSight

Analyzes image histograms and provides intelligent exposure adjustments
including shadow/highlight recovery and dynamic range optimization.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class ExposureAnalysis:
    """Results from exposure analysis"""
    # Histogram statistics
    mean_brightness: float
    median_brightness: float
    histogram_peaks: List[float]
    
    # Clipping analysis
    shadow_clipping: float  # Percentage of pixels in shadows
    highlight_clipping: float  # Percentage of pixels in highlights
    
    # Dynamic range metrics
    dynamic_range: float  # Ratio of usable tonal range
    contrast_ratio: float
    
    # Zone system analysis (Ansel Adams)
    zone_distribution: Dict[int, float]  # Distribution across 11 zones (0-10)
    
    # Suggested adjustments
    exposure_compensation: float  # EV adjustment
    shadow_lift: float  # 0-100
    highlight_recovery: float  # -100 to 0
    
    # Scene-specific recommendations
    preserve_highlights: bool
    lift_shadows: bool
    increase_contrast: bool
    
    # Quality metrics
    confidence: float
    quality_score: float


class ExposureOptimizer:
    """
    Intelligent exposure optimization with histogram analysis
    
    Features:
    - Multi-zone histogram analysis
    - Adaptive shadow/highlight recovery
    - Scene-aware exposure compensation
    - Preservation of important tonal regions
    """
    
    def __init__(self,
                 shadow_threshold: float = 0.15,
                 highlight_threshold: float = 0.85,
                 target_mean: float = 0.45,
                 preserve_skin_tones: bool = True):
        """
        Initialize exposure optimizer
        
        Args:
            shadow_threshold: Threshold for shadow regions (0-1)
            highlight_threshold: Threshold for highlight regions (0-1)
            target_mean: Target mean brightness (0-1)
            preserve_skin_tones: Whether to protect skin tone regions
        """
        self.shadow_threshold = shadow_threshold
        self.highlight_threshold = highlight_threshold
        self.target_mean = target_mean
        self.preserve_skin_tones = preserve_skin_tones
        
        # Zone system boundaries (11 zones from pure black to pure white)
        self.zone_boundaries = np.linspace(0, 1, 12)
    
    def analyze_exposure(self, image: np.ndarray, 
                        raw_bit_depth: int = 14) -> ExposureAnalysis:
        """
        Analyze image exposure and suggest optimizations
        
        Args:
            image: RGB image array (uint8 or uint16)
            raw_bit_depth: Original RAW bit depth for better analysis
            
        Returns:
            ExposureAnalysis with detailed metrics and suggestions
        """
        # Convert to float32 for analysis
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_float = image.astype(np.float32) / 65535.0
        else:
            img_float = image.astype(np.float32)
        
        # Convert to luminance
        luminance = self._calculate_luminance(img_float)
        
        # Basic statistics
        mean_brightness = np.mean(luminance)
        median_brightness = np.median(luminance)
        
        # Histogram analysis
        hist, bins = np.histogram(luminance, bins=256, range=(0, 1))
        hist_norm = hist / hist.sum()
        
        # Find histogram peaks
        peaks = self._find_histogram_peaks(hist_norm)
        
        # Clipping analysis
        shadow_clipping = np.sum(luminance < self.shadow_threshold) / luminance.size
        highlight_clipping = np.sum(luminance > self.highlight_threshold) / luminance.size
        
        # Dynamic range analysis
        percentile_5 = np.percentile(luminance, 5)
        percentile_95 = np.percentile(luminance, 95)
        dynamic_range = percentile_95 - percentile_5
        
        # Contrast ratio
        contrast_ratio = self._calculate_contrast_ratio(luminance)
        
        # Zone system analysis
        zone_distribution = self._analyze_zones(luminance)
        
        # Scene analysis
        is_low_key = mean_brightness < 0.3 and zone_distribution[0] + zone_distribution[1] > 0.4
        is_high_key = mean_brightness > 0.7 and zone_distribution[9] + zone_distribution[10] > 0.4
        is_backlit = self._detect_backlit_scene(luminance, img_float)
        
        # Calculate suggested adjustments
        exposure_comp = self._calculate_exposure_compensation(
            mean_brightness, median_brightness, zone_distribution, is_low_key, is_high_key
        )
        
        shadow_lift = self._calculate_shadow_adjustment(
            shadow_clipping, zone_distribution, is_low_key
        )
        
        highlight_recovery = self._calculate_highlight_adjustment(
            highlight_clipping, zone_distribution, is_high_key, is_backlit
        )
        
        # Determine preservation strategies
        preserve_highlights = highlight_clipping > 0.02 or is_backlit
        lift_shadows = shadow_clipping > 0.15 or (dynamic_range < 0.6 and not is_low_key)
        increase_contrast = dynamic_range < 0.5 and not (is_low_key or is_high_key)
        
        # Calculate confidence based on histogram quality
        confidence = self._calculate_confidence(hist_norm, dynamic_range)
        
        # Overall quality score
        quality_score = self._calculate_quality_score(
            shadow_clipping, highlight_clipping, dynamic_range, contrast_ratio
        )
        
        return ExposureAnalysis(
            mean_brightness=float(mean_brightness),
            median_brightness=float(median_brightness),
            histogram_peaks=peaks,
            shadow_clipping=float(shadow_clipping),
            highlight_clipping=float(highlight_clipping),
            dynamic_range=float(dynamic_range),
            contrast_ratio=float(contrast_ratio),
            zone_distribution=zone_distribution,
            exposure_compensation=float(exposure_comp),
            shadow_lift=float(shadow_lift),
            highlight_recovery=float(highlight_recovery),
            preserve_highlights=preserve_highlights,
            lift_shadows=lift_shadows,
            increase_contrast=increase_contrast,
            confidence=float(confidence),
            quality_score=float(quality_score)
        )
    
    def optimize_exposure(self, image: np.ndarray, 
                         analysis: Optional[ExposureAnalysis] = None) -> np.ndarray:
        """
        Apply exposure optimization to image
        
        Args:
            image: Input image (RGB, uint8 or uint16)
            analysis: Pre-computed analysis (will analyze if not provided)
            
        Returns:
            Optimized image in same format as input
        """
        # Store original dtype
        orig_dtype = image.dtype
        orig_max = 255 if orig_dtype == np.uint8 else 65535
        
        # Convert to float
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_float = image.astype(np.float32) / 65535.0
        else:
            img_float = image.copy()
        
        # Analyze if needed
        if analysis is None:
            analysis = self.analyze_exposure(image)
        
        # Apply exposure compensation
        if abs(analysis.exposure_compensation) > 0.1:
            img_float = self._apply_exposure_compensation(img_float, analysis.exposure_compensation)
        
        # Apply shadow/highlight adjustments
        if analysis.shadow_lift > 0 or analysis.highlight_recovery < 0:
            img_float = self._apply_shadow_highlight_recovery(
                img_float, 
                analysis.shadow_lift, 
                analysis.highlight_recovery,
                analysis.preserve_highlights
            )
        
        # Apply adaptive contrast if needed
        if analysis.increase_contrast:
            img_float = self._apply_adaptive_contrast(img_float, analysis.dynamic_range)
        
        # Convert back to original dtype
        img_float = np.clip(img_float, 0, 1)
        if orig_dtype == np.uint8:
            return (img_float * 255).astype(np.uint8)
        else:
            return (img_float * 65535).astype(np.uint16)
    
    def _calculate_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """Calculate perceptual luminance from RGB"""
        # Use Rec. 709 luma coefficients
        return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    
    def _find_histogram_peaks(self, hist: np.ndarray, min_prominence: float = 0.02) -> List[float]:
        """Find significant peaks in histogram"""
        peaks = []
        
        # Smooth histogram
        hist_smooth = ndimage.gaussian_filter1d(hist, sigma=2)
        
        # Find local maxima
        for i in range(1, len(hist_smooth) - 1):
            if (hist_smooth[i] > hist_smooth[i-1] and 
                hist_smooth[i] > hist_smooth[i+1] and
                hist_smooth[i] > min_prominence):
                peaks.append(i / len(hist_smooth))
        
        return peaks
    
    def _analyze_zones(self, luminance: np.ndarray) -> Dict[int, float]:
        """Analyze distribution across Ansel Adams zone system"""
        zones = {}
        total_pixels = luminance.size
        
        for i in range(11):
            lower = self.zone_boundaries[i]
            upper = self.zone_boundaries[i + 1]
            count = np.sum((luminance >= lower) & (luminance < upper))
            zones[i] = count / total_pixels
        
        return zones
    
    def _detect_backlit_scene(self, luminance: np.ndarray, rgb: np.ndarray) -> bool:
        """Detect if scene is backlit"""
        h, w = luminance.shape
        
        # Check center vs edges
        center_region = luminance[h//3:2*h//3, w//3:2*w//3]
        edge_regions = np.concatenate([
            luminance[:h//3, :].flatten(),
            luminance[2*h//3:, :].flatten(),
            luminance[:, :w//3].flatten(),
            luminance[:, 2*w//3:].flatten()
        ])
        
        center_mean = np.mean(center_region)
        edge_mean = np.mean(edge_regions)
        
        # Backlit if edges are significantly brighter than center
        return edge_mean > center_mean * 1.5
    
    def _calculate_exposure_compensation(self, mean_bright: float, median_bright: float,
                                       zones: Dict[int, float], is_low_key: bool, 
                                       is_high_key: bool) -> float:
        """Calculate optimal exposure compensation in EV"""
        # Target depends on scene type
        if is_low_key:
            target = 0.3
        elif is_high_key:
            target = 0.65
        else:
            target = self.target_mean
        
        # Calculate basic compensation
        current = mean_bright
        if current > 0:
            ev_adjustment = np.log2(target / current)
        else:
            ev_adjustment = 1.0
        
        # Limit adjustment based on scene
        if is_low_key:
            ev_adjustment = min(ev_adjustment, 0.5)  # Don't over-brighten low key
        elif is_high_key:
            ev_adjustment = max(ev_adjustment, -0.5)  # Don't over-darken high key
        
        # Consider zone distribution
        # If too much in extreme zones, reduce adjustment
        extreme_zones = zones[0] + zones[1] + zones[9] + zones[10]
        if extreme_zones > 0.3:
            ev_adjustment *= 0.7
        
        # Clamp to reasonable range
        return np.clip(ev_adjustment, -2.0, 2.0)
    
    def _calculate_shadow_adjustment(self, shadow_clip: float, zones: Dict[int, float], 
                                    is_low_key: bool) -> float:
        """Calculate shadow lifting amount (0-100)"""
        if is_low_key:
            # Minimal shadow lifting for low-key images
            return min(shadow_clip * 50, 15)
        
        # Base adjustment on clipping
        base_lift = shadow_clip * 100
        
        # Adjust based on zone distribution
        dark_zones = zones[0] + zones[1] + zones[2]
        if dark_zones > 0.4:
            # Significant shadow detail to recover
            base_lift *= 1.2
        elif dark_zones < 0.1:
            # Few shadows, less aggressive
            base_lift *= 0.5
        
        return np.clip(base_lift, 0, 75)
    
    def _calculate_highlight_adjustment(self, highlight_clip: float, zones: Dict[int, float],
                                       is_high_key: bool, is_backlit: bool) -> float:
        """Calculate highlight recovery amount (-100 to 0)"""
        if is_high_key:
            # Minimal highlight recovery for high-key images
            return max(highlight_clip * -30, -10)
        
        # Base adjustment on clipping
        base_recovery = highlight_clip * -100
        
        # More aggressive for backlit scenes
        if is_backlit:
            base_recovery *= 1.5
        
        # Adjust based on zone distribution
        bright_zones = zones[8] + zones[9] + zones[10]
        if bright_zones > 0.3:
            # Significant highlight detail to recover
            base_recovery *= 1.2
        
        return np.clip(base_recovery, -100, 0)
    
    def _calculate_contrast_ratio(self, luminance: np.ndarray) -> float:
        """Calculate scene contrast ratio"""
        # Use 5th and 95th percentiles to avoid outliers
        dark_point = np.percentile(luminance, 5)
        bright_point = np.percentile(luminance, 95)
        
        if dark_point > 0:
            return bright_point / dark_point
        else:
            return 100.0  # Max ratio
    
    def _calculate_confidence(self, histogram: np.ndarray, dynamic_range: float) -> float:
        """Calculate confidence in analysis"""
        # Good histogram has smooth distribution
        smoothness = 1.0 - np.std(np.diff(histogram))
        
        # Good dynamic range
        range_score = min(dynamic_range / 0.7, 1.0)
        
        # Avoid extreme distributions
        entropy = -np.sum(histogram * np.log(histogram + 1e-10))
        entropy_score = min(entropy / 5.0, 1.0)
        
        return (smoothness + range_score + entropy_score) / 3.0
    
    def _calculate_quality_score(self, shadow_clip: float, highlight_clip: float,
                                dynamic_range: float, contrast_ratio: float) -> float:
        """Calculate overall exposure quality score"""
        # Penalize clipping
        clipping_score = 1.0 - (shadow_clip + highlight_clip)
        
        # Reward good dynamic range
        range_score = min(dynamic_range / 0.7, 1.0)
        
        # Optimal contrast ratio around 100:1
        contrast_score = 1.0 - abs(np.log10(contrast_ratio) - 2.0) / 2.0
        contrast_score = max(0, min(1, contrast_score))
        
        return (clipping_score * 0.4 + range_score * 0.3 + contrast_score * 0.3)
    
    def _apply_exposure_compensation(self, img: np.ndarray, ev_adjustment: float) -> np.ndarray:
        """Apply exposure compensation in EV stops"""
        return img * (2 ** ev_adjustment)
    
    def _apply_shadow_highlight_recovery(self, img: np.ndarray, shadow_lift: float,
                                        highlight_recovery: float, 
                                        preserve_highlights: bool) -> np.ndarray:
        """Apply shadow and highlight adjustments"""
        # Create luminance map
        luminance = self._calculate_luminance(img)
        
        # Shadow adjustment
        if shadow_lift > 0:
            # Create shadow mask
            shadow_mask = np.maximum(0, 1 - luminance / self.shadow_threshold)
            shadow_mask = shadow_mask[:, :, np.newaxis]
            
            # Lift shadows
            shadow_adjustment = shadow_mask * (shadow_lift / 100.0)
            img = img + (shadow_adjustment * (1 - img))
        
        # Highlight adjustment
        if highlight_recovery < 0:
            # Create highlight mask
            highlight_mask = np.maximum(0, (luminance - self.highlight_threshold) / 
                                      (1 - self.highlight_threshold))
            highlight_mask = highlight_mask[:, :, np.newaxis]
            
            # Recover highlights
            recovery_strength = abs(highlight_recovery) / 100.0
            
            if preserve_highlights:
                # Gentle highlight compression
                img = img - (highlight_mask * img * recovery_strength * 0.5)
            else:
                # Stronger highlight recovery
                img = img - (highlight_mask * (img - 0.5) * recovery_strength)
        
        return np.clip(img, 0, 1)
    
    def _apply_adaptive_contrast(self, img: np.ndarray, current_range: float) -> np.ndarray:
        """Apply adaptive contrast enhancement"""
        # Target range is 0.7
        if current_range >= 0.65:
            return img
        
        # Calculate adaptive gamma
        gamma = 0.7 / current_range
        gamma = np.clip(gamma, 0.7, 1.4)
        
        # Apply S-curve for better contrast
        # Steepen the midtones while preserving extremes
        midpoint = 0.5
        
        # Create S-curve LUT
        x = np.linspace(0, 1, 256)
        # Sigmoid function centered at 0.5
        k = 3.0  # Steepness
        y = 1 / (1 + np.exp(-k * (x - midpoint)))
        # Normalize to maintain endpoints
        y = (y - y[0]) / (y[-1] - y[0])
        
        # Apply LUT to each channel
        result = img.copy()
        for c in range(3):
            result[:, :, c] = np.interp(img[:, :, c], x, y)
        
        return result
    
    def generate_tone_curve(self, analysis: ExposureAnalysis) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a tone curve based on analysis
        
        Args:
            analysis: Exposure analysis results
            
        Returns:
            Tuple of (input_values, output_values) for the curve
        """
        x = np.linspace(0, 1, 256)
        y = x.copy()
        
        # Apply exposure compensation
        if abs(analysis.exposure_compensation) > 0.1:
            y = y * (2 ** analysis.exposure_compensation)
        
        # Shadow lifting
        if analysis.shadow_lift > 0:
            shadow_region = x < self.shadow_threshold
            lift_amount = analysis.shadow_lift / 100.0
            y[shadow_region] = y[shadow_region] + (1 - y[shadow_region]) * lift_amount * \
                              (1 - x[shadow_region] / self.shadow_threshold)
        
        # Highlight recovery
        if analysis.highlight_recovery < 0:
            highlight_region = x > self.highlight_threshold
            recovery_amount = abs(analysis.highlight_recovery) / 100.0
            y[highlight_region] = y[highlight_region] - \
                                (y[highlight_region] - self.highlight_threshold) * recovery_amount * \
                                ((x[highlight_region] - self.highlight_threshold) / 
                                 (1 - self.highlight_threshold))
        
        # Ensure curve remains monotonic
        y = np.clip(y, 0, 1)
        for i in range(1, len(y)):
            if y[i] < y[i-1]:
                y[i] = y[i-1]
        
        return x, y
    
    def visualize_analysis(self, image: np.ndarray, analysis: ExposureAnalysis) -> np.ndarray:
        """
        Create visualization of exposure analysis
        
        Args:
            image: Original image
            analysis: Exposure analysis results
            
        Returns:
            Visualization image showing histogram and zones
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Exposure Analysis', fontsize=16)
        
        # Original image
        if image.dtype == np.uint16:
            img_display = (image.astype(np.float32) / 65535 * 255).astype(np.uint8)
        else:
            img_display = image
        
        axes[0, 0].imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Luminance map
        luminance = self._calculate_luminance(image.astype(np.float32) / 
                                           (255 if image.dtype == np.uint8 else 65535))
        axes[0, 1].imshow(luminance, cmap='gray')
        axes[0, 1].set_title('Luminance Map')
        axes[0, 1].axis('off')
        
        # Zone system visualization
        zone_img = np.zeros_like(luminance)
        for i in range(11):
            mask = (luminance >= self.zone_boundaries[i]) & (luminance < self.zone_boundaries[i+1])
            zone_img[mask] = i / 10.0
        
        axes[0, 2].imshow(zone_img, cmap='viridis')
        axes[0, 2].set_title('Zone System Map')
        axes[0, 2].axis('off')
        
        # Histogram
        hist, bins = np.histogram(luminance.flatten(), bins=256, range=(0, 1))
        axes[1, 0].plot(bins[:-1], hist / hist.max())
        axes[1, 0].axvline(self.shadow_threshold, color='blue', linestyle='--', label='Shadow threshold')
        axes[1, 0].axvline(self.highlight_threshold, color='red', linestyle='--', label='Highlight threshold')
        axes[1, 0].set_title('Luminance Histogram')
        axes[1, 0].set_xlabel('Brightness')
        axes[1, 0].set_ylabel('Relative Frequency')
        axes[1, 0].legend()
        
        # Zone distribution
        zones = list(analysis.zone_distribution.keys())
        values = list(analysis.zone_distribution.values())
        axes[1, 1].bar(zones, values)
        axes[1, 1].set_title('Zone Distribution')
        axes[1, 1].set_xlabel('Zone')
        axes[1, 1].set_ylabel('Percentage')
        
        # Tone curve
        x, y = self.generate_tone_curve(analysis)
        axes[1, 2].plot(x, y, 'b-', linewidth=2)
        axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 2].set_title('Suggested Tone Curve')
        axes[1, 2].set_xlabel('Input')
        axes[1, 2].set_ylabel('Output')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add analysis text
        analysis_text = f"Exposure: {analysis.exposure_compensation:+.2f} EV\n"
        analysis_text += f"Shadows: +{analysis.shadow_lift:.0f}%\n"
        analysis_text += f"Highlights: {analysis.highlight_recovery:.0f}%\n"
        analysis_text += f"Quality Score: {analysis.quality_score:.2f}"
        
        fig.text(0.02, 0.02, analysis_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        vis_image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        return vis_image