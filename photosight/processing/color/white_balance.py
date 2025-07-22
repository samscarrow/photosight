"""
White balance correction module for PhotoSight

Implements multiple auto white balance algorithms and manual temperature/tint controls.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class WhiteBalanceMethod(Enum):
    """Available white balance methods"""
    GRAY_WORLD = "gray_world"
    WHITE_PATCH = "white_patch"
    RETINEX = "retinex"
    ILLUMINANT_ESTIMATION = "illuminant_estimation"
    FACE_DETECTION = "face_detection"
    MANUAL = "manual"


@dataclass
class WhiteBalanceAnalysis:
    """Results from white balance analysis"""
    # Detected color temperature
    estimated_temp_kelvin: float
    confidence: float
    
    # RGB multipliers for correction
    red_multiplier: float
    green_multiplier: float
    blue_multiplier: float
    
    # Color cast analysis
    color_cast_strength: float
    dominant_cast: str  # "red", "green", "blue", "cyan", "magenta", "yellow"
    
    # Scene analysis
    method_used: WhiteBalanceMethod
    illuminant_type: str  # "daylight", "tungsten", "fluorescent", "mixed"
    
    # Manual adjustments (relative to detected)
    temperature_adjustment: int  # Kelvin adjustment
    tint_adjustment: int  # Green/Magenta adjustment
    
    # Quality metrics
    gray_deviation: float
    highlight_preservation: float


class WhiteBalanceCorrector:
    """
    Advanced white balance correction with multiple algorithms
    
    Features:
    - Multiple auto WB algorithms
    - Scene-aware white balance
    - Face-priority white balance
    - Manual temperature/tint controls
    """
    
    def __init__(self,
                 default_method: WhiteBalanceMethod = WhiteBalanceMethod.GRAY_WORLD,
                 preserve_skin_tones: bool = True,
                 clip_percentile: float = 0.95):
        """
        Initialize white balance corrector
        
        Args:
            default_method: Default auto WB method
            preserve_skin_tones: Whether to protect skin tones
            clip_percentile: Percentile for highlight clipping in white patch
        """
        self.default_method = default_method
        self.preserve_skin_tones = preserve_skin_tones
        self.clip_percentile = clip_percentile
        
        # Standard illuminant temperatures (Kelvin)
        self.illuminant_temps = {
            'candlelight': 1850,
            'tungsten': 2700,
            'warm_white': 3000,
            'cool_white': 4000,
            'daylight': 5500,
            'cloudy': 6500,
            'shade': 7500,
            'blue_sky': 10000
        }
        
        # RGB to temperature conversion matrix
        self._init_color_temp_data()
    
    def analyze_white_balance(self, image: np.ndarray,
                            method: Optional[WhiteBalanceMethod] = None,
                            roi: Optional[Tuple[int, int, int, int]] = None) -> WhiteBalanceAnalysis:
        """
        Analyze image white balance
        
        Args:
            image: RGB image array
            method: White balance method to use (None for auto-select)
            roi: Region of interest (x, y, width, height)
            
        Returns:
            WhiteBalanceAnalysis with correction parameters
        """
        # Convert to float for processing
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_float = image.astype(np.float32) / 65535.0
        else:
            img_float = image.astype(np.float32)
        
        # Extract ROI if specified
        if roi:
            x, y, w, h = roi
            img_roi = img_float[y:y+h, x:x+w]
        else:
            img_roi = img_float
        
        # Select method
        if method is None:
            method = self._select_best_method(img_roi)
        
        # Calculate RGB multipliers based on method
        if method == WhiteBalanceMethod.GRAY_WORLD:
            multipliers = self._gray_world_wb(img_roi)
        elif method == WhiteBalanceMethod.WHITE_PATCH:
            multipliers = self._white_patch_wb(img_roi)
        elif method == WhiteBalanceMethod.RETINEX:
            multipliers = self._retinex_wb(img_roi)
        elif method == WhiteBalanceMethod.ILLUMINANT_ESTIMATION:
            multipliers = self._illuminant_estimation_wb(img_roi)
        elif method == WhiteBalanceMethod.FACE_DETECTION:
            multipliers = self._face_detection_wb(img_float)  # Use full image
        else:
            multipliers = (1.0, 1.0, 1.0)
        
        # Normalize multipliers
        multipliers = self._normalize_multipliers(multipliers)
        
        # Estimate color temperature
        temp_kelvin = self._estimate_color_temperature(img_roi, multipliers)
        
        # Analyze color cast
        cast_strength, dominant_cast = self._analyze_color_cast(img_roi)
        
        # Determine illuminant type
        illuminant_type = self._classify_illuminant(temp_kelvin)
        
        # Calculate quality metrics
        gray_deviation = self._calculate_gray_deviation(img_roi)
        highlight_preservation = self._calculate_highlight_preservation(img_roi, multipliers)
        
        # Calculate confidence
        confidence = self._calculate_confidence(method, gray_deviation, cast_strength)
        
        return WhiteBalanceAnalysis(
            estimated_temp_kelvin=float(temp_kelvin),
            confidence=float(confidence),
            red_multiplier=float(multipliers[0]),
            green_multiplier=float(multipliers[1]),
            blue_multiplier=float(multipliers[2]),
            color_cast_strength=float(cast_strength),
            dominant_cast=dominant_cast,
            method_used=method,
            illuminant_type=illuminant_type,
            temperature_adjustment=0,
            tint_adjustment=0,
            gray_deviation=float(gray_deviation),
            highlight_preservation=float(highlight_preservation)
        )
    
    def correct_white_balance(self, image: np.ndarray,
                            analysis: Optional[WhiteBalanceAnalysis] = None,
                            temperature_adjust: int = 0,
                            tint_adjust: int = 0) -> np.ndarray:
        """
        Apply white balance correction
        
        Args:
            image: Input image
            analysis: Pre-computed analysis (will analyze if None)
            temperature_adjust: Additional temperature adjustment in Kelvin
            tint_adjust: Additional tint adjustment (-100 to +100)
            
        Returns:
            White balanced image
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
            analysis = self.analyze_white_balance(image)
        
        # Get base multipliers
        r_mult = analysis.red_multiplier
        g_mult = analysis.green_multiplier
        b_mult = analysis.blue_multiplier
        
        # Apply temperature adjustment
        if temperature_adjust != 0:
            temp_mult = self._temperature_to_rgb_multipliers(
                analysis.estimated_temp_kelvin + temperature_adjust
            )
            base_mult = self._temperature_to_rgb_multipliers(analysis.estimated_temp_kelvin)
            
            # Relative adjustment
            r_mult *= temp_mult[0] / base_mult[0]
            b_mult *= temp_mult[2] / base_mult[2]
        
        # Apply tint adjustment (green/magenta)
        if tint_adjust != 0:
            tint_factor = 1.0 + (tint_adjust / 200.0)  # -100 to +100 maps to 0.5 to 1.5
            g_mult *= tint_factor
        
        # Normalize multipliers
        multipliers = self._normalize_multipliers((r_mult, g_mult, b_mult))
        
        # Apply correction
        corrected = img_float.copy()
        corrected[:, :, 0] *= multipliers[0]  # Red
        corrected[:, :, 1] *= multipliers[1]  # Green
        corrected[:, :, 2] *= multipliers[2]  # Blue
        
        # Preserve skin tones if enabled
        if self.preserve_skin_tones:
            corrected = self._preserve_skin_tones(img_float, corrected)
        
        # Clip values
        corrected = np.clip(corrected, 0, 1)
        
        # Convert back to original dtype
        if orig_dtype == np.uint8:
            return (corrected * 255).astype(np.uint8)
        else:
            return (corrected * 65535).astype(np.uint16)
    
    def _gray_world_wb(self, image: np.ndarray) -> Tuple[float, float, float]:
        """Gray world white balance algorithm"""
        # Calculate mean of each channel
        mean_r = np.mean(image[:, :, 0])
        mean_g = np.mean(image[:, :, 1])
        mean_b = np.mean(image[:, :, 2])
        
        # Target gray (average of all channels)
        mean_gray = (mean_r + mean_g + mean_b) / 3.0
        
        # Calculate multipliers
        r_mult = mean_gray / mean_r if mean_r > 0 else 1.0
        g_mult = mean_gray / mean_g if mean_g > 0 else 1.0
        b_mult = mean_gray / mean_b if mean_b > 0 else 1.0
        
        return (r_mult, g_mult, b_mult)
    
    def _white_patch_wb(self, image: np.ndarray) -> Tuple[float, float, float]:
        """White patch (max RGB) white balance algorithm"""
        # Find bright pixels (top percentile)
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        threshold = np.percentile(luminance, self.clip_percentile * 100)
        
        bright_mask = luminance >= threshold
        
        if np.sum(bright_mask) < 10:  # Too few bright pixels
            return self._gray_world_wb(image)
        
        # Calculate mean of bright pixels
        bright_r = np.mean(image[:, :, 0][bright_mask])
        bright_g = np.mean(image[:, :, 1][bright_mask])
        bright_b = np.mean(image[:, :, 2][bright_mask])
        
        # Normalize to brightest channel
        max_bright = max(bright_r, bright_g, bright_b)
        
        r_mult = max_bright / bright_r if bright_r > 0 else 1.0
        g_mult = max_bright / bright_g if bright_g > 0 else 1.0
        b_mult = max_bright / bright_b if bright_b > 0 else 1.0
        
        return (r_mult, g_mult, b_mult)
    
    def _retinex_wb(self, image: np.ndarray) -> Tuple[float, float, float]:
        """Retinex-based white balance (simplified)"""
        # Apply logarithmic transform
        epsilon = 1e-6
        log_r = np.log(image[:, :, 0] + epsilon)
        log_g = np.log(image[:, :, 1] + epsilon)
        log_b = np.log(image[:, :, 2] + epsilon)
        
        # Calculate spatial averages
        kernel_size = 15
        avg_r = cv2.GaussianBlur(log_r, (kernel_size, kernel_size), 0)
        avg_g = cv2.GaussianBlur(log_g, (kernel_size, kernel_size), 0)
        avg_b = cv2.GaussianBlur(log_b, (kernel_size, kernel_size), 0)
        
        # Retinex output
        retinex_r = log_r - avg_r
        retinex_g = log_g - avg_g
        retinex_b = log_b - avg_b
        
        # Calculate channel adjustments
        mean_retinex = (np.mean(retinex_r) + np.mean(retinex_g) + np.mean(retinex_b)) / 3
        
        r_mult = np.exp(mean_retinex - np.mean(retinex_r))
        g_mult = np.exp(mean_retinex - np.mean(retinex_g))
        b_mult = np.exp(mean_retinex - np.mean(retinex_b))
        
        return (r_mult, g_mult, b_mult)
    
    def _illuminant_estimation_wb(self, image: np.ndarray) -> Tuple[float, float, float]:
        """Illuminant estimation using chromatic adaptation"""
        # Convert to LAB color space
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # Find neutral regions (low chromatic)
        chromatic_magnitude = np.sqrt(
            (a_channel - 128).astype(np.float32)**2 + 
            (b_channel - 128).astype(np.float32)**2
        )
        
        # Threshold for neutral pixels
        neutral_threshold = np.percentile(chromatic_magnitude, 20)
        neutral_mask = chromatic_magnitude < neutral_threshold
        
        if np.sum(neutral_mask) < 100:
            return self._gray_world_wb(image)
        
        # Calculate average color of neutral regions
        neutral_r = np.mean(image[:, :, 0][neutral_mask])
        neutral_g = np.mean(image[:, :, 1][neutral_mask])
        neutral_b = np.mean(image[:, :, 2][neutral_mask])
        
        # Target neutral gray
        target_gray = (neutral_r + neutral_g + neutral_b) / 3.0
        
        r_mult = target_gray / neutral_r if neutral_r > 0 else 1.0
        g_mult = target_gray / neutral_g if neutral_g > 0 else 1.0
        b_mult = target_gray / neutral_b if neutral_b > 0 else 1.0
        
        return (r_mult, g_mult, b_mult)
    
    def _face_detection_wb(self, image: np.ndarray) -> Tuple[float, float, float]:
        """Face-based white balance for skin tone optimization"""
        # Convert to uint8 for face detection
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        # Detect faces
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            # No faces found, fall back to gray world
            return self._gray_world_wb(image)
        
        # Sample skin tones from detected faces
        skin_samples = []
        for (x, y, w, h) in faces:
            # Sample from center region of face
            center_x = x + w // 2
            center_y = y + h // 2
            sample_size = min(w, h) // 4
            
            roi = image[
                max(0, center_y - sample_size):min(image.shape[0], center_y + sample_size),
                max(0, center_x - sample_size):min(image.shape[1], center_x + sample_size)
            ]
            
            if roi.size > 0:
                skin_samples.append(roi)
        
        if not skin_samples:
            return self._gray_world_wb(image)
        
        # Calculate average skin tone
        all_skin = np.concatenate([s.reshape(-1, 3) for s in skin_samples])
        mean_skin = np.mean(all_skin, axis=0)
        
        # Target skin tone (empirically determined for natural appearance)
        # These values represent typical skin tones under neutral lighting
        target_skin_r = 0.75
        target_skin_g = 0.60
        target_skin_b = 0.50
        
        # Calculate multipliers to achieve target skin tone
        r_mult = target_skin_r / mean_skin[0] if mean_skin[0] > 0 else 1.0
        g_mult = target_skin_g / mean_skin[1] if mean_skin[1] > 0 else 1.0
        b_mult = target_skin_b / mean_skin[2] if mean_skin[2] > 0 else 1.0
        
        # Moderate the correction to avoid overcorrection
        r_mult = 1.0 + (r_mult - 1.0) * 0.5
        g_mult = 1.0 + (g_mult - 1.0) * 0.5
        b_mult = 1.0 + (b_mult - 1.0) * 0.5
        
        return (r_mult, g_mult, b_mult)
    
    def _normalize_multipliers(self, multipliers: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Normalize multipliers to preserve overall brightness"""
        r_mult, g_mult, b_mult = multipliers
        
        # Normalize so green channel is 1.0 (common practice)
        if g_mult > 0:
            r_mult /= g_mult
            b_mult /= g_mult
            g_mult = 1.0
        
        # Limit extreme corrections
        max_mult = 2.5
        min_mult = 0.4
        
        r_mult = np.clip(r_mult, min_mult, max_mult)
        g_mult = np.clip(g_mult, min_mult, max_mult)
        b_mult = np.clip(b_mult, min_mult, max_mult)
        
        return (r_mult, g_mult, b_mult)
    
    def _init_color_temp_data(self):
        """Initialize color temperature to RGB conversion data"""
        # Planckian locus approximation for color temperature
        # Based on CIE 1931 color space
        self.temp_to_rgb_table = {
            1000: (1.000, 0.337, 0.000),
            1500: (1.000, 0.465, 0.000),
            2000: (1.000, 0.549, 0.081),
            2500: (1.000, 0.616, 0.213),
            3000: (1.000, 0.673, 0.337),
            3500: (1.000, 0.724, 0.446),
            4000: (1.000, 0.770, 0.544),
            4500: (1.000, 0.812, 0.630),
            5000: (1.000, 0.851, 0.708),
            5500: (1.000, 0.887, 0.778),
            6000: (1.000, 0.920, 0.843),
            6500: (1.000, 0.952, 0.903),
            7000: (0.949, 0.944, 1.000),
            7500: (0.913, 0.918, 1.000),
            8000: (0.883, 0.896, 1.000),
            8500: (0.858, 0.877, 1.000),
            9000: (0.835, 0.860, 1.000),
            9500: (0.815, 0.846, 1.000),
            10000: (0.798, 0.833, 1.000)
        }
    
    def _temperature_to_rgb_multipliers(self, temp_k: float) -> Tuple[float, float, float]:
        """Convert color temperature to RGB multipliers"""
        # Find surrounding temperatures in table
        temps = sorted(self.temp_to_rgb_table.keys())
        
        if temp_k <= temps[0]:
            return self.temp_to_rgb_table[temps[0]]
        if temp_k >= temps[-1]:
            return self.temp_to_rgb_table[temps[-1]]
        
        # Interpolate between nearest values
        for i in range(len(temps) - 1):
            if temps[i] <= temp_k <= temps[i + 1]:
                t1, t2 = temps[i], temps[i + 1]
                rgb1 = self.temp_to_rgb_table[t1]
                rgb2 = self.temp_to_rgb_table[t2]
                
                # Linear interpolation
                alpha = (temp_k - t1) / (t2 - t1)
                r = rgb1[0] + alpha * (rgb2[0] - rgb1[0])
                g = rgb1[1] + alpha * (rgb2[1] - rgb1[1])
                b = rgb1[2] + alpha * (rgb2[2] - rgb1[2])
                
                return (r, g, b)
        
        return (1.0, 1.0, 1.0)
    
    def _estimate_color_temperature(self, image: np.ndarray, 
                                  multipliers: Tuple[float, float, float]) -> float:
        """Estimate color temperature from image and multipliers"""
        # Apply inverse multipliers to get original illuminant color
        r_mult, g_mult, b_mult = multipliers
        
        # Average color under original illuminant
        avg_r = np.mean(image[:, :, 0]) / r_mult
        avg_g = np.mean(image[:, :, 1]) / g_mult
        avg_b = np.mean(image[:, :, 2]) / b_mult
        
        # Normalize
        total = avg_r + avg_g + avg_b
        if total > 0:
            avg_r /= total
            avg_g /= total
            avg_b /= total
        
        # Find closest temperature
        best_temp = 5500  # Default daylight
        best_distance = float('inf')
        
        for temp, (tr, tg, tb) in self.temp_to_rgb_table.items():
            # Normalize table values
            t_total = tr + tg + tb
            tr /= t_total
            tg /= t_total
            tb /= t_total
            
            # Calculate distance
            distance = np.sqrt(
                (avg_r - tr)**2 + 
                (avg_g - tg)**2 + 
                (avg_b - tb)**2
            )
            
            if distance < best_distance:
                best_distance = distance
                best_temp = temp
        
        return best_temp
    
    def _analyze_color_cast(self, image: np.ndarray) -> Tuple[float, str]:
        """Analyze color cast in image"""
        # Calculate channel means
        mean_r = np.mean(image[:, :, 0])
        mean_g = np.mean(image[:, :, 1])
        mean_b = np.mean(image[:, :, 2])
        
        # Normalize to sum to 1
        total = mean_r + mean_g + mean_b
        if total > 0:
            mean_r /= total
            mean_g /= total
            mean_b /= total
        
        # Ideal neutral is (1/3, 1/3, 1/3)
        neutral = 1/3
        
        # Calculate deviations
        dev_r = mean_r - neutral
        dev_g = mean_g - neutral
        dev_b = mean_b - neutral
        
        # Cast strength is the magnitude of deviation
        cast_strength = np.sqrt(dev_r**2 + dev_g**2 + dev_b**2)
        
        # Determine dominant cast
        if abs(dev_r) > abs(dev_g) and abs(dev_r) > abs(dev_b):
            if dev_r > 0:
                dominant_cast = "red"
            else:
                dominant_cast = "cyan"
        elif abs(dev_g) > abs(dev_r) and abs(dev_g) > abs(dev_b):
            if dev_g > 0:
                dominant_cast = "green"
            else:
                dominant_cast = "magenta"
        else:
            if dev_b > 0:
                dominant_cast = "blue"
            else:
                dominant_cast = "yellow"
        
        return cast_strength, dominant_cast
    
    def _classify_illuminant(self, temp_kelvin: float) -> str:
        """Classify illuminant type based on color temperature"""
        if temp_kelvin < 3000:
            return "tungsten"
        elif temp_kelvin < 4500:
            return "fluorescent"
        elif temp_kelvin < 6000:
            return "daylight"
        else:
            return "cloudy"
    
    def _calculate_gray_deviation(self, image: np.ndarray) -> float:
        """Calculate deviation from gray world assumption"""
        mean_r = np.mean(image[:, :, 0])
        mean_g = np.mean(image[:, :, 1])
        mean_b = np.mean(image[:, :, 2])
        
        avg_mean = (mean_r + mean_g + mean_b) / 3
        
        if avg_mean > 0:
            deviation = np.std([mean_r, mean_g, mean_b]) / avg_mean
        else:
            deviation = 0.0
        
        return deviation
    
    def _calculate_highlight_preservation(self, image: np.ndarray,
                                        multipliers: Tuple[float, float, float]) -> float:
        """Calculate how well highlights are preserved"""
        # Find bright pixels
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        highlight_mask = luminance > 0.9
        
        if np.sum(highlight_mask) < 10:
            return 1.0  # No highlights to preserve
        
        # Check clipping after correction
        r_mult, g_mult, b_mult = multipliers
        
        clipped_r = np.sum(image[:, :, 0][highlight_mask] * r_mult > 1.0)
        clipped_g = np.sum(image[:, :, 1][highlight_mask] * g_mult > 1.0)
        clipped_b = np.sum(image[:, :, 2][highlight_mask] * b_mult > 1.0)
        
        total_highlights = np.sum(highlight_mask)
        clipped_ratio = (clipped_r + clipped_g + clipped_b) / (3 * total_highlights)
        
        return 1.0 - clipped_ratio
    
    def _calculate_confidence(self, method: WhiteBalanceMethod,
                            gray_deviation: float, cast_strength: float) -> float:
        """Calculate confidence in white balance analysis"""
        # Base confidence on method
        method_confidence = {
            WhiteBalanceMethod.FACE_DETECTION: 0.9,
            WhiteBalanceMethod.ILLUMINANT_ESTIMATION: 0.8,
            WhiteBalanceMethod.RETINEX: 0.7,
            WhiteBalanceMethod.WHITE_PATCH: 0.7,
            WhiteBalanceMethod.GRAY_WORLD: 0.6,
            WhiteBalanceMethod.MANUAL: 1.0
        }
        
        base_conf = method_confidence.get(method, 0.5)
        
        # Adjust based on gray deviation (lower is better)
        gray_factor = np.exp(-gray_deviation * 2)
        
        # Adjust based on color cast (moderate cast is normal)
        if cast_strength < 0.1:
            cast_factor = 0.8  # Too neutral, might be pre-corrected
        elif cast_strength < 0.3:
            cast_factor = 1.0  # Normal range
        else:
            cast_factor = 0.6  # Strong cast, less confident
        
        return base_conf * gray_factor * cast_factor
    
    def _select_best_method(self, image: np.ndarray) -> WhiteBalanceMethod:
        """Auto-select best white balance method based on image content"""
        h, w = image.shape[:2]
        
        # Check for faces
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            return WhiteBalanceMethod.FACE_DETECTION
        
        # Check for highlight clipping
        max_vals = np.max(image.reshape(-1, 3), axis=0)
        if np.any(max_vals > 0.95):
            return WhiteBalanceMethod.WHITE_PATCH
        
        # Check scene variance
        variance = np.var(image)
        if variance > 0.1:
            return WhiteBalanceMethod.ILLUMINANT_ESTIMATION
        
        # Default to gray world
        return WhiteBalanceMethod.GRAY_WORLD
    
    def _preserve_skin_tones(self, original: np.ndarray, corrected: np.ndarray) -> np.ndarray:
        """Preserve skin tones during white balance correction"""
        # Convert to HSV for skin detection
        orig_hsv = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Skin tone detection in HSV
        # Hue: 0-20 and 340-360 degrees (in OpenCV: 0-10 and 170-180)
        # Saturation: 30-150
        # Value: 80-255
        lower_skin1 = np.array([0, 30, 80], dtype=np.uint8)
        upper_skin1 = np.array([10, 150, 255], dtype=np.uint8)
        lower_skin2 = np.array([170, 30, 80], dtype=np.uint8)
        upper_skin2 = np.array([180, 150, 255], dtype=np.uint8)
        
        skin_mask1 = cv2.inRange(orig_hsv, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(orig_hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Dilate mask for smoother blending
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (21, 21), 0)
        
        # Normalize mask
        skin_mask_float = skin_mask.astype(np.float32) / 255.0
        skin_mask_float = skin_mask_float[:, :, np.newaxis]
        
        # Blend original and corrected based on mask
        result = corrected * (1 - skin_mask_float * 0.5) + original * (skin_mask_float * 0.5)
        
        return result