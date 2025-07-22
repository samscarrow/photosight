"""
Color grading module for PhotoSight

Provides creative color adjustments including vibrance, saturation,
color wheels, and preset looks.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ColorGradingPreset(Enum):
    """Available color grading presets"""
    NONE = "none"
    CINEMATIC = "cinematic"
    VINTAGE = "vintage"
    MOODY = "moody"
    BRIGHT_AIRY = "bright_airy"
    FILM_EMULATION = "film_emulation"
    CROSS_PROCESS = "cross_process"
    BLEACH_BYPASS = "bleach_bypass"
    TEAL_ORANGE = "teal_orange"
    CUSTOM = "custom"


@dataclass
class ColorGradingSettings:
    """Color grading parameters"""
    # Global adjustments
    vibrance: float = 0.0  # -100 to +100
    saturation: float = 0.0  # -100 to +100
    
    # Color mixer (RGB channels)
    red_hue: float = 0.0  # -180 to +180
    red_saturation: float = 0.0  # -100 to +100
    red_luminance: float = 0.0  # -100 to +100
    
    green_hue: float = 0.0
    green_saturation: float = 0.0
    green_luminance: float = 0.0
    
    blue_hue: float = 0.0
    blue_saturation: float = 0.0
    blue_luminance: float = 0.0
    
    # Split toning
    highlight_hue: float = 0.0  # 0-360 degrees
    highlight_saturation: float = 0.0  # 0-100
    shadow_hue: float = 0.0  # 0-360 degrees
    shadow_saturation: float = 0.0  # 0-100
    split_balance: float = 0.0  # -100 to +100
    
    # Color wheels (lift/gamma/gain or shadows/midtones/highlights)
    shadows_r: float = 0.0  # -1 to +1
    shadows_g: float = 0.0
    shadows_b: float = 0.0
    
    midtones_r: float = 0.0
    midtones_g: float = 0.0
    midtones_b: float = 0.0
    
    highlights_r: float = 0.0
    highlights_g: float = 0.0
    highlights_b: float = 0.0
    
    # Advanced
    color_contrast: float = 0.0  # -100 to +100
    luminosity_masks: bool = True
    preserve_skin_tones: bool = True
    
    # Preset info
    preset: ColorGradingPreset = ColorGradingPreset.NONE
    preset_strength: float = 100.0  # 0-100


class ColorGrader:
    """
    Advanced color grading engine
    
    Features:
    - Vibrance and saturation controls
    - HSL color mixer
    - Split toning
    - Three-way color wheels
    - Creative presets
    """
    
    def __init__(self):
        """Initialize color grader"""
        self.presets = self._init_presets()
    
    def apply_color_grading(self, image: np.ndarray,
                          settings: ColorGradingSettings) -> np.ndarray:
        """
        Apply color grading to image
        
        Args:
            image: Input image (RGB)
            settings: Color grading settings
            
        Returns:
            Color graded image
        """
        # Convert to float
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_float = image.astype(np.float32) / 65535.0
        else:
            img_float = image.copy()
        
        # Apply preset if selected
        if settings.preset != ColorGradingPreset.NONE and settings.preset != ColorGradingPreset.CUSTOM:
            preset_settings = self.get_preset(settings.preset)
            settings = self._blend_settings(settings, preset_settings, settings.preset_strength / 100.0)
        
        # Apply global adjustments
        if settings.vibrance != 0:
            img_float = self._apply_vibrance(img_float, settings.vibrance)
        
        if settings.saturation != 0:
            img_float = self._apply_saturation(img_float, settings.saturation)
        
        # Apply HSL color mixer
        if any([settings.red_hue, settings.red_saturation, settings.red_luminance,
                settings.green_hue, settings.green_saturation, settings.green_luminance,
                settings.blue_hue, settings.blue_saturation, settings.blue_luminance]):
            img_float = self._apply_hsl_mixer(img_float, settings)
        
        # Apply color wheels
        if any([settings.shadows_r, settings.shadows_g, settings.shadows_b,
                settings.midtones_r, settings.midtones_g, settings.midtones_b,
                settings.highlights_r, settings.highlights_g, settings.highlights_b]):
            img_float = self._apply_color_wheels(img_float, settings)
        
        # Apply split toning
        if settings.highlight_saturation > 0 or settings.shadow_saturation > 0:
            img_float = self._apply_split_toning(img_float, settings)
        
        # Apply color contrast
        if settings.color_contrast != 0:
            img_float = self._apply_color_contrast(img_float, settings.color_contrast)
        
        # Preserve skin tones if enabled
        if settings.preserve_skin_tones:
            img_float = self._preserve_skin_tones(image.astype(np.float32) / 
                                                (255 if image.dtype == np.uint8 else 65535),
                                                img_float)
        
        # Convert back to original dtype
        img_float = np.clip(img_float, 0, 1)
        if image.dtype == np.uint8:
            return (img_float * 255).astype(np.uint8)
        else:
            return (img_float * 65535).astype(np.uint16)
    
    def _apply_vibrance(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Apply vibrance (smart saturation that protects skin tones)"""
        # Convert to HSV
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Calculate saturation mask (less saturated pixels get more boost)
        saturation = hsv[:, :, 1] / 255.0
        saturation_mask = 1.0 - saturation  # Inverse mask
        
        # Skin tone protection mask
        hue = hsv[:, :, 0]
        skin_mask = np.logical_or(
            np.logical_and(hue >= 0, hue <= 25),  # Red-orange hues
            np.logical_and(hue >= 340, hue <= 360)
        ).astype(np.float32)
        
        # Combine masks
        vibrance_mask = saturation_mask * (1.0 - skin_mask * 0.7)
        
        # Apply vibrance
        vibrance_factor = 1.0 + (amount / 100.0) * vibrance_mask
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * vibrance_factor, 0, 255)
        
        # Convert back to RGB
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255.0
    
    def _apply_saturation(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Apply global saturation adjustment"""
        # Convert to grayscale for luminance
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Blend based on saturation amount
        factor = (amount + 100.0) / 100.0
        result = gray + (img - gray) * factor
        
        return np.clip(result, 0, 1)
    
    def _apply_hsl_mixer(self, img: np.ndarray, settings: ColorGradingSettings) -> np.ndarray:
        """Apply HSL adjustments to specific color ranges"""
        # Convert to HSV for easier manipulation
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # Define color ranges in HSV
        # Red: 0-10 and 170-180, Green: 50-70, Blue: 100-130
        red_mask = np.logical_or(h < 10, h > 170).astype(np.float32)
        green_mask = np.logical_and(h >= 50, h <= 70).astype(np.float32)
        blue_mask = np.logical_and(h >= 100, h <= 130).astype(np.float32)
        
        # Smooth masks
        red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
        green_mask = cv2.GaussianBlur(green_mask, (5, 5), 0)
        blue_mask = cv2.GaussianBlur(blue_mask, (5, 5), 0)
        
        # Apply hue shifts
        if settings.red_hue != 0:
            h = h + settings.red_hue * red_mask
        if settings.green_hue != 0:
            h = h + settings.green_hue * green_mask
        if settings.blue_hue != 0:
            h = h + settings.blue_hue * blue_mask
        
        # Wrap hue values
        h = np.mod(h, 180)
        
        # Apply saturation adjustments
        if settings.red_saturation != 0:
            s = s * (1 + settings.red_saturation / 100.0 * red_mask)
        if settings.green_saturation != 0:
            s = s * (1 + settings.green_saturation / 100.0 * green_mask)
        if settings.blue_saturation != 0:
            s = s * (1 + settings.blue_saturation / 100.0 * blue_mask)
        
        s = np.clip(s, 0, 255)
        
        # Apply luminance adjustments
        if settings.red_luminance != 0:
            v = v * (1 + settings.red_luminance / 100.0 * red_mask)
        if settings.green_luminance != 0:
            v = v * (1 + settings.green_luminance / 100.0 * green_mask)
        if settings.blue_luminance != 0:
            v = v * (1 + settings.blue_luminance / 100.0 * blue_mask)
        
        v = np.clip(v, 0, 255)
        
        # Reconstruct HSV and convert back
        hsv = np.stack([h, s, v], axis=2).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255.0
    
    def _apply_color_wheels(self, img: np.ndarray, settings: ColorGradingSettings) -> np.ndarray:
        """Apply three-way color grading (shadows/midtones/highlights)"""
        # Create luminosity masks
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        
        # Shadow mask (dark areas)
        shadow_mask = np.maximum(0, 1 - luminance * 2)
        shadow_mask = shadow_mask ** 2  # Smooth falloff
        
        # Highlight mask (bright areas)
        highlight_mask = np.maximum(0, luminance * 2 - 1)
        highlight_mask = highlight_mask ** 2
        
        # Midtone mask (everything else)
        midtone_mask = 1 - shadow_mask - highlight_mask
        midtone_mask = np.maximum(0, midtone_mask)
        
        # Expand masks to 3 channels
        shadow_mask = shadow_mask[:, :, np.newaxis]
        midtone_mask = midtone_mask[:, :, np.newaxis]
        highlight_mask = highlight_mask[:, :, np.newaxis]
        
        # Apply color adjustments
        result = img.copy()
        
        # Shadows
        if any([settings.shadows_r, settings.shadows_g, settings.shadows_b]):
            shadow_adjustment = np.zeros_like(img)
            shadow_adjustment[:, :, 0] = settings.shadows_r
            shadow_adjustment[:, :, 1] = settings.shadows_g
            shadow_adjustment[:, :, 2] = settings.shadows_b
            result = result + shadow_adjustment * shadow_mask * 0.5
        
        # Midtones
        if any([settings.midtones_r, settings.midtones_g, settings.midtones_b]):
            midtone_adjustment = np.zeros_like(img)
            midtone_adjustment[:, :, 0] = settings.midtones_r
            midtone_adjustment[:, :, 1] = settings.midtones_g
            midtone_adjustment[:, :, 2] = settings.midtones_b
            result = result + midtone_adjustment * midtone_mask * 0.5
        
        # Highlights
        if any([settings.highlights_r, settings.highlights_g, settings.highlights_b]):
            highlight_adjustment = np.zeros_like(img)
            highlight_adjustment[:, :, 0] = settings.highlights_r
            highlight_adjustment[:, :, 1] = settings.highlights_g
            highlight_adjustment[:, :, 2] = settings.highlights_b
            result = result + highlight_adjustment * highlight_mask * 0.5
        
        return np.clip(result, 0, 1)
    
    def _apply_split_toning(self, img: np.ndarray, settings: ColorGradingSettings) -> np.ndarray:
        """Apply split toning (different colors to highlights and shadows)"""
        # Create luminosity masks
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        
        # Balance point (0 = midpoint at 0.5, +100 = all highlights, -100 = all shadows)
        balance_point = 0.5 + settings.split_balance / 200.0
        
        # Create masks with smooth transition
        highlight_mask = np.maximum(0, (luminance - balance_point) / (1 - balance_point))
        shadow_mask = np.maximum(0, (balance_point - luminance) / balance_point)
        
        # Apply falloff
        highlight_mask = highlight_mask ** 1.5
        shadow_mask = shadow_mask ** 1.5
        
        result = img.copy()
        
        # Apply highlight toning
        if settings.highlight_saturation > 0:
            highlight_color = self._hue_to_rgb(settings.highlight_hue)
            tint_strength = settings.highlight_saturation / 100.0 * 0.3
            
            for i in range(3):
                result[:, :, i] = result[:, :, i] + \
                                (highlight_color[i] - result[:, :, i]) * \
                                highlight_mask * tint_strength
        
        # Apply shadow toning
        if settings.shadow_saturation > 0:
            shadow_color = self._hue_to_rgb(settings.shadow_hue)
            tint_strength = settings.shadow_saturation / 100.0 * 0.3
            
            for i in range(3):
                result[:, :, i] = result[:, :, i] + \
                                (shadow_color[i] - result[:, :, i]) * \
                                shadow_mask * tint_strength
        
        return np.clip(result, 0, 1)
    
    def _apply_color_contrast(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Apply color contrast (increases saturation in midtones)"""
        # Convert to LAB color space
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Apply S-curve to a and b channels
        factor = 1.0 + amount / 100.0
        
        # Center around neutral (128)
        lab[:, :, 1] = 128 + (lab[:, :, 1] - 128) * factor
        lab[:, :, 2] = 128 + (lab[:, :, 2] - 128) * factor
        
        # Clip values
        lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)
        
        # Convert back to RGB
        rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return rgb.astype(np.float32) / 255.0
    
    def _hue_to_rgb(self, hue: float) -> Tuple[float, float, float]:
        """Convert hue (0-360) to RGB color"""
        h = hue / 60.0
        c = 1.0
        x = c * (1 - abs(h % 2 - 1))
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (r, g, b)
    
    def _preserve_skin_tones(self, original: np.ndarray, graded: np.ndarray) -> np.ndarray:
        """Preserve skin tones during color grading"""
        # Use the same skin detection as in white balance
        orig_hsv = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Skin tone ranges
        lower_skin1 = np.array([0, 30, 80], dtype=np.uint8)
        upper_skin1 = np.array([10, 150, 255], dtype=np.uint8)
        lower_skin2 = np.array([170, 30, 80], dtype=np.uint8)
        upper_skin2 = np.array([180, 150, 255], dtype=np.uint8)
        
        skin_mask1 = cv2.inRange(orig_hsv, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(orig_hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Smooth mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (21, 21), 0)
        
        # Normalize and expand
        skin_mask_float = skin_mask.astype(np.float32) / 255.0
        skin_mask_float = skin_mask_float[:, :, np.newaxis]
        
        # Blend with more weight to original in skin areas
        result = graded * (1 - skin_mask_float * 0.6) + original * (skin_mask_float * 0.6)
        
        return result
    
    def _init_presets(self) -> Dict[str, ColorGradingSettings]:
        """Initialize color grading presets"""
        presets = {}
        
        # Cinematic look
        presets[ColorGradingPreset.CINEMATIC] = ColorGradingSettings(
            vibrance=-10,
            saturation=-5,
            shadows_r=-0.05,
            shadows_g=-0.02,
            shadows_b=0.05,  # Slight blue in shadows
            highlights_r=0.08,
            highlights_g=0.05,
            highlights_b=-0.02,  # Warm highlights
            color_contrast=15,
            highlight_hue=30,  # Orange
            highlight_saturation=20,
            shadow_hue=210,  # Blue
            shadow_saturation=15,
            split_balance=10
        )
        
        # Vintage/Film look
        presets[ColorGradingPreset.VINTAGE] = ColorGradingSettings(
            vibrance=-20,
            saturation=-15,
            red_luminance=5,
            green_saturation=-10,
            blue_saturation=-20,
            shadows_r=0.08,
            shadows_g=0.05,
            shadows_b=0.02,  # Lifted blacks with warm tint
            midtones_r=0.05,
            midtones_g=0.03,
            midtones_b=-0.02,
            color_contrast=-10
        )
        
        # Moody/Dark look
        presets[ColorGradingPreset.MOODY] = ColorGradingSettings(
            vibrance=-30,
            saturation=-20,
            shadows_r=-0.1,
            shadows_g=-0.08,
            shadows_b=-0.05,  # Crushed shadows
            midtones_r=-0.03,
            midtones_g=-0.02,
            midtones_b=0.02,
            color_contrast=20,
            highlight_saturation=10,
            shadow_saturation=25,
            shadow_hue=220,  # Blue shadows
            split_balance=-20
        )
        
        # Bright and Airy
        presets[ColorGradingPreset.BRIGHT_AIRY] = ColorGradingSettings(
            vibrance=25,
            saturation=5,
            shadows_r=0.15,
            shadows_g=0.15,
            shadows_b=0.18,  # Lifted shadows with slight blue
            midtones_r=0.02,
            midtones_g=0.03,
            midtones_b=0.05,
            highlights_r=0.0,
            highlights_g=0.02,
            highlights_b=0.05,
            color_contrast=-15
        )
        
        # Film Emulation
        presets[ColorGradingPreset.FILM_EMULATION] = ColorGradingSettings(
            vibrance=10,
            saturation=-5,
            red_hue=-5,
            red_saturation=10,
            green_hue=5,
            green_saturation=-15,
            blue_saturation=-10,
            shadows_r=0.02,
            shadows_g=0.04,
            shadows_b=0.06,
            highlights_r=0.06,
            highlights_g=0.04,
            highlights_b=0.0,
            color_contrast=10
        )
        
        # Cross Process
        presets[ColorGradingPreset.CROSS_PROCESS] = ColorGradingSettings(
            vibrance=30,
            saturation=20,
            red_hue=15,
            green_hue=-10,
            blue_hue=20,
            shadows_r=-0.1,
            shadows_g=0.05,
            shadows_b=0.15,  # Strong blue shadows
            highlights_r=0.15,
            highlights_g=0.1,
            highlights_b=-0.1,  # Yellow highlights
            color_contrast=30
        )
        
        # Bleach Bypass
        presets[ColorGradingPreset.BLEACH_BYPASS] = ColorGradingSettings(
            vibrance=-40,
            saturation=-50,
            shadows_r=-0.05,
            shadows_g=-0.05,
            shadows_b=-0.05,
            midtones_r=0.0,
            midtones_g=0.0,
            midtones_b=0.0,
            highlights_r=0.05,
            highlights_g=0.05,
            highlights_b=0.05,
            color_contrast=25
        )
        
        # Teal and Orange
        presets[ColorGradingPreset.TEAL_ORANGE] = ColorGradingSettings(
            vibrance=15,
            shadow_hue=180,  # Teal
            shadow_saturation=30,
            highlight_hue=30,  # Orange
            highlight_saturation=25,
            split_balance=0,
            midtones_r=0.05,
            midtones_g=-0.02,
            midtones_b=-0.05,
            color_contrast=10
        )
        
        return presets
    
    def get_preset(self, preset: ColorGradingPreset) -> ColorGradingSettings:
        """Get a preset configuration"""
        return self.presets.get(preset, ColorGradingSettings())
    
    def _blend_settings(self, base: ColorGradingSettings, 
                       preset: ColorGradingSettings,
                       blend_amount: float) -> ColorGradingSettings:
        """Blend two settings based on amount (0-1)"""
        result = ColorGradingSettings()
        
        # Blend all numeric attributes
        for attr in dir(base):
            if not attr.startswith('_'):
                base_val = getattr(base, attr)
                preset_val = getattr(preset, attr)
                
                if isinstance(base_val, (int, float)):
                    blended = base_val + (preset_val - base_val) * blend_amount
                    setattr(result, attr, blended)
                elif isinstance(base_val, bool):
                    # For booleans, use preset if blend > 0.5
                    setattr(result, attr, preset_val if blend_amount > 0.5 else base_val)
                else:
                    # For other types, keep base value
                    setattr(result, attr, base_val)
        
        return result
    
    def create_lut(self, settings: ColorGradingSettings, 
                   lut_size: int = 64) -> np.ndarray:
        """
        Create a 3D LUT from color grading settings
        
        Args:
            settings: Color grading settings
            lut_size: Size of the LUT cube (default 64x64x64)
            
        Returns:
            3D LUT array
        """
        # Create identity LUT
        lut = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.float32)
        
        for r in range(lut_size):
            for g in range(lut_size):
                for b in range(lut_size):
                    # Normalize to 0-1
                    color = np.array([r, g, b], dtype=np.float32) / (lut_size - 1)
                    
                    # Apply color grading
                    graded = self.apply_color_grading(
                        color.reshape(1, 1, 3),
                        settings
                    )
                    
                    lut[r, g, b] = graded[0, 0]
        
        return lut