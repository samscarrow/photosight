"""
Local adjustment processor that applies masked adjustments to images.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2
from dataclasses import dataclass
import logging

from .models import AdjustmentLayer, LocalAdjustmentConfig
from .mask_generator import MaskGenerator

logger = logging.getLogger(__name__)


class LocalAdjustmentProcessor:
    """Applies local adjustments to images using generated masks."""
    
    def __init__(self, config: Optional[LocalAdjustmentConfig] = None):
        """Initialize the local adjustment processor."""
        self.config = config or LocalAdjustmentConfig()
        self.mask_cache = {} if self.config.cache_masks else None
        
    def apply_adjustment_layers(self, image: np.ndarray, 
                              layers: List[AdjustmentLayer]) -> np.ndarray:
        """
        Apply all adjustment layers to an image.
        
        Args:
            image: Input image (float32, 0-1 range, RGB)
            layers: List of adjustment layers to apply
            
        Returns:
            Processed image with all adjustments applied
        """
        if not layers:
            return image
        
        # Ensure image is float32 and in correct range
        working_image = self._prepare_image(image)
        
        # Apply each enabled layer
        for layer in layers:
            if not layer.enabled or not layer.has_adjustments():
                continue
                
            try:
                working_image = self._apply_single_layer(working_image, layer)
            except Exception as e:
                logger.error(f"Failed to apply layer '{layer.name}': {e}")
                continue
        
        return working_image
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for processing."""
        # Convert to float32 if needed
        if image.dtype != np.float32:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            else:
                image = image.astype(np.float32)
        
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        return np.clip(image, 0, 1)
    
    def _apply_single_layer(self, image: np.ndarray, 
                           layer: AdjustmentLayer) -> np.ndarray:
        """Apply a single adjustment layer to the image."""
        height, width = image.shape[:2]
        
        # Generate or retrieve mask
        mask = self._get_mask(image, layer, (height, width))
        
        # Apply mask refinements
        if layer.mask_blur > 0 or layer.mask_contrast != 1.0:
            mask = MaskGenerator.refine_mask(
                mask, 
                blur_radius=layer.mask_blur,
                contrast=layer.mask_contrast
            )
        
        # Create adjusted version of the image
        adjusted = self._apply_adjustments(image, layer.adjustments)
        
        # Blend based on mask and layer settings
        result = self._blend_images(
            image, adjusted, mask, 
            layer.blend_mode, layer.opacity
        )
        
        return result
    
    def _get_mask(self, image: np.ndarray, layer: AdjustmentLayer,
                  shape: Tuple[int, int]) -> np.ndarray:
        """Generate or retrieve cached mask for a layer."""
        if not layer.mask_data:
            # No mask means full image adjustment
            return np.ones(shape, dtype=np.float32)
        
        # Check cache if enabled
        cache_key = None
        if self.mask_cache is not None:
            cache_key = f"{layer.name}_{shape}_{hash(str(layer.mask_data))}"
            if cache_key in self.mask_cache:
                return self.mask_cache[cache_key]
        
        # Generate mask
        mask = MaskGenerator.generate_mask(shape, layer.mask_data, image)
        
        # Cache if enabled
        if cache_key and self.mask_cache is not None:
            self.mask_cache[cache_key] = mask
        
        # Save mask for debugging if requested
        if self.config.save_masks and self.config.mask_save_path:
            self._save_debug_mask(mask, layer.name)
        
        return mask
    
    def _apply_adjustments(self, image: np.ndarray, 
                          adjustments: Dict[str, float]) -> np.ndarray:
        """Apply adjustment values to create modified image."""
        adjusted = image.copy()
        
        # Exposure adjustment (in stops/EV)
        if 'exposure' in adjustments and adjustments['exposure'] != 0:
            exposure_factor = 2.0 ** adjustments['exposure']
            adjusted = adjusted * exposure_factor
        
        # Contrast adjustment
        if 'contrast' in adjustments and adjustments['contrast'] != 0:
            contrast = adjustments['contrast'] / 100.0  # -1 to 1
            # Apply S-curve for contrast
            adjusted = self._apply_contrast(adjusted, contrast)
        
        # Highlights and shadows (tone mapping)
        if 'highlights' in adjustments or 'shadows' in adjustments:
            adjusted = self._apply_tone_adjustments(
                adjusted,
                highlights=adjustments.get('highlights', 0) / 100.0,
                shadows=adjustments.get('shadows', 0) / 100.0
            )
        
        # Whites and blacks (endpoint adjustment)
        if 'whites' in adjustments or 'blacks' in adjustments:
            adjusted = self._apply_endpoint_adjustments(
                adjusted,
                whites=adjustments.get('whites', 0) / 100.0,
                blacks=adjustments.get('blacks', 0) / 100.0
            )
        
        # Clarity (local contrast)
        if 'clarity' in adjustments and adjustments['clarity'] != 0:
            adjusted = self._apply_clarity(adjusted, adjustments['clarity'] / 100.0)
        
        # Vibrance and saturation
        if 'vibrance' in adjustments or 'saturation' in adjustments:
            adjusted = self._apply_color_adjustments(
                adjusted,
                vibrance=adjustments.get('vibrance', 0) / 100.0,
                saturation=adjustments.get('saturation', 0) / 100.0
            )
        
        # Temperature and tint
        if 'temperature' in adjustments or 'tint' in adjustments:
            adjusted = self._apply_color_temperature(
                adjusted,
                temperature=adjustments.get('temperature', 0) / 100.0,
                tint=adjustments.get('tint', 0) / 100.0
            )
        
        return np.clip(adjusted, 0, 1)
    
    def _apply_contrast(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Apply contrast adjustment using S-curve."""
        if amount == 0:
            return image
        
        # Simple contrast formula around midpoint
        midpoint = 0.5
        if amount > 0:
            # Increase contrast
            factor = 1 + amount
            return np.clip((image - midpoint) * factor + midpoint, 0, 1)
        else:
            # Decrease contrast
            factor = 1 + amount  # amount is negative
            return (image - midpoint) * factor + midpoint
    
    def _apply_tone_adjustments(self, image: np.ndarray, 
                               highlights: float, shadows: float) -> np.ndarray:
        """Apply highlights and shadows adjustments."""
        result = image.copy()
        
        # Create luminance map for masking
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        
        if highlights != 0:
            # Affect bright areas more than dark
            highlight_mask = np.power(luminance, 2)[:, :, np.newaxis]
            result = result * (1 + highlights * highlight_mask)
        
        if shadows != 0:
            # Affect dark areas more than bright
            shadow_mask = np.power(1 - luminance, 2)[:, :, np.newaxis]
            result = result + shadows * shadow_mask * (1 - result)
        
        return np.clip(result, 0, 1)
    
    def _apply_endpoint_adjustments(self, image: np.ndarray,
                                   whites: float, blacks: float) -> np.ndarray:
        """Adjust white and black points."""
        result = image.copy()
        
        # Adjust black point
        if blacks != 0:
            if blacks > 0:
                # Lift blacks
                result = result * (1 - blacks) + blacks
            else:
                # Crush blacks
                result = result * (1 + blacks)
        
        # Adjust white point
        if whites != 0:
            if whites > 0:
                # Extend whites
                result = result / (1 - whites)
            else:
                # Compress whites
                result = result * (1 + whites)
        
        return np.clip(result, 0, 1)
    
    def _apply_clarity(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Apply clarity (local contrast enhancement)."""
        if amount == 0:
            return image
        
        # Convert to LAB for better local contrast
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32) / 255.0
        
        # Create edge-aware local contrast map
        blurred = cv2.GaussianBlur(l_channel, (15, 15), 0)
        detail = l_channel - blurred
        
        # Enhance detail
        enhanced_l = l_channel + detail * amount * 2
        
        # Put back enhanced luminance
        lab[:, :, 0] = np.clip(enhanced_l * 255, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        return result
    
    def _apply_color_adjustments(self, image: np.ndarray,
                                vibrance: float, saturation: float) -> np.ndarray:
        """Apply vibrance and saturation adjustments."""
        if vibrance == 0 and saturation == 0:
            return image
        
        # Convert to HSV
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] /= 255.0  # Normalize saturation channel
        
        if saturation != 0:
            # Global saturation adjustment
            hsv[:, :, 1] *= (1 + saturation)
        
        if vibrance != 0:
            # Vibrance: smart saturation that protects skin tones
            # and already saturated colors
            sat_mask = 1 - hsv[:, :, 1]  # Less effect on saturated colors
            hsv[:, :, 1] += vibrance * sat_mask * hsv[:, :, 1]
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
        hsv[:, :, 1] *= 255
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return result
    
    def _apply_color_temperature(self, image: np.ndarray,
                                temperature: float, tint: float) -> np.ndarray:
        """Apply color temperature and tint adjustments."""
        if temperature == 0 and tint == 0:
            return image
        
        result = image.copy()
        
        # Temperature adjustment (blue-yellow axis)
        if temperature != 0:
            if temperature > 0:
                # Warmer (more yellow/red)
                result[:, :, 0] *= (1 + temperature * 0.3)  # Red
                result[:, :, 2] *= (1 - temperature * 0.3)  # Blue
            else:
                # Cooler (more blue)
                result[:, :, 0] *= (1 + temperature * 0.3)  # Red
                result[:, :, 2] *= (1 - temperature * 0.3)  # Blue
        
        # Tint adjustment (green-magenta axis)
        if tint != 0:
            if tint > 0:
                # More magenta (less green)
                result[:, :, 1] *= (1 - tint * 0.3)  # Green
            else:
                # More green
                result[:, :, 1] *= (1 - tint * 0.3)  # Green
        
        return np.clip(result, 0, 1)
    
    def _blend_images(self, base: np.ndarray, overlay: np.ndarray,
                     mask: np.ndarray, blend_mode: str, opacity: float) -> np.ndarray:
        """Blend two images using mask and blend mode."""
        # Ensure mask is 3-channel
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        # Apply opacity to mask
        effective_mask = mask * opacity
        
        if blend_mode == 'normal':
            # Standard alpha blending
            result = overlay * effective_mask + base * (1 - effective_mask)
        elif blend_mode == 'multiply':
            blended = base * overlay
            result = blended * effective_mask + base * (1 - effective_mask)
        elif blend_mode == 'screen':
            blended = 1 - (1 - base) * (1 - overlay)
            result = blended * effective_mask + base * (1 - effective_mask)
        elif blend_mode == 'overlay':
            # Overlay blend mode
            low = 2 * base * overlay
            high = 1 - 2 * (1 - base) * (1 - overlay)
            blended = np.where(base < 0.5, low, high)
            result = blended * effective_mask + base * (1 - effective_mask)
        else:
            # Default to normal blend
            result = overlay * effective_mask + base * (1 - effective_mask)
        
        return np.clip(result, 0, 1)
    
    def _save_debug_mask(self, mask: np.ndarray, layer_name: str):
        """Save mask image for debugging."""
        try:
            from pathlib import Path
            import os
            
            save_dir = Path(self.config.mask_save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to uint8 for saving
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Save with descriptive filename
            filename = f"mask_{layer_name.replace(' ', '_')}.png"
            filepath = save_dir / filename
            
            cv2.imwrite(str(filepath), mask_uint8)
            logger.debug(f"Saved debug mask: {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug mask: {e}")
    
    def clear_mask_cache(self):
        """Clear the mask cache to free memory."""
        if self.mask_cache is not None:
            self.mask_cache.clear()
            logger.debug("Cleared mask cache")