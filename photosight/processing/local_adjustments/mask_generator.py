"""
Mask generation strategies for local adjustments.
"""

import numpy as np
from typing import Tuple, Optional
import cv2
from scipy.ndimage import gaussian_filter

from .models import MaskData, MaskType


class MaskGenerator:
    """Generates masks for local adjustments based on various strategies."""
    
    @staticmethod
    def generate_mask(image_shape: Tuple[int, int], mask_data: MaskData,
                     image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a mask based on the specified mask data.
        
        Args:
            image_shape: (height, width) of the target image
            mask_data: MaskData object specifying mask type and parameters
            image: Optional image data for luminosity/color-based masks
            
        Returns:
            Grayscale mask as float32 array (0-1)
        """
        if mask_data.type == MaskType.RADIAL:
            mask = MaskGenerator._generate_radial_mask(image_shape, mask_data.parameters)
        elif mask_data.type == MaskType.LINEAR:
            mask = MaskGenerator._generate_linear_mask(image_shape, mask_data.parameters)
        elif mask_data.type == MaskType.LUMINOSITY:
            if image is None:
                raise ValueError("Image required for luminosity mask")
            mask = MaskGenerator._generate_luminosity_mask(image, mask_data.parameters)
        elif mask_data.type == MaskType.COLOR_RANGE:
            if image is None:
                raise ValueError("Image required for color range mask")
            mask = MaskGenerator._generate_color_range_mask(image, mask_data.parameters)
        else:
            raise ValueError(f"Unsupported mask type: {mask_data.type}")
        
        # Apply opacity
        if mask_data.opacity < 1.0:
            mask *= mask_data.opacity
        
        return mask.astype(np.float32)
    
    @staticmethod
    def _generate_radial_mask(shape: Tuple[int, int], params: dict) -> np.ndarray:
        """Generate a radial gradient mask."""
        height, width = shape
        center = params['center']  # Normalized (0-1)
        radius = params['radius']  # Normalized (rx, ry)
        feather = params['feather']
        invert = params.get('invert', False)
        
        # Convert normalized coordinates to pixels
        cx = center[0] * width
        cy = center[1] * height
        rx = radius[0] * width
        ry = radius[1] * height
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Calculate normalized distance from center (elliptical)
        dist = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
        
        # Apply feathering
        if feather > 0:
            # Smooth transition from 1 to 0 over feather range
            inner_radius = 1.0 - feather
            mask = np.clip((1.0 - dist) / feather + (1.0 - inner_radius) / feather, 0, 1)
        else:
            # Hard edge
            mask = (dist <= 1.0).astype(np.float32)
        
        if invert:
            mask = 1.0 - mask
        
        return mask
    
    @staticmethod
    def _generate_linear_mask(shape: Tuple[int, int], params: dict) -> np.ndarray:
        """Generate a linear gradient mask."""
        height, width = shape
        start = params['start_point']  # Normalized (0-1)
        end = params['end_point']      # Normalized (0-1)
        feather = params['feather']
        invert = params.get('invert', False)
        
        # Convert normalized coordinates to pixels
        x1, y1 = start[0] * width, start[1] * height
        x2, y2 = end[0] * width, end[1] * height
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Calculate gradient direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 1e-6:
            # Degenerate case: start and end are the same
            return np.ones(shape, dtype=np.float32)
        
        # Normalize direction vector
        dx /= length
        dy /= length
        
        # Project each pixel onto the gradient line
        projection = (x - x1) * dx + (y - y1) * dy
        
        # Normalize projection to 0-1 range
        normalized_proj = projection / length
        
        # Apply feathering
        if feather > 0:
            # Extend the gradient beyond start/end points by feather amount
            extended_start = -feather
            extended_end = 1.0 + feather
            mask = np.clip((normalized_proj - extended_start) / (extended_end - extended_start), 0, 1)
        else:
            # Hard edge at start and end
            mask = np.clip(normalized_proj, 0, 1)
        
        if invert:
            mask = 1.0 - mask
        
        return mask.astype(np.float32)
    
    @staticmethod
    def _generate_luminosity_mask(image: np.ndarray, params: dict) -> np.ndarray:
        """Generate a mask based on luminosity range."""
        lum_range = params['range']
        feather = params['feather']
        invert = params.get('invert', False)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            # Use perceptual luminance weights
            luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            luminance = image
        
        # Ensure luminance is in 0-1 range
        if luminance.max() > 1.0:
            luminance = luminance / 255.0
        
        min_lum, max_lum = lum_range
        
        if feather > 0:
            # Smooth transition at boundaries
            lower_mask = np.clip((luminance - min_lum) / feather, 0, 1)
            upper_mask = np.clip((max_lum - luminance) / feather, 0, 1)
            mask = lower_mask * upper_mask
        else:
            # Hard selection
            mask = ((luminance >= min_lum) & (luminance <= max_lum)).astype(np.float32)
        
        if invert:
            mask = 1.0 - mask
        
        return mask
    
    @staticmethod
    def _generate_color_range_mask(image: np.ndarray, params: dict) -> np.ndarray:
        """Generate a mask based on color range selection."""
        if len(image.shape) != 3:
            raise ValueError("Color range mask requires RGB image")
        
        target_hue = params['target_hue']
        hue_range = params['hue_range']
        sat_range = params['saturation_range']
        val_range = params['value_range']
        feather = params['feather']
        invert = params.get('invert', False)
        
        # Convert to HSV
        # Ensure image is in 0-255 range for cv2
        if image.max() <= 1.0:
            image_255 = (image * 255).astype(np.uint8)
        else:
            image_255 = image.astype(np.uint8)
        
        hsv = cv2.cvtColor(image_255, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Normalize HSV values
        hsv[:, :, 0] = hsv[:, :, 0] * 2  # OpenCV hue is 0-180, convert to 0-360
        hsv[:, :, 1] = hsv[:, :, 1] / 255.0  # Saturation to 0-1
        hsv[:, :, 2] = hsv[:, :, 2] / 255.0  # Value to 0-1
        
        # Calculate hue distance (circular)
        hue_diff = np.abs(hsv[:, :, 0] - target_hue)
        hue_diff = np.minimum(hue_diff, 360 - hue_diff)  # Wrap around
        
        # Create individual masks
        if feather > 0:
            # Smooth selection
            hue_mask = np.clip(1.0 - (hue_diff / hue_range), 0, 1)
            
            # Saturation mask
            sat_mask = np.ones_like(hsv[:, :, 1])
            if sat_range[0] > 0:
                sat_mask *= np.clip((hsv[:, :, 1] - sat_range[0]) / feather, 0, 1)
            if sat_range[1] < 1:
                sat_mask *= np.clip((sat_range[1] - hsv[:, :, 1]) / feather, 0, 1)
            
            # Value mask
            val_mask = np.ones_like(hsv[:, :, 2])
            if val_range[0] > 0:
                val_mask *= np.clip((hsv[:, :, 2] - val_range[0]) / feather, 0, 1)
            if val_range[1] < 1:
                val_mask *= np.clip((val_range[1] - hsv[:, :, 2]) / feather, 0, 1)
        else:
            # Hard selection
            hue_mask = (hue_diff <= hue_range).astype(np.float32)
            sat_mask = ((hsv[:, :, 1] >= sat_range[0]) & 
                       (hsv[:, :, 1] <= sat_range[1])).astype(np.float32)
            val_mask = ((hsv[:, :, 2] >= val_range[0]) & 
                       (hsv[:, :, 2] <= val_range[1])).astype(np.float32)
        
        # Combine masks
        mask = hue_mask * sat_mask * val_mask
        
        if invert:
            mask = 1.0 - mask
        
        return mask
    
    @staticmethod
    def refine_mask(mask: np.ndarray, blur_radius: float = 0.0, 
                   contrast: float = 1.0) -> np.ndarray:
        """
        Apply refinements to a generated mask.
        
        Args:
            mask: Input mask
            blur_radius: Gaussian blur radius in pixels
            contrast: Contrast adjustment (1.0 = no change)
            
        Returns:
            Refined mask
        """
        refined = mask.copy()
        
        # Apply blur if requested
        if blur_radius > 0:
            refined = gaussian_filter(refined, blur_radius)
        
        # Apply contrast adjustment
        if contrast != 1.0:
            # Adjust contrast around midpoint (0.5)
            refined = np.clip((refined - 0.5) * contrast + 0.5, 0, 1)
        
        return refined
    
    @staticmethod
    def combine_masks(masks: list, blend_modes: list, opacities: list) -> np.ndarray:
        """
        Combine multiple masks using various blend modes.
        
        Args:
            masks: List of masks to combine
            blend_modes: List of blend modes ('add', 'multiply', 'screen', 'overlay')
            opacities: List of opacity values for each mask
            
        Returns:
            Combined mask
        """
        if not masks:
            raise ValueError("No masks to combine")
        
        result = masks[0] * opacities[0]
        
        for i in range(1, len(masks)):
            mask = masks[i] * opacities[i]
            mode = blend_modes[i] if i < len(blend_modes) else 'normal'
            
            if mode == 'add':
                result = np.clip(result + mask, 0, 1)
            elif mode == 'multiply':
                result = result * mask
            elif mode == 'screen':
                result = 1.0 - (1.0 - result) * (1.0 - mask)
            elif mode == 'overlay':
                # Overlay blend mode
                low = 2 * result * mask
                high = 1.0 - 2 * (1.0 - result) * (1.0 - mask)
                result = np.where(result < 0.5, low, high)
            else:  # 'normal' or default
                result = mask
        
        return np.clip(result, 0, 1)