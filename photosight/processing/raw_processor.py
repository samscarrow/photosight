"""
Non-destructive RAW post-processing pipeline with iterative preview generation
"""

import numpy as np
import rawpy
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import logging
import json
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProcessingRecipe:
    """Stores all processing parameters for reproducible results"""
    
    # Geometry adjustments
    rotation_angle: float = 0.0
    crop_bounds: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    aspect_ratio: Optional[str] = None  # "3:2", "4:3", "16:9", "1:1", "4:5", or None
    
    # Exposure adjustments
    exposure_adjustment: float = 0.0  # EV adjustment
    highlights: float = 0.0  # -100 to +100
    shadows: float = 0.0  # -100 to +100
    whites: float = 0.0  # -100 to +100
    blacks: float = 0.0  # -100 to +100
    contrast: float = 0.0  # -100 to +100
    
    # Tone curve points
    tone_curve: Optional[List[Tuple[float, float]]] = None
    
    # White balance
    temperature_adjustment: float = 0.0  # Kelvin adjustment
    tint_adjustment: float = 0.0  # Green/Magenta adjustment
    auto_white_balance: bool = False
    
    # Color adjustments
    vibrance: float = 0.0  # 0 to 100
    saturation: float = 0.0  # -100 to +100
    
    # HSL adjustments per color range
    hsl_adjustments: Optional[Dict[str, Dict[str, float]]] = None
    
    # Clarity and texture
    clarity: float = 0.0  # -100 to +100
    texture: float = 0.0  # -100 to +100
    
    # Sharpening
    sharpening_amount: float = 50.0  # 0 to 150
    sharpening_radius: float = 1.0  # 0.5 to 3.0
    sharpening_detail: float = 25.0  # 0 to 100
    
    # Noise reduction
    noise_reduction_luminance: float = 0.0  # 0 to 100
    noise_reduction_color: float = 25.0  # 0 to 100
    
    # Vignetting
    vignette_amount: float = 0.0  # -100 to +100
    vignette_midpoint: float = 50.0  # 0 to 100
    
    # Metadata
    version: str = "1.0"
    created_at: str = ""
    name: str = ""
    
    def to_dict(self) -> Dict:
        """Convert recipe to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessingRecipe':
        """Create recipe from dictionary"""
        return cls(**data)
    
    def generate_hash(self) -> str:
        """Generate unique hash for this recipe"""
        # Exclude metadata fields from hash
        recipe_data = self.to_dict()
        for field in ['created_at', 'name']:
            recipe_data.pop(field, None)
        
        recipe_str = json.dumps(recipe_data, sort_keys=True)
        return hashlib.md5(recipe_str.encode()).hexdigest()[:8]


class RawPostProcessor:
    """
    Non-destructive RAW processor with iterative preview generation
    
    This processor:
    1. Never modifies the original RAW file
    2. Generates small preview JPEGs for iterative adjustments
    3. Applies full processing only when approved
    4. Saves processing recipes for reproducibility
    """
    
    def __init__(self, 
                 preview_size: int = 800,
                 preview_quality: int = 85,
                 cache_previews: bool = True):
        """
        Initialize the RAW post-processor
        
        Args:
            preview_size: Maximum dimension for preview images
            preview_quality: JPEG quality for previews (1-100)
            cache_previews: Whether to cache generated previews
        """
        self.preview_size = preview_size
        self.preview_quality = preview_quality
        self.cache_previews = cache_previews
        self.preview_cache = {}
        
        # Initialize sub-processors (lazy loading)
        self._horizon_detector = None
        self._smart_cropper = None
        self._exposure_optimizer = None
        self._color_enhancer = None
        
    def create_default_recipe(self, image_path: Path) -> ProcessingRecipe:
        """
        Create a default processing recipe for an image
        
        Args:
            image_path: Path to RAW image
            
        Returns:
            Default ProcessingRecipe
        """
        recipe = ProcessingRecipe(
            created_at=datetime.now().isoformat(),
            name=f"Recipe for {image_path.name}"
        )
        
        # Optionally analyze image and suggest initial adjustments
        # This will be implemented as we build the sub-modules
        
        return recipe
    
    def generate_preview(self, 
                        image_path: Path, 
                        recipe: ProcessingRecipe,
                        size: Optional[int] = None) -> np.ndarray:
        """
        Generate a preview JPEG with applied processing
        
        Args:
            image_path: Path to RAW image
            recipe: Processing recipe to apply
            size: Override preview size (None uses default)
            
        Returns:
            Preview image as RGB array
        """
        size = size or self.preview_size
        
        # Check cache if enabled
        cache_key = f"{image_path}_{recipe.generate_hash()}_{size}"
        if self.cache_previews and cache_key in self.preview_cache:
            logger.debug(f"Using cached preview for {image_path.name}")
            return self.preview_cache[cache_key]
        
        # Load RAW file
        with rawpy.imread(str(image_path)) as raw:
            # Use lower quality processing for preview
            rgb = raw.postprocess(
                use_camera_wb=not recipe.auto_white_balance,
                half_size=True,  # Faster processing
                no_auto_bright=True,
                output_bps=8
            )
        
        # Apply processing steps in order
        processed = self._apply_processing_pipeline(rgb, recipe, is_preview=True)
        
        # Resize for preview
        processed = self._resize_for_preview(processed, size)
        
        # Cache if enabled
        if self.cache_previews:
            self.preview_cache[cache_key] = processed
            
        return processed
    
    def export_full_size(self, 
                        image_path: Path, 
                        recipe: ProcessingRecipe,
                        output_path: Path,
                        output_format: str = 'jpeg',
                        quality: int = 95) -> Path:
        """
        Export full-size processed image
        
        Args:
            image_path: Path to RAW image
            recipe: Approved processing recipe
            output_path: Output file path
            output_format: Output format ('jpeg', 'tiff', 'png')
            quality: Quality for lossy formats (1-100)
            
        Returns:
            Path to exported image
        """
        logger.info(f"Exporting full-size image: {image_path.name}")
        
        # Load RAW file at full resolution
        with rawpy.imread(str(image_path)) as raw:
            # High quality processing
            rgb = raw.postprocess(
                use_camera_wb=not recipe.auto_white_balance,
                half_size=False,  # Full resolution
                no_auto_bright=True,
                output_bps=16,  # 16-bit processing
                use_auto_wb=recipe.auto_white_balance
            )
        
        # Apply full processing pipeline
        processed = self._apply_processing_pipeline(rgb, recipe, is_preview=False)
        
        # Convert to 8-bit if needed for JPEG
        if output_format == 'jpeg' and processed.dtype != np.uint8:
            processed = (np.clip(processed, 0, 65535) / 256).astype(np.uint8)
        
        # Save based on format
        if output_format == 'jpeg':
            cv2.imwrite(
                str(output_path), 
                cv2.cvtColor(processed, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
        elif output_format == 'tiff':
            cv2.imwrite(
                str(output_path),
                cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            )
        elif output_format == 'png':
            cv2.imwrite(
                str(output_path),
                cv2.cvtColor(processed, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 9]
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Save recipe alongside image
        recipe_path = output_path.with_suffix('.recipe.json')
        self.save_recipe(recipe, recipe_path)
        
        logger.info(f"Exported to: {output_path}")
        return output_path
    
    def _apply_processing_pipeline(self, 
                                  image: np.ndarray, 
                                  recipe: ProcessingRecipe,
                                  is_preview: bool = False) -> np.ndarray:
        """
        Apply all processing steps in the correct order
        
        Args:
            image: Input image array
            recipe: Processing recipe
            is_preview: Whether this is for preview (can skip some steps)
            
        Returns:
            Processed image array
        """
        result = image.copy()
        
        # 1. Geometry corrections (rotation and crop)
        if recipe.rotation_angle != 0:
            result = self._apply_rotation(result, recipe.rotation_angle)
        
        if recipe.crop_bounds:
            result = self._apply_crop(result, recipe.crop_bounds)
        elif recipe.aspect_ratio:
            result = self._apply_aspect_ratio_crop(result, recipe.aspect_ratio)
        
        # 2. Exposure adjustments
        if any([recipe.exposure_adjustment, recipe.highlights, recipe.shadows, 
                recipe.whites, recipe.blacks, recipe.contrast]):
            result = self._apply_exposure_adjustments(result, recipe)
        
        # 3. White balance
        if recipe.temperature_adjustment != 0 or recipe.tint_adjustment != 0:
            result = self._apply_white_balance(result, recipe)
        
        # 4. Color adjustments
        if recipe.vibrance != 0 or recipe.saturation != 0:
            result = self._apply_color_adjustments(result, recipe)
        
        # 5. Clarity and texture (skip for preview if expensive)
        if not is_preview or self.preview_size > 1200:
            if recipe.clarity != 0 or recipe.texture != 0:
                result = self._apply_clarity_texture(result, recipe)
        
        # 6. Sharpening (only for final export)
        if not is_preview and recipe.sharpening_amount > 0:
            result = self._apply_sharpening(result, recipe)
        
        # 7. Noise reduction (only if needed)
        if recipe.noise_reduction_luminance > 0:
            result = self._apply_noise_reduction(result, recipe)
        
        # 8. Vignetting
        if recipe.vignette_amount != 0:
            result = self._apply_vignette(result, recipe)
        
        return result
    
    def _apply_rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Apply rotation with edge filling"""
        if angle == 0:
            return image
            
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust matrix for new center
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2
        
        # Apply rotation with reflection padding
        rotated = cv2.warpAffine(
            image, matrix, (new_w, new_h),
            borderMode=cv2.BORDER_REFLECT
        )
        
        return rotated
    
    def _apply_crop(self, image: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply crop with bounds checking"""
        x, y, width, height = bounds
        h, w = image.shape[:2]
        
        # Ensure bounds are within image
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        width = min(width, w - x)
        height = min(height, h - y)
        
        return image[y:y+height, x:x+width]
    
    def _resize_for_preview(self, image: np.ndarray, max_size: int) -> np.ndarray:
        """Resize image for preview maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        if max(h, w) <= max_size:
            return image
        
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def save_recipe(self, recipe: ProcessingRecipe, path: Path) -> None:
        """Save processing recipe to JSON file"""
        with open(path, 'w') as f:
            json.dump(recipe.to_dict(), f, indent=2)
        logger.info(f"Saved recipe to: {path}")
    
    def load_recipe(self, path: Path) -> ProcessingRecipe:
        """Load processing recipe from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return ProcessingRecipe.from_dict(data)
    
    # Placeholder methods for processing steps
    # These will be implemented as we build the sub-modules
    
    def _apply_aspect_ratio_crop(self, image: np.ndarray, ratio: str) -> np.ndarray:
        """Apply aspect ratio crop (to be implemented)"""
        # This will use the smart cropper module
        return image
    
    def _apply_exposure_adjustments(self, image: np.ndarray, recipe: ProcessingRecipe) -> np.ndarray:
        """Apply exposure adjustments (to be implemented)"""
        # This will use the exposure optimization module
        return image
    
    def _apply_white_balance(self, image: np.ndarray, recipe: ProcessingRecipe) -> np.ndarray:
        """Apply white balance adjustments (to be implemented)"""
        # This will use the color enhancement module
        return image
    
    def _apply_color_adjustments(self, image: np.ndarray, recipe: ProcessingRecipe) -> np.ndarray:
        """Apply color adjustments (to be implemented)"""
        # This will use the color enhancement module
        return image
    
    def _apply_clarity_texture(self, image: np.ndarray, recipe: ProcessingRecipe) -> np.ndarray:
        """Apply clarity and texture (to be implemented)"""
        # This will use detail enhancement
        return image
    
    def _apply_sharpening(self, image: np.ndarray, recipe: ProcessingRecipe) -> np.ndarray:
        """Apply sharpening (to be implemented)"""
        # This will use the detail module
        return image
    
    def _apply_noise_reduction(self, image: np.ndarray, recipe: ProcessingRecipe) -> np.ndarray:
        """Apply noise reduction (to be implemented)"""
        # This will use the detail module
        return image
    
    def _apply_vignette(self, image: np.ndarray, recipe: ProcessingRecipe) -> np.ndarray:
        """Apply vignetting effect"""
        if recipe.vignette_amount == 0:
            return image
            
        h, w = image.shape[:2]
        
        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w / 2, h / 2
        
        # Calculate distance from center
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Normalize and adjust with midpoint
        vignette = dist_from_center / max_dist
        midpoint = recipe.vignette_midpoint / 100.0
        vignette = np.clip((vignette - midpoint) / (1 - midpoint), 0, 1)
        
        # Apply vignette
        amount = recipe.vignette_amount / 100.0
        if amount > 0:
            # Darken edges
            vignette = 1 - (vignette * amount)
        else:
            # Lighten edges
            vignette = 1 + (vignette * abs(amount) * 0.5)
        
        # Apply to all channels
        result = image.copy()
        for i in range(3):
            result[:, :, i] = np.clip(result[:, :, i] * vignette, 0, 255).astype(image.dtype)
        
        return result