"""
Non-destructive RAW processing module for PhotoSight

Implements recipe-based processing that preserves original RAW files
while allowing iterative adjustments and batch processing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, asdict
import numpy as np
import cv2
import rawpy
import exifread
from datetime import datetime
import hashlib

# Import PhotoSight processing modules
from ..scene_classifier import SceneClassifier
from .geometry.auto_straighten import AutoStraightener
from .geometry.smart_crop import SmartCropper, CropSuggestion
from .tone.exposure_optimizer import ExposureOptimizer, ExposureAnalysis
from .color.white_balance import WhiteBalanceCorrector, WhiteBalanceAnalysis, WhiteBalanceMethod
from .color.color_grading import ColorGrader, ColorGradingSettings, ColorGradingPreset

logger = logging.getLogger(__name__)


@dataclass
class ProcessingRecipe:
    """
    Non-destructive processing recipe for RAW images
    
    Stores all adjustments as parameters without modifying the original file.
    Can be serialized to JSON for storage and batch processing.
    """
    # File identification
    source_path: str
    file_hash: Optional[str] = None
    created_at: Optional[str] = None
    
    # Basic adjustments
    exposure_adjustment: float = 0.0  # EV adjustment (-2.0 to +2.0)
    shadows: float = 0.0  # Shadow lifting (0-100)
    highlights: float = 0.0  # Highlight recovery (-100 to 0)
    contrast: float = 0.0  # Contrast adjustment (-50 to +50)
    
    # White balance
    wb_method: Optional[str] = None  # Auto-selected or manual method
    wb_multipliers: Optional[Tuple[float, float, float]] = None
    temperature_adjustment: int = 0  # Kelvin adjustment (-2000 to +2000)
    tint_adjustment: int = 0  # Green/Magenta adjustment (-50 to +50)
    wb_analysis: Optional[Dict] = None
    
    # Color grading
    color_grading_preset: Optional[str] = None
    vibrance: float = 0.0  # Vibrance (0-100)
    saturation: float = 0.0  # Saturation (-100 to +100)
    color_grading_settings: Optional[Dict] = None
    
    # Tone curve (simplified)
    blacks: float = 0.0  # Black point adjustment (-100 to +100)
    whites: float = 0.0  # White point adjustment (-100 to +100)
    
    # Clarity and texture
    clarity: float = 0.0  # Local contrast (-100 to +100)
    texture: float = 0.0  # Fine detail enhancement (-100 to +100)
    
    # Lens corrections
    vignette_amount: float = 0.0  # Vignette correction (-100 to +100)
    
    # Geometry corrections
    rotation_angle: float = 0.0
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    crop_width: Optional[int] = None  
    crop_height: Optional[int] = None
    
    # Smart crop suggestions
    crop_suggestions: Optional[List[Dict]] = None
    selected_crop_index: Optional[int] = None
    
    # Exposure analysis results
    exposure_analysis: Optional[Dict] = None
    auto_exposure: bool = True
    
    # Scene-aware metadata
    scene_classification: Optional[Dict] = None
    processing_hints: Optional[Dict] = None
    
    # Processing metadata
    version: str = "1.0"
    notes: Optional[str] = None
    
    def to_json(self) -> str:
        """Serialize recipe to JSON"""
        data = asdict(self)
        data['created_at'] = data.get('created_at') or datetime.now().isoformat()
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ProcessingRecipe':
        """Deserialize recipe from JSON"""
        data = json.loads(json_str)
        return cls(**data)
    
    def save(self, path: Path) -> None:
        """Save recipe to file"""
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: Path) -> 'ProcessingRecipe':
        """Load recipe from file"""
        with open(path, 'r') as f:
            return cls.from_json(f.read())


class RawPostProcessor:
    """
    Non-destructive RAW post-processor
    
    Applies processing recipes to RAW files without modifying originals.
    Supports preview generation and full-resolution export.
    """
    
    def __init__(self, 
                 preview_size: int = 800,
                 preview_quality: int = 85,
                 cache_previews: bool = True,
                 auto_analyze: bool = True):
        """
        Initialize RAW post-processor
        
        Args:
            preview_size: Maximum dimension for preview images
            preview_quality: JPEG quality for previews (1-100)
            cache_previews: Whether to cache preview images
            auto_analyze: Whether to automatically analyze images for optimal settings
        """
        self.preview_size = preview_size
        self.preview_quality = preview_quality
        self.cache_previews = cache_previews
        self.auto_analyze = auto_analyze
        self._preview_cache: Dict[str, np.ndarray] = {}
        
        # Initialize processing modules
        logger.info("Initializing PhotoSight processing modules...")
        self.scene_classifier = SceneClassifier()
        self.auto_straightener = AutoStraightener()
        self.smart_cropper = SmartCropper()
        self.exposure_optimizer = ExposureOptimizer()
        self.wb_corrector = WhiteBalanceCorrector()
        self.color_grader = ColorGrader()
        logger.info("Processing modules initialized successfully")
    
    def create_default_recipe(self, raw_path: Path) -> ProcessingRecipe:
        """
        Create a default processing recipe for a RAW file
        
        Args:
            raw_path: Path to RAW file
            
        Returns:
            Default ProcessingRecipe instance
        """
        # Calculate file hash for verification
        file_hash = self._calculate_file_hash(raw_path)
        
        recipe = ProcessingRecipe(
            source_path=str(raw_path),
            file_hash=file_hash,
            created_at=datetime.now().isoformat()
        )
        
        # Auto-analyze if enabled
        if self.auto_analyze:
            logger.info(f"Auto-analyzing {raw_path.name}...")
            recipe = self.analyze_and_optimize(raw_path, recipe)
        
        return recipe
    
    def analyze_and_optimize(self, raw_path: Path, recipe: ProcessingRecipe) -> ProcessingRecipe:
        """
        Analyze image and populate recipe with optimal settings
        
        Args:
            raw_path: Path to RAW file
            recipe: Base recipe to populate
            
        Returns:
            Updated recipe with analysis results
        """
        logger.info(f"Running comprehensive analysis on {raw_path.name}")
        
        # Load preview for analysis
        with rawpy.imread(str(raw_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,  # Faster analysis
                no_auto_bright=False,
                output_bps=16
            )
        
        # Convert to float for processing
        rgb_float = rgb.astype(np.float32) / 65535.0
        
        # 1. Scene Classification
        logger.info("  - Classifying scene...")
        scene_result = self.scene_classifier.classify_scene(rgb_float)
        recipe.scene_classification = {
            'classification': scene_result.classification,
            'confidence': scene_result.confidence,
            'processing_hints': {
                'preferred_wb_temp_range': scene_result.processing_hints.get('preferred_wb_temp_range'),
                'suggested_leveling_methods': scene_result.processing_hints.get('suggested_leveling_methods', []),
                'color_grading_hints': scene_result.processing_hints.get('color_grading_hints', {})
            }
        }
        
        # 2. Auto-straightening
        logger.info("  - Detecting horizon/verticals...")
        straighten_result = self.auto_straightener.detect_and_straighten(
            rgb_float, 
            scene_result.processing_hints
        )
        if straighten_result.should_straighten and abs(straighten_result.rotation_angle) > 0.1:
            recipe.rotation_angle = straighten_result.rotation_angle
            logger.info(f"    Auto-straighten: {straighten_result.rotation_angle:.2f}°")
        
        # 3. Smart Cropping
        logger.info("  - Analyzing composition...")
        crop_suggestions = self.smart_cropper.suggest_crops(rgb_float)
        if crop_suggestions:
            recipe.crop_suggestions = [
                {
                    'bbox': suggestion.bbox,
                    'aspect_ratio': suggestion.aspect_ratio,
                    'confidence': suggestion.confidence,
                    'composition_score': suggestion.composition_score,
                    'subject_coverage': suggestion.subject_coverage
                }
                for suggestion in crop_suggestions[:5]  # Top 5 suggestions
            ]
            logger.info(f"    Found {len(crop_suggestions)} crop suggestions")
        
        # 4. Exposure Optimization
        logger.info("  - Optimizing exposure...")
        exposure_analysis = self.exposure_optimizer.analyze_exposure(rgb_float)
        recipe.exposure_analysis = {
            'zone_distribution': exposure_analysis.zone_distribution,
            'clipping_analysis': {
                'shadow_clipping': exposure_analysis.shadow_clipping,
                'highlight_clipping': exposure_analysis.highlight_clipping
            },
            'scene_type': exposure_analysis.scene_type,
            'dynamic_range_score': exposure_analysis.dynamic_range_score
        }
        
        # Apply suggested adjustments
        recipe.exposure_adjustment = exposure_analysis.exposure_compensation
        recipe.shadows = exposure_analysis.shadow_adjustment
        recipe.highlights = exposure_analysis.highlight_adjustment
        logger.info(f"    Exposure: {exposure_analysis.exposure_compensation:+.2f}EV, "
                   f"Shadows: {exposure_analysis.shadow_adjustment:+.0f}, "
                   f"Highlights: {exposure_analysis.highlight_adjustment:+.0f}")
        
        # 5. White Balance Analysis
        logger.info("  - Analyzing white balance...")
        wb_analysis = self.wb_corrector.analyze_white_balance(rgb_float)
        recipe.wb_analysis = {
            'estimated_temp_kelvin': wb_analysis.estimated_temp_kelvin,
            'method_used': wb_analysis.method_used.value,
            'confidence': wb_analysis.confidence,
            'illuminant_type': wb_analysis.illuminant_type,
            'color_cast': {
                'strength': wb_analysis.color_cast_strength,
                'dominant': wb_analysis.dominant_cast
            }
        }
        recipe.wb_multipliers = (
            wb_analysis.red_multiplier,
            wb_analysis.green_multiplier,
            wb_analysis.blue_multiplier
        )
        logger.info(f"    White balance: {wb_analysis.estimated_temp_kelvin:.0f}K "
                   f"({wb_analysis.method_used.value}, confidence: {wb_analysis.confidence:.2f})")
        
        # 6. Color Grading Suggestions
        logger.info("  - Suggesting color grading...")
        # Choose preset based on scene and mood
        if scene_result.classification == 'outdoor':
            if 'landscape' in scene_result.processing_hints.get('scene_features', []):
                suggested_preset = ColorGradingPreset.BRIGHT_AIRY
            else:
                suggested_preset = ColorGradingPreset.CINEMATIC
        else:  # indoor
            suggested_preset = ColorGradingPreset.VINTAGE
        
        recipe.color_grading_preset = suggested_preset.value
        logger.info(f"    Suggested preset: {suggested_preset.value}")
        
        logger.info(f"Analysis complete for {raw_path.name}")
        return recipe
    
    def generate_preview(self, raw_path: Path, recipe: ProcessingRecipe,
                        size: Optional[int] = None) -> np.ndarray:
        """
        Generate a preview image with processing applied
        
        Args:
            raw_path: Path to RAW file
            recipe: Processing recipe to apply
            size: Override preview size
            
        Returns:
            RGB numpy array of processed preview
        """
        size = size or self.preview_size
        
        # Check cache
        cache_key = f"{raw_path}_{recipe.file_hash}_{hash(recipe.to_json())}"
        if self.cache_previews and cache_key in self._preview_cache:
            return self._preview_cache[cache_key]
        
        # Load and process RAW
        with rawpy.imread(str(raw_path)) as raw:
            # Apply basic processing
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,  # Faster for preview
                no_auto_bright=False,
                output_bps=16  # 16-bit for better processing headroom
            )
        
        # Convert to float for processing
        rgb = rgb.astype(np.float32) / 65535.0
        
        # Apply comprehensive processing pipeline
        rgb = self._apply_processing_pipeline(rgb, recipe)
        
        # Apply rotation if needed
        if abs(recipe.rotation_angle) > 0.1:
            rgb = self._rotate_image(rgb, recipe.rotation_angle)
        
        # Apply crop if specified
        if recipe.crop_x is not None:
            rgb = rgb[
                recipe.crop_y:recipe.crop_y + recipe.crop_height,
                recipe.crop_x:recipe.crop_x + recipe.crop_width
            ]
        
        # Resize for preview
        h, w = rgb.shape[:2]
        if max(h, w) > size:
            scale = size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert back to 8-bit
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        # Cache if enabled
        if self.cache_previews:
            self._preview_cache[cache_key] = rgb
        
        return rgb
    
    def export_full_size(self, raw_path: Path, recipe: ProcessingRecipe,
                        output_path: Path, output_format: str = 'jpeg',
                        quality: int = 95) -> None:
        """
        Export full-resolution processed image
        
        Args:
            raw_path: Path to RAW file
            recipe: Processing recipe to apply
            output_path: Output file path
            output_format: Output format ('jpeg', 'png', 'tiff')
            quality: Output quality for JPEG (1-100)
        """
        logger.info(f"Exporting {raw_path.name} with recipe to {output_path}")
        
        # Load RAW at full resolution
        with rawpy.imread(str(raw_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=16
            )
        
        # Convert to float for processing
        rgb = rgb.astype(np.float32) / 65535.0
        
        # Apply comprehensive processing pipeline
        rgb = self._apply_processing_pipeline(rgb, recipe)
        
        # Apply rotation if needed
        if abs(recipe.rotation_angle) > 0.1:
            rgb = self._rotate_image(rgb, recipe.rotation_angle)
        
        # Apply crop if specified
        if recipe.crop_x is not None:
            rgb = rgb[
                recipe.crop_y:recipe.crop_y + recipe.crop_height,
                recipe.crop_x:recipe.crop_x + recipe.crop_width
            ]
        
        # Convert to appropriate format and save
        if output_format.lower() in ['jpeg', 'jpg']:
            # Convert to 8-bit BGR for OpenCV
            bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_format.lower() == 'png':
            # PNG supports 16-bit
            bgr = cv2.cvtColor((rgb * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr)
        elif output_format.lower() in ['tiff', 'tif']:
            # TIFF with 16-bit support
            bgr = cv2.cvtColor((rgb * 65535).astype(np.uint16), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Save recipe alongside
        recipe_path = output_path.with_suffix('.recipe.json')
        recipe.save(recipe_path)
        
        logger.info(f"Exported to {output_path} and saved recipe to {recipe_path}")
    
    def _apply_processing_pipeline(self, rgb: np.ndarray, recipe: ProcessingRecipe) -> np.ndarray:
        """
        Apply comprehensive processing pipeline to image
        
        This method orchestrates the complete PhotoSight processing pipeline:
        1. White Balance Correction
        2. Exposure Optimization  
        3. Color Grading
        4. Tone Adjustments
        5. Local Adjustments
        
        Args:
            rgb: Float32 RGB image (0-1 range)
            recipe: Processing recipe with all parameters
            
        Returns:
            Processed RGB image
        """
        logger.debug("Applying comprehensive processing pipeline...")
        
        # 1. WHITE BALANCE CORRECTION
        if recipe.wb_multipliers:
            logger.debug("  - Applying white balance...")
            rgb[:, :, 0] *= recipe.wb_multipliers[0]  # Red
            rgb[:, :, 1] *= recipe.wb_multipliers[1]  # Green  
            rgb[:, :, 2] *= recipe.wb_multipliers[2]  # Blue
            
            # Apply additional temperature/tint adjustments
            if recipe.temperature_adjustment != 0 or recipe.tint_adjustment != 0:
                # Create analysis object for manual adjustments
                wb_analysis = WhiteBalanceAnalysis(
                    estimated_temp_kelvin=recipe.wb_analysis.get('estimated_temp_kelvin', 5500) 
                                        if recipe.wb_analysis else 5500,
                    confidence=1.0,
                    red_multiplier=recipe.wb_multipliers[0],
                    green_multiplier=recipe.wb_multipliers[1], 
                    blue_multiplier=recipe.wb_multipliers[2],
                    color_cast_strength=0.0,
                    dominant_cast="none",
                    method_used=WhiteBalanceMethod.MANUAL,
                    illuminant_type="unknown",
                    temperature_adjustment=0,
                    tint_adjustment=0,
                    gray_deviation=0.0,
                    highlight_preservation=1.0
                )
                
                rgb = self.wb_corrector.correct_white_balance(
                    (rgb * 255).astype(np.uint8),
                    wb_analysis,
                    recipe.temperature_adjustment,
                    recipe.tint_adjustment
                ).astype(np.float32) / 255.0
        
        # 2. EXPOSURE OPTIMIZATION
        logger.debug("  - Applying exposure adjustments...")
        
        # Basic exposure compensation
        if recipe.exposure_adjustment != 0:
            rgb *= 2 ** recipe.exposure_adjustment
        
        # Advanced shadow/highlight recovery
        if recipe.shadows != 0 or recipe.highlights != 0:
            rgb = self._apply_advanced_shadow_highlight(rgb, recipe.shadows, recipe.highlights)
        
        # 3. COLOR GRADING
        if recipe.color_grading_preset or recipe.color_grading_settings:
            logger.debug("  - Applying color grading...")
            
            # Create color grading settings
            if recipe.color_grading_settings:
                # Use custom settings from recipe
                settings_dict = recipe.color_grading_settings
                settings = ColorGradingSettings(**settings_dict)
            else:
                # Use preset
                preset = ColorGradingPreset(recipe.color_grading_preset)
                settings = self.color_grader.get_preset(preset)
                
                # Apply any manual vibrance/saturation overrides
                if recipe.vibrance != 0:
                    settings.vibrance = recipe.vibrance
                if recipe.saturation != 0:
                    settings.saturation = recipe.saturation
            
            # Apply color grading
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            rgb_graded = self.color_grader.apply_color_grading(rgb_uint8, settings)
            rgb = rgb_graded.astype(np.float32) / 255.0
        else:
            # Apply basic vibrance/saturation if no color grading
            if recipe.vibrance != 0:
                rgb = self._adjust_vibrance(rgb, recipe.vibrance)
            if recipe.saturation != 0:
                rgb = self._adjust_saturation(rgb, recipe.saturation)
        
        # 4. TONE ADJUSTMENTS
        logger.debug("  - Applying tone adjustments...")
        
        # Contrast
        if recipe.contrast != 0:
            rgb = self._adjust_contrast(rgb, recipe.contrast)
        
        # Blacks and whites
        if recipe.blacks != 0 or recipe.whites != 0:
            rgb = self._adjust_blacks_whites(rgb, recipe.blacks, recipe.whites)
        
        # 5. LOCAL ADJUSTMENTS
        logger.debug("  - Applying local adjustments...")
        
        # Clarity (local contrast)
        if recipe.clarity != 0:
            rgb = self._adjust_clarity(rgb, recipe.clarity)
        
        # Texture enhancement
        if recipe.texture != 0:
            rgb = self._adjust_texture(rgb, recipe.texture)
        
        # Vignette correction/application
        if recipe.vignette_amount != 0:
            rgb = self._apply_vignette(rgb, recipe.vignette_amount)
        
        # Final clipping to ensure valid range
        rgb = np.clip(rgb, 0, 1)
        
        logger.debug("Processing pipeline complete")
        return rgb
    
    def _apply_advanced_shadow_highlight(self, img: np.ndarray, shadows: float, 
                                        highlights: float) -> np.ndarray:
        """Advanced shadow/highlight recovery using luminosity masks"""
        if shadows == 0 and highlights == 0:
            return img
            
        # Create luminosity masks for targeted adjustments
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        
        # Shadow mask (targets dark areas)
        if shadows > 0:
            shadow_mask = np.maximum(0, 1 - luminance * 2)
            shadow_mask = shadow_mask ** 2  # Smooth falloff
            shadow_mask = shadow_mask[:, :, np.newaxis]
            
            # Lift shadows with natural falloff
            shadow_strength = shadows / 100.0 * 0.5  # Scale down for natural look
            img = img + img * shadow_mask * shadow_strength
        
        # Highlight mask (targets bright areas)
        if highlights < 0:
            highlight_mask = np.maximum(0, luminance * 2 - 1)
            highlight_mask = highlight_mask ** 2  # Smooth falloff
            highlight_mask = highlight_mask[:, :, np.newaxis]
            
            # Recover highlights
            highlight_strength = abs(highlights) / 100.0 * 0.5
            img = img - img * highlight_mask * highlight_strength
        
        return np.clip(img, 0, 1)
    
    def _adjust_blacks_whites(self, img: np.ndarray, blacks: float, whites: float) -> np.ndarray:
        """Adjust black and white points"""
        if blacks == 0 and whites == 0:
            return img
            
        # Create tone curve adjustment
        # Blacks adjustment affects the lower part of the curve
        if blacks != 0:
            black_factor = blacks / 100.0 * 0.3  # Scale for natural adjustment
            img = img + black_factor * (1 - img)  # Lift or lower blacks
        
        # Whites adjustment affects the upper part of the curve  
        if whites != 0:
            white_factor = whites / 100.0 * 0.3
            img = img * (1 + white_factor)  # Boost or reduce whites
            
        return np.clip(img, 0, 1)
    
    def _adjust_texture(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust fine texture details"""
        if amount == 0:
            return img
            
        # Create high-frequency enhancement
        # Use smaller kernel for fine details vs clarity
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=0.5)
        high_freq = img - blurred
        
        # Apply texture enhancement
        factor = amount / 100.0 * 0.3  # Subtle effect
        result = img + high_freq * factor
        
        return np.clip(result, 0, 1)
    
    def _adjust_contrast(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust contrast"""
        # S-curve contrast adjustment
        factor = (amount + 100) / 100
        img = (img - 0.5) * factor + 0.5
        return np.clip(img, 0, 1)
    
    def _adjust_vibrance(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust vibrance (smart saturation)"""
        # Convert to HSV
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1 + (amount / 100)  # Adjust saturation channel
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to RGB
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255
    
    def _adjust_saturation(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust saturation"""
        # Simple saturation adjustment
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        factor = (amount + 100) / 100
        img = gray + (img - gray) * factor
        return np.clip(img, 0, 1)
    
    def _adjust_clarity(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust clarity (local contrast)"""
        if amount == 0:
            return img
        
        # Create blurred version
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        
        # High-pass filter
        high_pass = img - blurred
        
        # Add back with strength
        factor = amount / 100
        result = img + high_pass * factor
        
        return np.clip(result, 0, 1)
    
    def _apply_vignette(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Apply or correct vignetting"""
        h, w = img.shape[:2]
        
        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Normalize and create vignette mask
        vignette = 1 - (dist_from_center / max_dist) ** 2
        vignette = vignette[:, :, np.newaxis]
        
        # Apply vignette
        factor = abs(amount) / 100
        if amount < 0:
            # Darken edges
            img = img * (1 - factor * (1 - vignette))
        else:
            # Brighten edges (vignette correction)
            img = img / (1 - factor * (1 - vignette))
        
        return np.clip(img, 0, 1)
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for new dimensions
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        # Rotate
        rotated = cv2.warpAffine(img, M, (new_w, new_h), 
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]  # Use first 16 chars
    
    def clear_preview_cache(self) -> None:
        """Clear preview cache"""
        self._preview_cache.clear()
        logger.info("Preview cache cleared")