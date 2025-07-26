"""
Preview Processor with adaptive quality modes.

Provides fast preview processing by using simplified algorithms and
reduced quality settings for real-time responsiveness.
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2
from pathlib import Path

from .models import ProcessingMode, ProxyLevel, PreviewConfig
from ..processing.raw_processor import ProcessingRecipe, RawPostProcessor
from ..processing.noise import NoiseReductionSettings, EdgePreservationMethod

logger = logging.getLogger(__name__)


class PreviewProcessor:
    """
    Adaptive processor that provides different quality modes for preview vs export.
    
    Preview mode: Fast algorithms, reduced quality, optimized for responsiveness
    Export mode: Full quality algorithms, all features enabled
    """
    
    def __init__(self, config: PreviewConfig):
        self.config = config
        self.raw_processor = RawPostProcessor()
        
        # Cache for expensive operations
        self._wb_cache = {}
        self._exposure_cache = {}
        
        logger.info("PreviewProcessor initialized")
    
    def process_for_preview(self, image: np.ndarray, recipe: ProcessingRecipe,
                          mode: ProcessingMode = ProcessingMode.PREVIEW,
                          target_level: ProxyLevel = ProxyLevel.MEDIUM) -> np.ndarray:
        """
        Process image with mode-appropriate quality settings.
        
        Args:
            image: Input image (float32, 0-1 range, RGB)
            recipe: Processing recipe
            mode: Processing quality mode
            target_level: Target proxy level (affects quality choices)
            
        Returns:
            Processed image
        """
        start_time = time.time()
        
        try:
            # Create optimized recipe for the mode
            optimized_recipe = self._optimize_recipe_for_mode(recipe, mode, target_level)
            
            # Process with optimized settings
            result = self._apply_preview_pipeline(image, optimized_recipe, mode)
            
            duration = time.time() - start_time
            logger.debug(f"Preview processing ({mode.value}) completed in {duration:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Preview processing failed: {e}")
            # Return original image as fallback
            return image.copy()
    
    def process_tiles(self, image: np.ndarray, recipe: ProcessingRecipe,
                     tile_size: int = 512, overlap: int = 32,
                     mode: ProcessingMode = ProcessingMode.PREVIEW) -> np.ndarray:
        """
        Process large image in tiles for memory efficiency.
        
        Args:
            image: Input image
            recipe: Processing recipe
            tile_size: Size of processing tiles
            overlap: Overlap between tiles for seamless blending
            mode: Processing quality mode
            
        Returns:
            Processed image
        """
        height, width = image.shape[:2]
        
        # If image is small enough, process normally
        if max(height, width) <= tile_size:
            return self.process_for_preview(image, recipe, mode)
        
        logger.debug(f"Processing {width}x{height} image in tiles (size: {tile_size}, overlap: {overlap})")
        
        result = np.zeros_like(image)
        blend_mask = np.zeros((height, width), dtype=np.float32)
        
        # Calculate tile positions
        tiles = self._calculate_tile_positions(width, height, tile_size, overlap)
        
        for i, (x, y, w, h, ox, oy) in enumerate(tiles):
            try:
                # Extract tile with overlap
                tile = image[y-oy:y+h+oy, x-ox:x+w+ox]
                
                # Process tile
                processed_tile = self.process_for_preview(tile, recipe, mode)
                
                # Create blend weights (fade out at edges)
                tile_mask = np.ones((h, w), dtype=np.float32)
                if ox > 0:
                    fade_x = np.linspace(0, 1, ox)
                    tile_mask[:, :ox] *= fade_x[np.newaxis, :]
                    tile_mask[:, -ox:] *= fade_x[np.newaxis, ::-1]
                if oy > 0:
                    fade_y = np.linspace(0, 1, oy)
                    tile_mask[:oy, :] *= fade_y[:, np.newaxis]
                    tile_mask[-oy:, :] *= fade_y[::-1, np.newaxis]
                
                # Extract core area (without overlap)
                core_tile = processed_tile[oy:oy+h, ox:ox+w]
                
                # Blend into result
                result[y:y+h, x:x+w] += core_tile * tile_mask[:, :, np.newaxis]
                blend_mask[y:y+h, x:x+w] += tile_mask
                
            except Exception as e:
                logger.error(f"Failed to process tile {i}: {e}")
        
        # Normalize by blend weights
        valid_mask = blend_mask > 0
        result[valid_mask] /= blend_mask[valid_mask, np.newaxis]
        
        return result
    
    def get_processing_estimate(self, image_shape: Tuple[int, int], 
                               recipe: ProcessingRecipe,
                               mode: ProcessingMode) -> float:
        """
        Estimate processing time for given parameters.
        
        Args:
            image_shape: (height, width) of image
            recipe: Processing recipe
            mode: Processing mode
            
        Returns:
            Estimated processing time in seconds
        """
        height, width = image_shape
        pixel_count = height * width
        
        # Base processing time per megapixel
        base_times = {
            ProcessingMode.PREVIEW: 0.1,   # 100ms per MP
            ProcessingMode.BALANCED: 0.3,  # 300ms per MP
            ProcessingMode.EXPORT: 1.0     # 1s per MP
        }
        
        base_time = base_times[mode] * (pixel_count / 1_000_000)
        
        # Adjust for recipe complexity
        complexity_factor = 1.0
        
        if recipe.adjustment_layers:
            complexity_factor += len(recipe.adjustment_layers) * 0.2
        
        if recipe.noise_reduction and recipe.noise_reduction.quality_mode == "high":
            complexity_factor += 0.5
        
        if recipe.color_grading_preset or recipe.color_grading_settings:
            complexity_factor += 0.3
        
        return base_time * complexity_factor
    
    def _optimize_recipe_for_mode(self, recipe: ProcessingRecipe, 
                                 mode: ProcessingMode,
                                 target_level: ProxyLevel) -> ProcessingRecipe:
        """Create optimized recipe for the given processing mode."""
        # Create a copy to avoid modifying original
        optimized = ProcessingRecipe(
            source_path=recipe.source_path,
            file_hash=recipe.file_hash,
            created_at=recipe.created_at,
            
            # Copy basic adjustments
            exposure_adjustment=recipe.exposure_adjustment,
            shadows=recipe.shadows,
            highlights=recipe.highlights,
            contrast=recipe.contrast,
            
            # Copy white balance
            wb_method=recipe.wb_method,
            wb_multipliers=recipe.wb_multipliers,
            temperature_adjustment=recipe.temperature_adjustment,
            tint_adjustment=recipe.tint_adjustment,
            wb_analysis=recipe.wb_analysis,
            
            # Copy color grading (might simplify for preview)
            color_grading_preset=recipe.color_grading_preset,
            vibrance=recipe.vibrance,
            saturation=recipe.saturation,
            
            # Copy tone adjustments
            blacks=recipe.blacks,
            whites=recipe.whites,
            clarity=recipe.clarity,
            texture=recipe.texture,
            vignette_amount=recipe.vignette_amount,
            
            # Copy geometry
            rotation_angle=recipe.rotation_angle,
            crop_x=recipe.crop_x,
            crop_y=recipe.crop_y,
            crop_width=recipe.crop_width,
            crop_height=recipe.crop_height,
            
            # Copy metadata
            exif_data=recipe.exif_data,
            version=recipe.version,
            notes=recipe.notes
        )
        
        # Optimize based on mode
        if mode == ProcessingMode.PREVIEW:
            optimized = self._optimize_for_preview(optimized, target_level)
        elif mode == ProcessingMode.BALANCED:
            optimized = self._optimize_for_balanced(optimized, target_level)
        else:  # EXPORT mode
            # Use full quality settings
            optimized.color_grading_settings = recipe.color_grading_settings
            optimized.adjustment_layers = recipe.adjustment_layers.copy()
            optimized.noise_reduction = recipe.noise_reduction
        
        return optimized
    
    def _optimize_for_preview(self, recipe: ProcessingRecipe,
                             target_level: ProxyLevel) -> ProcessingRecipe:
        """Optimize recipe for fast preview processing."""
        
        # Simplify color grading for preview
        if recipe.color_grading_settings:
            # Use preset instead of custom settings for speed
            recipe.color_grading_preset = "neutral"
            recipe.color_grading_settings = None
        
        # Limit adjustment layers for preview
        if recipe.adjustment_layers:
            # Only use first 3 layers for preview
            recipe.adjustment_layers = recipe.adjustment_layers[:3]
            
            # Simplify layer settings
            for layer in recipe.adjustment_layers:
                # Reduce opacity for subtler effects
                layer.opacity *= 0.7
                # Use simpler blend modes
                if layer.blend_mode not in ['normal', 'multiply', 'screen']:
                    layer.blend_mode = 'normal'
        
        # Optimize noise reduction for preview
        if recipe.noise_reduction:
            preview_nr = NoiseReductionSettings(
                luminance_amount=recipe.noise_reduction.luminance_amount * 0.8,
                chrominance_amount=recipe.noise_reduction.chrominance_amount * 0.8,
                detail_preservation=max(recipe.noise_reduction.detail_preservation, 70.0),
                edge_preservation_method=EdgePreservationMethod.BILATERAL,  # Fastest
                quality_mode="fast",
                iso_adaptive=recipe.noise_reduction.iso_adaptive,
                hot_pixel_removal=False,  # Skip for preview
                pattern_suppression=False,  # Skip for preview
                multi_scale=False  # Single scale for speed
            )
            recipe.noise_reduction = preview_nr
        
        # Reduce some effect strengths for preview
        recipe.clarity = recipe.clarity * 0.8 if recipe.clarity != 0 else 0
        recipe.texture = recipe.texture * 0.8 if recipe.texture != 0 else 0
        
        return recipe
    
    def _optimize_for_balanced(self, recipe: ProcessingRecipe,
                              target_level: ProxyLevel) -> ProcessingRecipe:
        """Optimize recipe for balanced quality/speed."""
        
        # Keep most settings but optimize expensive operations
        if recipe.adjustment_layers:
            # Keep all layers but reduce some complexity
            for layer in recipe.adjustment_layers:
                # Slightly reduce expensive adjustments
                if 'clarity' in layer.adjustments:
                    layer.adjustments['clarity'] *= 0.9
                if 'texture' in layer.adjustments:
                    layer.adjustments['texture'] *= 0.9
        
        # Optimize noise reduction for balanced mode
        if recipe.noise_reduction:
            balanced_nr = NoiseReductionSettings(
                luminance_amount=recipe.noise_reduction.luminance_amount,
                chrominance_amount=recipe.noise_reduction.chrominance_amount,
                detail_preservation=recipe.noise_reduction.detail_preservation,
                edge_preservation_method=EdgePreservationMethod.GUIDED,  # Good balance
                quality_mode="balanced",
                iso_adaptive=recipe.noise_reduction.iso_adaptive,
                hot_pixel_removal=recipe.noise_reduction.hot_pixel_removal,
                pattern_suppression=recipe.noise_reduction.pattern_suppression,
                multi_scale=target_level in [ProxyLevel.LARGE, ProxyLevel.FULL]
            )
            recipe.noise_reduction = balanced_nr
        
        return recipe
    
    def _apply_preview_pipeline(self, image: np.ndarray, recipe: ProcessingRecipe,
                               mode: ProcessingMode) -> np.ndarray:
        """Apply the processing pipeline optimized for preview."""
        
        # Use the existing raw processor pipeline
        # The optimized recipe will control the quality/speed tradeoffs
        return self.raw_processor._apply_processing_pipeline(image, recipe)
    
    def _calculate_tile_positions(self, width: int, height: int, 
                                 tile_size: int, overlap: int) -> list:
        """Calculate optimal tile positions for processing."""
        tiles = []
        
        y = 0
        while y < height:
            x = 0
            while x < width:
                # Calculate tile dimensions
                w = min(tile_size, width - x)
                h = min(tile_size, height - y)
                
                # Calculate overlap
                ox = overlap if x > 0 else 0
                oy = overlap if y > 0 else 0
                
                # Ensure we don't go beyond image bounds
                if x + w + overlap > width:
                    ox = min(ox, width - x - w)
                if y + h + overlap > height:
                    oy = min(oy, height - y - h)
                
                tiles.append((x, y, w, h, ox, oy))
                
                x += tile_size - overlap
                if x >= width:
                    break
            
            y += tile_size - overlap
            if y >= height:
                break
        
        return tiles
    
    def clear_caches(self):
        """Clear internal processing caches."""
        self._wb_cache.clear()
        self._exposure_cache.clear()
        logger.debug("Preview processor caches cleared")