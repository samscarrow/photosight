"""
Comprehensive tests for the Local Adjustments framework.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from photosight.processing.local_adjustments import (
    AdjustmentLayer, MaskData, MaskType, MaskGenerator, 
    LocalAdjustmentProcessor, LocalAdjustmentConfig
)


class TestMaskGeneration:
    """Test mask generation strategies."""
    
    def test_radial_mask_generation(self):
        """Test radial gradient mask generation."""
        # Create test parameters
        shape = (100, 100)
        mask_data = MaskData.radial(
            center=(0.5, 0.5),  # Center of image
            radius=(0.3, 0.3),  # 30% of image size
            feather=0.5,
            invert=False
        )
        
        # Generate mask
        mask = MaskGenerator.generate_mask(shape, mask_data)
        
        # Verify mask properties
        assert mask.shape == shape
        assert mask.dtype == np.float32
        assert 0 <= mask.min() <= mask.max() <= 1
        
        # Center should be bright
        center_value = mask[50, 50]
        assert center_value > 0.8
        
        # Corners should be dark
        corner_value = mask[0, 0]
        assert corner_value < 0.2
    
    def test_radial_mask_elliptical(self):
        """Test elliptical radial mask."""
        shape = (100, 200)  # Wide image
        mask_data = MaskData.radial(
            center=(0.5, 0.5),
            radius=(0.2, 0.4),  # Wider horizontally
            feather=0.3,
            invert=False
        )
        
        mask = MaskGenerator.generate_mask(shape, mask_data)
        
        # Check elliptical shape
        center_y, center_x = 50, 100
        # Horizontal extent should be larger
        assert mask[center_y, center_x + 60] > mask[center_y + 30, center_x]
    
    def test_radial_mask_inverted(self):
        """Test inverted radial mask."""
        shape = (100, 100)
        mask_data = MaskData.radial(
            center=(0.5, 0.5),
            radius=(0.3, 0.3),
            feather=0.5,
            invert=True
        )
        
        mask = MaskGenerator.generate_mask(shape, mask_data)
        
        # Center should be dark (inverted)
        assert mask[50, 50] < 0.2
        # Corners should be bright
        assert mask[0, 0] > 0.8
    
    def test_linear_mask_generation(self):
        """Test linear gradient mask generation."""
        shape = (100, 100)
        mask_data = MaskData.linear(
            start=(0, 0.5),    # Left middle
            end=(1, 0.5),      # Right middle
            feather=0.5,
            invert=False
        )
        
        mask = MaskGenerator.generate_mask(shape, mask_data)
        
        # Left side should be dark
        assert mask[50, 10] < 0.3
        # Right side should be bright
        assert mask[50, 90] > 0.7
        # Should be horizontal gradient
        assert abs(mask[25, 50] - mask[75, 50]) < 0.1
    
    def test_linear_mask_diagonal(self):
        """Test diagonal linear gradient."""
        shape = (100, 100)
        mask_data = MaskData.linear(
            start=(0, 0),      # Top-left
            end=(1, 1),        # Bottom-right
            feather=0.3,
            invert=False
        )
        
        mask = MaskGenerator.generate_mask(shape, mask_data)
        
        # Top-left should be darker than bottom-right
        assert mask[10, 10] < mask[90, 90]
        # Should have diagonal gradient
        assert mask[30, 30] < mask[70, 70]
    
    def test_luminosity_mask_generation(self):
        """Test luminosity-based mask generation."""
        # Create test image with gradient
        image = np.zeros((100, 100, 3), dtype=np.float32)
        for i in range(100):
            image[i, :, :] = i / 100.0  # Vertical gradient
        
        # Select mid-tones
        mask_data = MaskData.luminosity(
            min_lum=0.3,
            max_lum=0.7,
            feather=0.1,
            invert=False
        )
        
        mask = MaskGenerator.generate_mask(image.shape[:2], mask_data, image)
        
        # Top (dark) should be excluded
        assert mask[10, 50] < 0.2
        # Middle (mid-tones) should be selected
        assert mask[50, 50] > 0.8
        # Bottom (bright) should be excluded
        assert mask[90, 50] < 0.2
    
    def test_color_range_mask_generation(self):
        """Test color range mask generation."""
        # Create test image with color patches
        image = np.zeros((100, 100, 3), dtype=np.float32)
        # Red patch
        image[20:40, 20:40] = [1, 0, 0]
        # Green patch
        image[20:40, 60:80] = [0, 1, 0]
        # Blue patch
        image[60:80, 40:60] = [0, 0, 1]
        
        # Select red colors
        mask_data = MaskData.color_range(
            target_hue=0,      # Red
            hue_range=30,
            saturation_range=(0.5, 1.0),
            value_range=(0.5, 1.0),
            feather=0.1,
            invert=False
        )
        
        mask = MaskGenerator.generate_mask(image.shape[:2], mask_data, image)
        
        # Red patch should be selected
        assert mask[30, 30] > 0.8
        # Green patch should not be selected
        assert mask[30, 70] < 0.2
        # Blue patch should not be selected
        assert mask[70, 50] < 0.2


class TestAdjustmentLayerModel:
    """Test AdjustmentLayer data model."""
    
    def test_adjustment_layer_creation(self):
        """Test creating adjustment layer."""
        layer = AdjustmentLayer(
            name="Test Layer",
            enabled=True,
            mask_data=MaskData.radial((0.5, 0.5), (0.3, 0.3)),
            adjustments={
                'exposure': 0.5,
                'contrast': 20,
                'saturation': -10
            }
        )
        
        assert layer.name == "Test Layer"
        assert layer.enabled
        assert layer.has_adjustments()
        assert layer.get_adjustment_value('exposure') == 0.5
        assert layer.get_adjustment_value('nonexistent', -1) == -1
    
    def test_adjustment_validation(self):
        """Test adjustment value validation."""
        # Valid adjustments
        layer = AdjustmentLayer(
            name="Valid",
            adjustments={
                'exposure': 2.0,      # Within -5 to 5
                'contrast': 50,       # Within -100 to 100
                'saturation': -50     # Within -100 to 100
            }
        )
        assert layer.has_adjustments()
        
        # Invalid adjustment should raise error
        with pytest.raises(ValueError):
            AdjustmentLayer(
                name="Invalid",
                adjustments={
                    'exposure': 10.0  # Out of range
                }
            )


class TestLocalAdjustmentProcessor:
    """Test the local adjustment processor."""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        # Create gradient image for testing
        image = np.zeros((200, 200, 3), dtype=np.float32)
        for i in range(200):
            image[i, :, :] = i / 200.0
        return image
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        config = LocalAdjustmentConfig(
            mask_resolution_factor=0.5,
            cache_masks=True
        )
        processor = LocalAdjustmentProcessor(config)
        
        assert processor.config.mask_resolution_factor == 0.5
        assert processor.mask_cache is not None
    
    def test_single_layer_adjustment(self, test_image):
        """Test applying a single adjustment layer."""
        processor = LocalAdjustmentProcessor()
        
        # Create adjustment layer with radial mask
        layer = AdjustmentLayer(
            name="Brighten Center",
            mask_data=MaskData.radial((0.5, 0.5), (0.4, 0.4), feather=0.5),
            adjustments={'exposure': 1.0}  # +1 stop
        )
        
        # Apply adjustment
        result = processor.apply_adjustment_layers(test_image, [layer])
        
        # Center should be brighter
        center_original = test_image[100, 100, 0]
        center_adjusted = result[100, 100, 0]
        assert center_adjusted > center_original * 1.5  # Roughly 2x brighter
        
        # Corners should be less affected
        corner_original = test_image[10, 10, 0]
        corner_adjusted = result[10, 10, 0]
        assert abs(corner_adjusted - corner_original) < 0.1
    
    def test_multiple_layers(self, test_image):
        """Test applying multiple adjustment layers."""
        processor = LocalAdjustmentProcessor()
        
        layers = [
            # Darken top half
            AdjustmentLayer(
                name="Darken Sky",
                mask_data=MaskData.linear((0, 0), (0, 0.5), feather=0.2),
                adjustments={'exposure': -0.5}
            ),
            # Add contrast to center
            AdjustmentLayer(
                name="Center Contrast",
                mask_data=MaskData.radial((0.5, 0.5), (0.3, 0.3)),
                adjustments={'contrast': 50}
            )
        ]
        
        result = processor.apply_adjustment_layers(test_image, layers)
        
        # Top should be darker
        assert result[20, 100, 0] < test_image[20, 100, 0]
        # Center should have more contrast
        center_region = result[90:110, 90:110, 0]
        assert center_region.std() > test_image[90:110, 90:110, 0].std()
    
    def test_disabled_layer(self, test_image):
        """Test that disabled layers are skipped."""
        processor = LocalAdjustmentProcessor()
        
        layer = AdjustmentLayer(
            name="Disabled",
            enabled=False,
            adjustments={'exposure': 5.0}  # Extreme adjustment
        )
        
        result = processor.apply_adjustment_layers(test_image, [layer])
        
        # Should be unchanged
        np.testing.assert_array_almost_equal(result, test_image)
    
    def test_color_adjustments(self, test_image):
        """Test color-specific adjustments."""
        processor = LocalAdjustmentProcessor()
        
        # Make test image colored
        colored_image = test_image.copy()
        colored_image[:, :, 0] *= 0.8  # Less red
        colored_image[:, :, 2] *= 1.2  # More blue
        
        layer = AdjustmentLayer(
            name="Warm Center",
            mask_data=MaskData.radial((0.5, 0.5), (0.4, 0.4)),
            adjustments={
                'temperature': 50,   # Warmer
                'vibrance': 30,
                'saturation': 20
            }
        )
        
        result = processor.apply_adjustment_layers(colored_image, [layer])
        
        # Center should be warmer (more red, less blue)
        center_y, center_x = 100, 100
        assert result[center_y, center_x, 0] > colored_image[center_y, center_x, 0]
        assert result[center_y, center_x, 2] < colored_image[center_y, center_x, 2]
    
    def test_mask_refinement(self, test_image):
        """Test mask blur and contrast refinement."""
        processor = LocalAdjustmentProcessor()
        
        layer = AdjustmentLayer(
            name="Soft Edge",
            mask_data=MaskData.radial((0.5, 0.5), (0.3, 0.3), feather=0.1),
            adjustments={'exposure': 1.0},
            mask_blur=5.0,        # Blur the mask
            mask_contrast=1.5     # Increase contrast
        )
        
        result = processor.apply_adjustment_layers(test_image, [layer])
        
        # Transition should be smooth due to blur
        # Check gradient along radius
        center_line = result[100, :, 0]
        differences = np.diff(center_line)
        # No sharp transitions
        assert np.max(np.abs(differences)) < 0.1
    
    def test_blend_modes(self, test_image):
        """Test different blend modes."""
        processor = LocalAdjustmentProcessor()
        
        # Create high contrast adjustment
        layer = AdjustmentLayer(
            name="Multiply Blend",
            mask_data=MaskData.radial((0.5, 0.5), (0.4, 0.4)),
            adjustments={'exposure': 0.5},
            blend_mode='multiply'
        )
        
        result = processor.apply_adjustment_layers(test_image, [layer])
        
        # Multiply should darken more than normal blend
        center = result[100, 100, 0]
        assert center < test_image[100, 100, 0]


class TestMaskCombination:
    """Test combining multiple masks."""
    
    def test_mask_combination(self):
        """Test combining masks with different blend modes."""
        shape = (100, 100)
        
        # Create two overlapping masks
        mask1 = MaskGenerator.generate_mask(
            shape, 
            MaskData.radial((0.3, 0.5), (0.3, 0.3))
        )
        mask2 = MaskGenerator.generate_mask(
            shape,
            MaskData.radial((0.7, 0.5), (0.3, 0.3))
        )
        
        # Test add mode
        combined_add = MaskGenerator.combine_masks(
            [mask1, mask2],
            ['normal', 'add'],
            [1.0, 1.0]
        )
        # Overlap should be brighter
        assert combined_add[50, 50] > mask1[50, 50]
        assert combined_add[50, 50] > mask2[50, 50]
        
        # Test multiply mode
        combined_multiply = MaskGenerator.combine_masks(
            [mask1, mask2],
            ['normal', 'multiply'],
            [1.0, 1.0]
        )
        # Should be darker than individual masks
        assert combined_multiply.max() < mask2.max()


class TestIntegration:
    """Integration tests with processing pipeline."""
    
    def test_recipe_with_adjustment_layers(self):
        """Test that recipes can contain adjustment layers."""
        from photosight.processing.raw_processor import ProcessingRecipe
        
        recipe = ProcessingRecipe(
            source_path="/test/image.raw",
            adjustment_layers=[
                AdjustmentLayer(
                    name="Graduated Filter",
                    mask_data=MaskData.linear((0, 0), (0, 0.5)),
                    adjustments={'exposure': -0.5}
                ),
                AdjustmentLayer(
                    name="Radial Filter",
                    mask_data=MaskData.radial((0.5, 0.5), (0.4, 0.4)),
                    adjustments={'clarity': 30}
                )
            ]
        )
        
        assert len(recipe.adjustment_layers) == 2
        assert recipe.adjustment_layers[0].name == "Graduated Filter"
    
    def test_performance_with_cache(self):
        """Test that mask caching improves performance."""
        import time
        
        config = LocalAdjustmentConfig(cache_masks=True)
        processor = LocalAdjustmentProcessor(config)
        
        # Large image for performance testing
        image = np.random.rand(1000, 1000, 3).astype(np.float32)
        
        layer = AdjustmentLayer(
            name="Complex Mask",
            mask_data=MaskData.radial((0.5, 0.5), (0.4, 0.4)),
            adjustments={'exposure': 1.0}
        )
        
        # First run (no cache)
        start = time.time()
        result1 = processor.apply_adjustment_layers(image, [layer])
        time1 = time.time() - start
        
        # Second run (with cache)
        start = time.time()
        result2 = processor.apply_adjustment_layers(image, [layer])
        time2 = time.time() - start
        
        # Cache should make it faster
        assert time2 < time1
        # Results should be identical
        np.testing.assert_array_almost_equal(result1, result2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])