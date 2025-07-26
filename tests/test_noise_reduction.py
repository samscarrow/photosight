"""
Tests for the noise reduction system.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from photosight.processing.noise import (
    NoiseReducer, NoiseReductionSettings, NoiseProfile,
    EdgePreservationMethod, get_noise_profile
)
from photosight.processing.noise.profiles import CameraNoiseProfiles


class TestNoiseProfiles:
    """Test camera-specific noise profiles."""
    
    def test_get_exact_profile(self):
        """Test retrieving exact ISO match."""
        profile = get_noise_profile("Sony A7 III", 1600)
        assert profile is not None
        assert profile.iso_value == 1600
        assert profile.luminance_noise_level == 12.0
        assert profile.chrominance_noise_level == 8.0
        
    def test_get_interpolated_profile(self):
        """Test ISO interpolation."""
        # ISO 800 should interpolate between 400 and 1600
        profile = get_noise_profile("Sony A7 III", 800)
        assert profile is not None
        assert profile.iso_value == 800
        # Should be between 400 and 1600 values
        assert 5.0 < profile.luminance_noise_level < 12.0
        assert 3.0 < profile.chrominance_noise_level < 8.0
        
    def test_camera_name_variants(self):
        """Test different camera name formats."""
        # Both should work
        p1 = get_noise_profile("Sony A7 III", 6400)
        p2 = get_noise_profile("Sony ILCE-7M3", 6400)
        assert p1 is not None
        assert p2 is not None
        assert p1.luminance_noise_level == p2.luminance_noise_level
        
    def test_generic_profile_fallback(self):
        """Test generic profile for unknown cameras."""
        from photosight.processing.noise.profiles import get_generic_profile
        
        profile = get_generic_profile(3200)
        assert profile.camera_model == "Generic"
        assert profile.iso_value == 3200
        assert profile.luminance_noise_level == 18.0
        
    def test_high_iso_characteristics(self):
        """Test high ISO specific issues."""
        profile = get_noise_profile("Sony A7 III", 25600)
        assert profile.has_pattern_noise is True
        assert profile.has_banding is True
        assert profile.hot_pixel_threshold > 0
        assert profile.blue_channel_factor > 1.0


class TestNoiseReductionSettings:
    """Test noise reduction settings."""
    
    def test_default_settings(self):
        """Test default settings initialization."""
        settings = NoiseReductionSettings()
        assert settings.luminance_amount == 0.0
        assert settings.chrominance_amount == 50.0
        assert settings.detail_preservation == 50.0
        assert settings.iso_adaptive is True
        
    def test_auto_settings_from_profile(self):
        """Test generating auto settings from profile."""
        profile = get_noise_profile("Sony A7 III", 6400)
        settings = profile.get_auto_settings()
        
        assert settings.luminance_amount == profile.suggested_luminance_reduction
        assert settings.chrominance_amount == profile.suggested_chrominance_reduction
        assert settings.detail_preservation == profile.suggested_detail_preservation
        assert settings.iso_adaptive is True
        
    def test_scale_settings(self):
        """Test scaling settings by factor."""
        settings = NoiseReductionSettings(
            luminance_amount=50.0,
            chrominance_amount=60.0
        )
        
        scaled = settings.scale_by_factor(0.5)
        assert scaled.luminance_amount == 25.0
        assert scaled.chrominance_amount == 30.0
        # Detail preservation should not change
        assert scaled.detail_preservation == settings.detail_preservation


class TestNoiseReducer:
    """Test the main noise reduction processor."""
    
    @pytest.fixture
    def sample_noisy_image(self):
        """Create a synthetic noisy image."""
        # Create base gradient
        height, width = 512, 512
        base = np.zeros((height, width, 3), dtype=np.float32)
        
        # Add gradient
        for i in range(height):
            base[i, :] = i / height
            
        # Add noise
        noise = np.random.normal(0, 0.05, base.shape)
        noisy = np.clip(base + noise, 0, 1)
        
        return noisy
    
    @pytest.fixture
    def reducer(self):
        """Create noise reducer instance."""
        return NoiseReducer()
    
    def test_basic_noise_reduction(self, reducer, sample_noisy_image):
        """Test basic noise reduction."""
        settings = NoiseReductionSettings(
            luminance_amount=50.0,
            chrominance_amount=50.0,
            quality_mode="fast"
        )
        
        result = reducer.reduce_noise(sample_noisy_image, settings)
        
        # Result should be valid
        assert result.shape == sample_noisy_image.shape
        assert result.dtype == np.float32
        assert result.min() >= 0
        assert result.max() <= 1
        
        # Should have reduced noise (lower std dev)
        orig_std = np.std(sample_noisy_image)
        result_std = np.std(result)
        assert result_std < orig_std
    
    def test_iso_adaptive_reduction(self, reducer, sample_noisy_image):
        """Test ISO-adaptive noise reduction."""
        settings = NoiseReductionSettings(
            iso_adaptive=True,
            luminance_amount=0,  # Should be auto-set
            chrominance_amount=0
        )
        
        # High ISO should trigger automatic settings
        result = reducer.reduce_noise(
            sample_noisy_image, 
            settings,
            iso=6400,
            camera_model="Sony A7 III"
        )
        
        # Should have applied noise reduction
        assert np.std(result) < np.std(sample_noisy_image)
    
    def test_edge_preservation_methods(self, reducer):
        """Test different edge preservation methods."""
        # Create image with sharp edge
        image = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        image[:, 128:] = 0.8
        
        # Add noise
        noise = np.random.normal(0, 0.05, image.shape)
        noisy = np.clip(image + noise, 0, 1)
        
        methods = [
            EdgePreservationMethod.BILATERAL,
            EdgePreservationMethod.GUIDED,
            EdgePreservationMethod.NLM,
            EdgePreservationMethod.WAVELET
        ]
        
        for method in methods:
            settings = NoiseReductionSettings(
                luminance_amount=50.0,
                edge_preservation_method=method,
                quality_mode="high" if method in [EdgePreservationMethod.NLM, EdgePreservationMethod.WAVELET] else "balanced"
            )
            
            result = reducer.reduce_noise(noisy, settings)
            
            # Check edge is preserved (measure gradient)
            edge_orig = np.abs(np.mean(image[:, 129]) - np.mean(image[:, 127]))
            edge_result = np.abs(np.mean(result[:, 129]) - np.mean(result[:, 127]))
            
            # Edge should be mostly preserved (at least 70%)
            assert edge_result > edge_orig * 0.7
    
    def test_multi_scale_processing(self, reducer, sample_noisy_image):
        """Test multi-scale noise reduction."""
        settings = NoiseReductionSettings(
            luminance_amount=50.0,
            quality_mode="high",
            multi_scale=True
        )
        
        result = reducer.reduce_noise(sample_noisy_image, settings)
        
        # Should produce valid result
        assert result.shape == sample_noisy_image.shape
        assert np.std(result) < np.std(sample_noisy_image)
    
    def test_hot_pixel_removal(self, reducer):
        """Test hot pixel removal."""
        # Create image with hot pixels
        image = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        
        # Add hot pixels
        hot_pixels = np.random.randint(0, 256, size=(20, 2))
        for y, x in hot_pixels:
            image[y, x] = 1.0  # Bright hot pixel
            
        settings = NoiseReductionSettings(
            hot_pixel_removal=True,
            luminance_amount=0  # Only hot pixel removal
        )
        
        result = reducer.reduce_noise(image, settings, iso=12800)
        
        # Hot pixels should be removed
        for y, x in hot_pixels:
            # Should be close to surrounding average, not 1.0
            assert result[y, x].max() < 0.9
    
    def test_chrominance_noise_reduction(self, reducer):
        """Test color noise reduction."""
        # Create grayscale image with color noise
        image = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        
        # Add color noise (different per channel)
        image[:, :, 0] += np.random.normal(0, 0.05, (256, 256))
        image[:, :, 1] += np.random.normal(0, 0.05, (256, 256))
        image[:, :, 2] += np.random.normal(0, 0.05, (256, 256))
        image = np.clip(image, 0, 1)
        
        settings = NoiseReductionSettings(
            luminance_amount=0,
            chrominance_amount=75.0,
            quality_mode="high"
        )
        
        result = reducer.reduce_noise(image, settings)
        
        # Color channels should be more similar (less color noise)
        orig_color_var = np.var(image, axis=2).mean()
        result_color_var = np.var(result, axis=2).mean()
        assert result_color_var < orig_color_var * 0.5
    
    def test_mask_application(self, reducer, sample_noisy_image):
        """Test masked noise reduction."""
        # Create mask for half the image
        mask = np.zeros((512, 512), dtype=np.float32)
        mask[:, 256:] = 1.0
        
        settings = NoiseReductionSettings(
            luminance_amount=50.0,
            mask_enabled=True
        )
        
        result = reducer.reduce_noise(sample_noisy_image, settings, mask=mask)
        
        # Left half should be unchanged
        left_diff = np.abs(result[:, :256] - sample_noisy_image[:, :256])
        assert left_diff.max() < 0.01
        
        # Right half should be denoised
        right_orig_std = np.std(sample_noisy_image[:, 256:])
        right_result_std = np.std(result[:, 256:])
        assert right_result_std < right_orig_std
    
    def test_quality_modes(self, reducer, sample_noisy_image):
        """Test different quality modes."""
        modes = ["fast", "balanced", "high"]
        
        for mode in modes:
            settings = NoiseReductionSettings(
                luminance_amount=50.0,
                quality_mode=mode
            )
            
            result = reducer.reduce_noise(sample_noisy_image, settings)
            
            # All modes should produce valid results
            assert result.shape == sample_noisy_image.shape
            assert np.std(result) < np.std(sample_noisy_image)
    
    def test_real_image_if_available(self, reducer):
        """Test with real image if available."""
        test_image_path = Path("test_data/noisy_high_iso.jpg")
        if not test_image_path.exists():
            pytest.skip("Test image not available")
            
        # Load and convert to float32
        image = cv2.imread(str(test_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Apply noise reduction
        settings = NoiseReductionSettings(
            luminance_amount=50.0,
            chrominance_amount=60.0,
            quality_mode="high",
            iso_adaptive=True
        )
        
        result = reducer.reduce_noise(
            image, 
            settings,
            iso=6400,
            camera_model="Sony A7 III"
        )
        
        # Save result for visual inspection
        output_path = Path("test_output/noise_reduction_result.jpg")
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        
        # Basic validation
        assert result.shape == image.shape
        assert np.std(result) < np.std(image)


class TestNoiseAnalysis:
    """Test noise analysis functionality."""
    
    def test_noise_analysis(self):
        """Test image noise analysis."""
        # Create test image with known noise
        image = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        noise = np.random.normal(0, 0.05, image.shape)
        noisy = np.clip(image + noise, 0, 1)
        
        reducer = NoiseReducer()
        analysis = reducer._analyze_noise(noisy)
        
        # Should detect reasonable noise levels
        assert 0 < analysis.estimated_luminance_noise < 100
        assert 0 < analysis.estimated_chrominance_noise < 100
        assert analysis.dominant_noise_frequency in ["fine", "medium", "coarse"]
        assert 0 <= analysis.noise_uniformity <= 1


class TestIntegration:
    """Test integration with processing pipeline."""
    
    def test_processing_recipe_integration(self):
        """Test integration with ProcessingRecipe."""
        from photosight.processing import ProcessingRecipe
        
        # Create recipe with noise reduction
        recipe = ProcessingRecipe(
            source_path="test.raw",
            noise_reduction=NoiseReductionSettings(
                luminance_amount=40.0,
                chrominance_amount=50.0,
                iso_adaptive=True
            )
        )
        
        # Should serialize properly
        recipe_dict = recipe.to_dict()
        assert "noise_reduction" in recipe_dict
        assert recipe_dict["noise_reduction"]["luminance_amount"] == 40.0
        
        # Should deserialize properly
        recipe2 = ProcessingRecipe.from_dict(recipe_dict)
        assert recipe2.noise_reduction is not None
        assert recipe2.noise_reduction.luminance_amount == 40.0