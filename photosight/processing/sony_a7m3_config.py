#!/usr/bin/env python3
"""
Sony Alpha 7 Mark III specific configuration and optimizations.

This module contains camera-specific settings optimized for the Sony A7 III,
including recommended processing parameters, lens profiles, and workflow presets.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

@dataclass 
class SonyA7M3Profile:
    """Optimized processing profile for Sony Alpha 7 Mark III"""
    
    # Camera characteristics
    camera_make: str = "Sony"
    camera_model: str = "ILCE-7M3"
    
    # ISO performance
    base_iso: int = 100
    sweet_spot_iso: int = 400          # Best image quality
    high_iso_threshold: int = 2500     # When to switch processing approach
    max_usable_iso: int = 12800        # Beyond this, aggressive noise reduction needed
    
    # Dynamic range characteristics  
    highlight_headroom: float = 1.3    # Stops of highlight recovery available
    shadow_lift_capability: float = 4.0  # Stops of shadow lifting without artifacts
    base_dynamic_range: float = 14.7   # Measured dynamic range at base ISO
    
    # White balance characteristics
    wb_confidence_boost: float = 1.2   # Sony auto WB is quite good
    wb_temperature_accuracy: int = 150  # ±150K typical accuracy
    
    # Color characteristics
    color_grading_bias: str = "cinematic"  # Works well with Sony color science
    color_depth: int = 14              # 14-bit A/D conversion
    
    # Noise characteristics
    noise_floor: float = 0.001         # Base noise level
    noise_pattern: str = "fine_grain"  # Sony's noise pattern type
    
    # Lens mount specific
    mount_type: str = "FE"             # Sony FE mount
    crop_factor: float = 1.0           # Full frame


class SonyA7M3Optimizer:
    """Processing optimizer specifically tuned for Sony A7 III"""
    
    def __init__(self):
        self.profile = SonyA7M3Profile()
        self.lens_profiles = self._init_lens_profiles()
        
    def _init_lens_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common Sony FE lens profiles"""
        return {
            # Sony native lenses
            'FE 24-70mm F2.8 GM': {
                'vignetting_correction': 0.3,
                'distortion_correction': 0.2, 
                'chromatic_aberration': 0.1,
                'optimal_aperture': 5.6,
                'character': 'clinical_sharp'
            },
            'FE 70-200mm F2.8 GM OSS': {
                'vignetting_correction': 0.4,
                'distortion_correction': 0.1,
                'chromatic_aberration': 0.2,
                'optimal_aperture': 4.0,
                'character': 'creamy_bokeh'
            },
            'FE 85mm F1.4 GM': {
                'vignetting_correction': 0.5,
                'distortion_correction': 0.0,
                'chromatic_aberration': 0.1,
                'optimal_aperture': 2.8,
                'character': 'portrait_rendering'
            },
            'FE 16-35mm F2.8 GM': {
                'vignetting_correction': 0.6,
                'distortion_correction': 0.4,
                'chromatic_aberration': 0.2,
                'optimal_aperture': 8.0,
                'character': 'wide_angle_correction'
            },
            
            # Third-party popular lenses
            'Sigma 24-70mm F2.8 DG DN Art': {
                'vignetting_correction': 0.3,
                'distortion_correction': 0.2,
                'chromatic_aberration': 0.15,
                'optimal_aperture': 5.6,
                'character': 'neutral_sharp'
            },
            'Tamron 28-75mm F2.8 Di III VXD G2': {
                'vignetting_correction': 0.4,
                'distortion_correction': 0.3,
                'chromatic_aberration': 0.2,
                'optimal_aperture': 5.6,
                'character': 'versatile_standard'
            }
        }
    
    def get_optimal_processing_settings(self, iso: int, aperture: float, 
                                      lens_model: str = None) -> Dict[str, Any]:
        """
        Get optimal processing settings based on shooting parameters
        
        Args:
            iso: ISO setting used
            aperture: Aperture value  
            lens_model: Lens model if available
            
        Returns:
            Dictionary of optimal processing parameters
        """
        settings = {
            'exposure_compensation': 0.0,
            'shadows': 0,
            'highlights': 0,
            'whites': 0,
            'blacks': 0,
            'clarity': 0,
            'vibrance': 0,
            'saturation': 0
        }
        
        # ISO-based adjustments
        if iso <= self.profile.base_iso:
            # Base ISO - minimal processing needed
            settings.update({
                'shadows': 5,          # Slight shadow lift for Sony files
                'clarity': 10,         # Enhance micro-contrast
                'vibrance': 15         # Boost colors naturally
            })
            
        elif iso <= self.profile.sweet_spot_iso:
            # Sweet spot range - gentle enhancements
            settings.update({
                'shadows': 10,
                'highlights': -5,      # Protect highlights slightly
                'clarity': 5,
                'vibrance': 10
            })
            
        elif iso <= self.profile.high_iso_threshold:
            # Good quality range - moderate processing
            settings.update({
                'shadows': 15,         # More shadow lifting needed
                'highlights': -10,     # More highlight protection
                'clarity': 0,          # No local contrast to avoid noise
                'vibrance': 5,         # Reduce color enhancement
                'noise_reduction': 20  # Start noise reduction
            })
            
        else:
            # High ISO - aggressive processing
            settings.update({
                'shadows': 25,         # Significant shadow recovery
                'highlights': -15,     # Strong highlight protection  
                'clarity': -10,        # Slight negative clarity for smoothing
                'vibrance': 0,         # Minimal color enhancement
                'saturation': -5,      # Slight desaturation for cleaner look
                'noise_reduction': 40  # Strong noise reduction
            })
        
        # Aperture-based adjustments
        if aperture <= 2.0:
            # Wide open - may need vignetting correction
            settings['vignetting_correction'] = 0.4
            settings['highlights'] -= 5  # Extra highlight protection
            
        elif aperture >= 8.0:
            # Stopped down - enhance sharpness
            settings['clarity'] += 10
            settings['structure'] = 5
        
        # Lens-specific adjustments
        if lens_model and lens_model in self.lens_profiles:
            lens_profile = self.lens_profiles[lens_model]
            settings.update({
                'vignetting_correction': lens_profile['vignetting_correction'],
                'distortion_correction': lens_profile['distortion_correction'], 
                'chromatic_aberration_correction': lens_profile['chromatic_aberration']
            })
            
            # Apply lens character adjustments
            character = lens_profile['character']
            if character == 'portrait_rendering':
                settings['shadows'] += 5
                settings['highlights'] -= 5
                settings['vibrance'] += 5
            elif character == 'wide_angle_correction':
                settings['clarity'] += 15
                settings['structure'] = 10
        
        logger.info(f"Generated A7M3 optimal settings for ISO {iso}, f/{aperture}")
        return settings
    
    def get_recommended_workflow(self, photo_count: int, 
                                project_type: str = "general") -> Dict[str, Any]:
        """
        Get recommended workflow settings for Sony A7 III projects
        
        Args:
            photo_count: Number of photos in project
            project_type: Type of photography project
            
        Returns:
            Workflow recommendations
        """
        workflows = {
            "portrait": {
                "color_grading": "cinematic",
                "skin_tone_protection": True,
                "local_adjustments": True,
                "export_format": "tiff_16bit",
                "sharpening": "portraits"
            },
            "landscape": {
                "color_grading": "bright_airy", 
                "clarity_boost": True,
                "hdr_processing": True,
                "export_format": "tiff_16bit",
                "sharpening": "landscapes"
            },
            "wedding": {
                "color_grading": "film_emulation",
                "batch_consistency": True,
                "fast_export": True,
                "export_format": "jpeg_max",
                "sharpening": "standard"
            },
            "general": {
                "color_grading": "cinematic",
                "auto_adjustments": True,
                "export_format": "jpeg_high",
                "sharpening": "standard"
            }
        }
        
        base_workflow = workflows.get(project_type, workflows["general"])
        
        # Adjust based on photo count
        if photo_count > 500:
            base_workflow.update({
                "batch_processing": True,
                "auto_culling": True,
                "fast_export": True
            })
        
        return base_workflow


# Global instance for easy access
sony_a7m3_optimizer = SonyA7M3Optimizer()

def get_a7m3_settings(iso: int, aperture: float, lens_model: str = None) -> Dict[str, Any]:
    """Convenience function to get A7M3 optimal settings"""
    return sony_a7m3_optimizer.get_optimal_processing_settings(iso, aperture, lens_model)