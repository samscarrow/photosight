"""
Camera-specific noise profiles for adaptive noise reduction.
"""

from typing import Dict, Optional, List
from .models import NoiseProfile


class CameraNoiseProfiles:
    """
    Database of camera-specific noise profiles.
    
    These profiles are based on extensive testing and analysis
    of various camera models at different ISO settings.
    """
    
    # Sony A7 III noise profiles
    SONY_A7M3_PROFILES = {
        100: NoiseProfile(
            camera_model="Sony A7 III",
            iso_value=100,
            luminance_noise_level=2.0,
            chrominance_noise_level=1.0,
            noise_grain_size=0.8,
            suggested_luminance_reduction=0.0,
            suggested_chrominance_reduction=10.0,
            suggested_detail_preservation=90.0
        ),
        400: NoiseProfile(
            camera_model="Sony A7 III",
            iso_value=400,
            luminance_noise_level=5.0,
            chrominance_noise_level=3.0,
            noise_grain_size=0.9,
            suggested_luminance_reduction=15.0,
            suggested_chrominance_reduction=25.0,
            suggested_detail_preservation=85.0
        ),
        1600: NoiseProfile(
            camera_model="Sony A7 III",
            iso_value=1600,
            luminance_noise_level=12.0,
            chrominance_noise_level=8.0,
            noise_grain_size=1.0,
            suggested_luminance_reduction=30.0,
            suggested_chrominance_reduction=40.0,
            suggested_detail_preservation=75.0
        ),
        6400: NoiseProfile(
            camera_model="Sony A7 III",
            iso_value=6400,
            luminance_noise_level=25.0,
            chrominance_noise_level=18.0,
            noise_grain_size=1.2,
            blue_channel_factor=1.3,
            suggested_luminance_reduction=45.0,
            suggested_chrominance_reduction=55.0,
            suggested_detail_preservation=65.0
        ),
        12800: NoiseProfile(
            camera_model="Sony A7 III",
            iso_value=12800,
            luminance_noise_level=40.0,
            chrominance_noise_level=30.0,
            noise_grain_size=1.4,
            blue_channel_factor=1.4,
            has_pattern_noise=True,
            suggested_luminance_reduction=60.0,
            suggested_chrominance_reduction=70.0,
            suggested_detail_preservation=55.0
        ),
        25600: NoiseProfile(
            camera_model="Sony A7 III",
            iso_value=25600,
            luminance_noise_level=60.0,
            chrominance_noise_level=45.0,
            noise_grain_size=1.6,
            blue_channel_factor=1.5,
            has_pattern_noise=True,
            has_banding=True,
            hot_pixel_threshold=20000.0,
            suggested_luminance_reduction=75.0,
            suggested_chrominance_reduction=80.0,
            suggested_detail_preservation=45.0
        )
    }
    
    # Canon 5D Mark IV profiles
    CANON_5DM4_PROFILES = {
        100: NoiseProfile(
            camera_model="Canon 5D Mark IV",
            iso_value=100,
            luminance_noise_level=1.5,
            chrominance_noise_level=1.0,
            noise_grain_size=0.7,
            suggested_luminance_reduction=0.0,
            suggested_chrominance_reduction=10.0,
            suggested_detail_preservation=95.0
        ),
        1600: NoiseProfile(
            camera_model="Canon 5D Mark IV",
            iso_value=1600,
            luminance_noise_level=10.0,
            chrominance_noise_level=6.0,
            noise_grain_size=0.9,
            suggested_luminance_reduction=25.0,
            suggested_chrominance_reduction=35.0,
            suggested_detail_preservation=80.0
        ),
        6400: NoiseProfile(
            camera_model="Canon 5D Mark IV",
            iso_value=6400,
            luminance_noise_level=22.0,
            chrominance_noise_level=15.0,
            noise_grain_size=1.1,
            blue_channel_factor=1.25,
            suggested_luminance_reduction=40.0,
            suggested_chrominance_reduction=50.0,
            suggested_detail_preservation=70.0
        )
    }
    
    # Nikon Z7 II profiles
    NIKON_Z7II_PROFILES = {
        64: NoiseProfile(
            camera_model="Nikon Z7 II",
            iso_value=64,
            luminance_noise_level=1.0,
            chrominance_noise_level=0.5,
            noise_grain_size=0.6,
            suggested_luminance_reduction=0.0,
            suggested_chrominance_reduction=5.0,
            suggested_detail_preservation=98.0
        ),
        1600: NoiseProfile(
            camera_model="Nikon Z7 II",
            iso_value=1600,
            luminance_noise_level=8.0,
            chrominance_noise_level=5.0,
            noise_grain_size=0.85,
            suggested_luminance_reduction=20.0,
            suggested_chrominance_reduction=30.0,
            suggested_detail_preservation=82.0
        ),
        6400: NoiseProfile(
            camera_model="Nikon Z7 II",
            iso_value=6400,
            luminance_noise_level=20.0,
            chrominance_noise_level=12.0,
            noise_grain_size=1.05,
            blue_channel_factor=1.2,
            suggested_luminance_reduction=35.0,
            suggested_chrominance_reduction=45.0,
            suggested_detail_preservation=72.0
        )
    }
    
    # Fujifilm X-T4 profiles (X-Trans sensor)
    FUJI_XT4_PROFILES = {
        160: NoiseProfile(
            camera_model="Fujifilm X-T4",
            iso_value=160,
            luminance_noise_level=2.5,
            chrominance_noise_level=1.5,
            noise_grain_size=0.9,
            noise_pattern="x-trans",  # Special pattern
            suggested_luminance_reduction=5.0,
            suggested_chrominance_reduction=15.0,
            suggested_detail_preservation=90.0
        ),
        1600: NoiseProfile(
            camera_model="Fujifilm X-T4",
            iso_value=1600,
            luminance_noise_level=11.0,
            chrominance_noise_level=7.0,
            noise_grain_size=1.0,
            noise_pattern="x-trans",
            suggested_luminance_reduction=28.0,
            suggested_chrominance_reduction=38.0,
            suggested_detail_preservation=78.0
        ),
        6400: NoiseProfile(
            camera_model="Fujifilm X-T4",
            iso_value=6400,
            luminance_noise_level=24.0,
            chrominance_noise_level=16.0,
            noise_grain_size=1.15,
            noise_pattern="x-trans",
            blue_channel_factor=1.3,
            suggested_luminance_reduction=42.0,
            suggested_chrominance_reduction=52.0,
            suggested_detail_preservation=68.0
        )
    }
    
    # All profiles organized by camera
    ALL_PROFILES: Dict[str, Dict[int, NoiseProfile]] = {
        "Sony A7 III": SONY_A7M3_PROFILES,
        "Sony ILCE-7M3": SONY_A7M3_PROFILES,  # Alternative name
        "Canon 5D Mark IV": CANON_5DM4_PROFILES,
        "Canon EOS 5D Mark IV": CANON_5DM4_PROFILES,
        "Nikon Z7 II": NIKON_Z7II_PROFILES,
        "Nikon Z 7II": NIKON_Z7II_PROFILES,
        "Fujifilm X-T4": FUJI_XT4_PROFILES,
        "FUJIFILM X-T4": FUJI_XT4_PROFILES
    }


def get_noise_profile(camera_model: str, iso: int) -> Optional[NoiseProfile]:
    """
    Get noise profile for specific camera and ISO.
    
    Args:
        camera_model: Camera model name from EXIF
        iso: ISO value
        
    Returns:
        NoiseProfile if found, None otherwise
    """
    # Check if we have profiles for this camera
    if camera_model not in CameraNoiseProfiles.ALL_PROFILES:
        # Try to find a close match
        for key in CameraNoiseProfiles.ALL_PROFILES:
            if key.lower() in camera_model.lower() or camera_model.lower() in key.lower():
                camera_model = key
                break
        else:
            return None
    
    camera_profiles = CameraNoiseProfiles.ALL_PROFILES[camera_model]
    
    # Exact match
    if iso in camera_profiles:
        return camera_profiles[iso]
    
    # Find closest ISO
    iso_values = sorted(camera_profiles.keys())
    
    # If below minimum ISO, use minimum
    if iso < iso_values[0]:
        return camera_profiles[iso_values[0]]
    
    # If above maximum ISO, use maximum
    if iso > iso_values[-1]:
        return camera_profiles[iso_values[-1]]
    
    # Interpolate between two closest values
    for i in range(len(iso_values) - 1):
        if iso_values[i] <= iso <= iso_values[i + 1]:
            # Simple interpolation of noise levels
            lower_profile = camera_profiles[iso_values[i]]
            upper_profile = camera_profiles[iso_values[i + 1]]
            
            # Calculate interpolation factor
            factor = (iso - iso_values[i]) / (iso_values[i + 1] - iso_values[i])
            
            # Create interpolated profile
            return NoiseProfile(
                camera_model=camera_model,
                iso_value=iso,
                luminance_noise_level=(
                    lower_profile.luminance_noise_level * (1 - factor) +
                    upper_profile.luminance_noise_level * factor
                ),
                chrominance_noise_level=(
                    lower_profile.chrominance_noise_level * (1 - factor) +
                    upper_profile.chrominance_noise_level * factor
                ),
                noise_grain_size=(
                    lower_profile.noise_grain_size * (1 - factor) +
                    upper_profile.noise_grain_size * factor
                ),
                noise_pattern=lower_profile.noise_pattern,
                red_channel_factor=(
                    lower_profile.red_channel_factor * (1 - factor) +
                    upper_profile.red_channel_factor * factor
                ),
                blue_channel_factor=(
                    lower_profile.blue_channel_factor * (1 - factor) +
                    upper_profile.blue_channel_factor * factor
                ),
                has_banding=upper_profile.has_banding if factor > 0.5 else lower_profile.has_banding,
                has_pattern_noise=upper_profile.has_pattern_noise if factor > 0.5 else lower_profile.has_pattern_noise,
                hot_pixel_threshold=(
                    lower_profile.hot_pixel_threshold * (1 - factor) +
                    upper_profile.hot_pixel_threshold * factor
                ),
                suggested_luminance_reduction=(
                    lower_profile.suggested_luminance_reduction * (1 - factor) +
                    upper_profile.suggested_luminance_reduction * factor
                ),
                suggested_chrominance_reduction=(
                    lower_profile.suggested_chrominance_reduction * (1 - factor) +
                    upper_profile.suggested_chrominance_reduction * factor
                ),
                suggested_detail_preservation=(
                    lower_profile.suggested_detail_preservation * (1 - factor) +
                    upper_profile.suggested_detail_preservation * factor
                )
            )
    
    return None


def get_generic_profile(iso: int) -> NoiseProfile:
    """
    Get a generic noise profile for unknown cameras.
    
    Args:
        iso: ISO value
        
    Returns:
        Generic NoiseProfile based on ISO
    """
    # Generic noise characteristics based on ISO
    if iso <= 200:
        lum_noise = 3.0
        chr_noise = 2.0
        lum_reduction = 5.0
        chr_reduction = 15.0
        detail_preserve = 90.0
    elif iso <= 800:
        lum_noise = 8.0
        chr_noise = 5.0
        lum_reduction = 20.0
        chr_reduction = 30.0
        detail_preserve = 85.0
    elif iso <= 3200:
        lum_noise = 18.0
        chr_noise = 12.0
        lum_reduction = 35.0
        chr_reduction = 45.0
        detail_preserve = 75.0
    elif iso <= 12800:
        lum_noise = 35.0
        chr_noise = 25.0
        lum_reduction = 55.0
        chr_reduction = 65.0
        detail_preserve = 60.0
    else:
        lum_noise = 55.0
        chr_noise = 40.0
        lum_reduction = 70.0
        chr_reduction = 80.0
        detail_preserve = 45.0
    
    return NoiseProfile(
        camera_model="Generic",
        iso_value=iso,
        luminance_noise_level=lum_noise,
        chrominance_noise_level=chr_noise,
        noise_grain_size=1.0 + (iso / 25600),  # Grain gets coarser
        blue_channel_factor=1.0 + (iso / 51200),  # Blue gets noisier
        has_pattern_noise=iso > 6400,
        has_banding=iso > 12800,
        hot_pixel_threshold=float(iso) if iso > 12800 else 0.0,
        suggested_luminance_reduction=lum_reduction,
        suggested_chrominance_reduction=chr_reduction,
        suggested_detail_preservation=detail_preserve
    )