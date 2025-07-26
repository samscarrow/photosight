"""
Technical quality analysis for photos.

Analyzes technical aspects like sharpness, exposure, noise, and other
objective quality metrics.
"""

import logging
import numpy as np
from PIL import Image, ExifTags
from pathlib import Path
from typing import Dict, Union, Optional
import cv2

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Analyzes technical quality aspects of photos.
    
    Provides objective measurements of:
    - Sharpness/blur detection
    - Exposure quality
    - Noise levels
    - Contrast and dynamic range
    - Color accuracy
    """
    
    def __init__(self, config: Dict):
        """Initialize the technical analyzer."""
        self.config = config
        self.analysis_config = config.get('technical_analysis', {})
        
    def analyze_photo(self, photo_input: Union[str, Path, Image.Image], photo_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Perform comprehensive technical analysis of a photo.
        
        Args:
            photo_input: Either a path to the photo file OR a PIL Image object
            photo_path: Optional path for metadata (used when photo_input is an Image)
            
        Returns:
            Dictionary containing technical analysis results
        """
        try:
            # Handle both file path and Image object inputs
            if isinstance(photo_input, Image.Image):
                image = photo_input
                # If photo_path not provided, use a default name for logging
                file_path = Path(photo_path) if photo_path else Path("unknown_image")
            else:
                # Traditional file path input
                file_path = Path(photo_input)
                image = Image.open(file_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Perform various technical analyses
            results = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'image_dimensions': {
                    'width': image.width,
                    'height': image.height,
                    'aspect_ratio': image.width / image.height
                }
            }
            
            # Sharpness analysis
            sharpness_score = self._analyze_sharpness(img_array)
            results['sharpness'] = sharpness_score
            results['is_sharp'] = sharpness_score > self.analysis_config.get('sharpness_threshold', 100)
            
            # Exposure analysis
            exposure_analysis = self._analyze_exposure(img_array)
            results.update(exposure_analysis)
            
            # Noise analysis
            noise_analysis = self._analyze_noise(img_array)
            results.update(noise_analysis)
            
            # Contrast and dynamic range
            contrast_analysis = self._analyze_contrast(img_array)
            results.update(contrast_analysis)
            
            # Color analysis
            color_analysis = self._analyze_color(img_array)
            results.update(color_analysis)
            
            # EXIF data analysis
            exif_analysis = self._analyze_exif(image)
            results.update(exif_analysis)
            
            # Overall technical quality score
            results['technical_quality'] = self._calculate_technical_score(results)
            
            return results
            
        except Exception as e:
            # Handle error logging for both input types
            if isinstance(photo_input, Image.Image):
                error_path = photo_path if photo_path else "unknown_image"
            else:
                error_path = photo_input
            
            logger.error(f"Error analyzing photo {error_path}: {e}")
            return {
                'error': str(e),
                'file_path': str(error_path),
                'file_name': Path(error_path).name if error_path != "unknown_image" else "unknown_image",
                'technical_quality': 0.0
            }
    
    def _analyze_sharpness(self, img_array: np.ndarray) -> float:
        """
        Analyze image sharpness using Laplacian variance.
        
        Returns:
            Sharpness score (higher is sharper)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            return float(laplacian_var)
            
        except Exception as e:
            logger.warning(f"Sharpness analysis error: {e}")
            return 0.0
    
    def _analyze_exposure(self, img_array: np.ndarray) -> Dict:
        """
        Analyze exposure quality and characteristics.
        
        Returns:
            Dictionary with exposure analysis results
        """
        try:
            # Convert to grayscale for luminance analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate histogram
            hist, bins = np.histogram(gray, bins=256, range=(0, 256))
            
            # Normalize histogram
            hist_norm = hist / hist.sum()
            
            # Calculate statistics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Check for clipping
            shadows_clipped = hist_norm[0] > 0.01  # More than 1% pure black
            highlights_clipped = hist_norm[-1] > 0.01  # More than 1% pure white
            
            # Calculate exposure quality score
            # Ideal exposure has mean around 128 with good spread
            exposure_score = 1.0 - abs(mean_brightness - 128) / 128
            exposure_score *= (1.0 if std_brightness > 30 else std_brightness / 30)  # Penalize low contrast
            
            if shadows_clipped or highlights_clipped:
                exposure_score *= 0.7  # Penalty for clipping
            
            return {
                'mean_brightness': float(mean_brightness),
                'brightness_std': float(std_brightness),
                'shadows_clipped': shadows_clipped,
                'highlights_clipped': highlights_clipped,
                'exposure_quality': max(0.0, min(1.0, exposure_score)),
                'histogram': hist.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Exposure analysis error: {e}")
            return {
                'exposure_quality': 0.5,
                'mean_brightness': 128.0,
                'brightness_std': 0.0,
                'shadows_clipped': False,
                'highlights_clipped': False
            }
    
    def _analyze_noise(self, img_array: np.ndarray) -> Dict:
        """
        Analyze noise levels in the image.
        
        Returns:
            Dictionary with noise analysis results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur and calculate difference
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_map = cv2.absdiff(gray, blurred)
            
            # Calculate noise metrics
            noise_level = np.mean(noise_map) / 255.0
            noise_std = np.std(noise_map) / 255.0
            
            # Noise quality score (lower noise is better)
            noise_quality = 1.0 - min(noise_level * 2, 1.0)
            
            return {
                'noise_level': float(noise_level),
                'noise_std': float(noise_std),
                'noise_quality': float(noise_quality),
                'is_noisy': noise_level > 0.1
            }
            
        except Exception as e:
            logger.warning(f"Noise analysis error: {e}")
            return {
                'noise_level': 0.0,
                'noise_std': 0.0,
                'noise_quality': 1.0,
                'is_noisy': False
            }
    
    def _analyze_contrast(self, img_array: np.ndarray) -> Dict:
        """
        Analyze contrast and dynamic range.
        
        Returns:
            Dictionary with contrast analysis results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate contrast metrics
            min_val = np.min(gray)
            max_val = np.max(gray)
            dynamic_range = max_val - min_val
            
            # RMS contrast
            mean_val = np.mean(gray)
            rms_contrast = np.sqrt(np.mean((gray - mean_val) ** 2))
            
            # Michelson contrast with overflow protection
            denominator = float(max_val) + float(min_val)
            if denominator > 0:
                michelson_contrast = float(max_val - min_val) / denominator
            else:
                michelson_contrast = 0.0
            
            # Contrast quality score
            contrast_quality = min(dynamic_range / 255.0, 1.0)
            contrast_quality *= min(rms_contrast / 64.0, 1.0)  # Normalize to reasonable range
            
            return {
                'dynamic_range': float(dynamic_range),
                'rms_contrast': float(rms_contrast),
                'michelson_contrast': float(michelson_contrast),
                'contrast_quality': float(contrast_quality),
                'min_value': int(min_val),
                'max_value': int(max_val)
            }
            
        except Exception as e:
            logger.warning(f"Contrast analysis error: {e}")
            return {
                'dynamic_range': 0.0,
                'rms_contrast': 0.0,
                'michelson_contrast': 0.0,
                'contrast_quality': 0.5,
                'min_value': 0,
                'max_value': 255
            }
    
    def _analyze_color(self, img_array: np.ndarray) -> Dict:
        """
        Analyze color characteristics.
        
        Returns:
            Dictionary with color analysis results
        """
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Analyze saturation
            saturation = hsv[:, :, 1]
            mean_saturation = np.mean(saturation) / 255.0
            saturation_std = np.std(saturation) / 255.0
            
            # Analyze color distribution
            r_channel = img_array[:, :, 0]
            g_channel = img_array[:, :, 1]
            b_channel = img_array[:, :, 2]
            
            # Color balance
            r_mean = np.mean(r_channel)
            g_mean = np.mean(g_channel)
            b_mean = np.mean(b_channel)
            
            # Color cast detection (simplified)
            total_mean = (r_mean + g_mean + b_mean) / 3
            color_cast = max(
                abs(r_mean - total_mean),
                abs(g_mean - total_mean),
                abs(b_mean - total_mean)
            ) / 255.0
            
            # Color quality score
            color_quality = mean_saturation * 0.5 + (1.0 - color_cast) * 0.5
            
            return {
                'mean_saturation': float(mean_saturation),
                'saturation_std': float(saturation_std),
                'color_cast': float(color_cast),
                'color_balance': {
                    'red': float(r_mean),
                    'green': float(g_mean),
                    'blue': float(b_mean)
                },
                'color_quality': float(color_quality)
            }
            
        except Exception as e:
            logger.warning(f"Color analysis error: {e}")
            return {
                'mean_saturation': 0.5,
                'saturation_std': 0.0,
                'color_cast': 0.0,
                'color_quality': 0.5,
                'color_balance': {'red': 128.0, 'green': 128.0, 'blue': 128.0}
            }
    
    def _analyze_exif(self, image: Image.Image) -> Dict:
        """
        Extract and analyze EXIF data.
        
        Returns:
            Dictionary with EXIF analysis results
        """
        try:
            exif_dict = {}
            
            # Get EXIF data
            exif = image.getexif()
            
            if exif:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = value
            
            # Extract key technical parameters
            camera_settings = {}
            
            # ISO
            if 'ISOSpeedRatings' in exif_dict:
                camera_settings['iso'] = exif_dict['ISOSpeedRatings']
            
            # Aperture
            if 'FNumber' in exif_dict:
                f_number = exif_dict['FNumber']
                if isinstance(f_number, tuple):
                    camera_settings['aperture'] = f_number[0] / f_number[1]
                else:
                    camera_settings['aperture'] = float(f_number)
            
            # Shutter speed
            if 'ExposureTime' in exif_dict:
                exposure_time = exif_dict['ExposureTime']
                if isinstance(exposure_time, tuple):
                    camera_settings['shutter_speed'] = exposure_time[0] / exposure_time[1]
                else:
                    camera_settings['shutter_speed'] = float(exposure_time)
            
            # Focal length
            if 'FocalLength' in exif_dict:
                focal_length = exif_dict['FocalLength']
                if isinstance(focal_length, tuple):
                    camera_settings['focal_length'] = focal_length[0] / focal_length[1]
                else:
                    camera_settings['focal_length'] = float(focal_length)
            
            # Camera info
            camera_info = {}
            if 'Make' in exif_dict:
                camera_info['make'] = str(exif_dict['Make'])
            if 'Model' in exif_dict:
                camera_info['model'] = str(exif_dict['Model'])
            if 'LensModel' in exif_dict:
                camera_info['lens'] = str(exif_dict['LensModel'])
            
            return {
                'exif_available': True,
                'camera_settings': camera_settings,
                'camera_info': camera_info,
                'full_exif': exif_dict
            }
            
        except Exception as e:
            logger.warning(f"EXIF analysis error: {e}")
            return {
                'exif_available': False,
                'camera_settings': {},
                'camera_info': {},
                'full_exif': {}
            }
    
    def _calculate_technical_score(self, analysis: Dict) -> float:
        """
        Calculate overall technical quality score.
        
        Args:
            analysis: Dictionary containing all analysis results
            
        Returns:
            Technical quality score (0.0-1.0)
        """
        try:
            # Weight factors for different aspects
            weights = {
                'sharpness': 0.25,
                'exposure': 0.25,
                'noise': 0.20,
                'contrast': 0.15,
                'color': 0.15
            }
            
            # Normalize sharpness score
            sharpness_score = min(analysis.get('sharpness', 0) / 500.0, 1.0)
            
            # Get other scores
            exposure_score = analysis.get('exposure_quality', 0.5)
            noise_score = analysis.get('noise_quality', 0.5)
            contrast_score = analysis.get('contrast_quality', 0.5)
            color_score = analysis.get('color_quality', 0.5)
            
            # Calculate weighted score
            technical_score = (
                sharpness_score * weights['sharpness'] +
                exposure_score * weights['exposure'] +
                noise_score * weights['noise'] +
                contrast_score * weights['contrast'] +
                color_score * weights['color']
            )
            
            return max(0.0, min(1.0, technical_score))
            
        except Exception as e:
            logger.warning(f"Technical score calculation error: {e}")
            return 0.5