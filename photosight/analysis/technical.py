"""
Technical image analysis for PhotoSight Stage 1 culling
Performs objective quality checks: sharpness, exposure, metadata filtering
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from .improved_blur_detection import ImprovedSharpnessAnalyzer, integrate_with_technical_analyzer

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Performs technical quality analysis on images"""
    
    def __init__(self, config: Dict):
        """
        Initialize analyzer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['technical_filters']
        self.sharpness_threshold = self.config['sharpness']['laplacian_variance_minimum']
        self.black_clip_threshold = self.config['exposure']['histogram_black_clip_threshold']
        self.white_clip_threshold = self.config['exposure']['histogram_white_clip_threshold']
        self.min_mean_brightness = self.config['exposure'].get('minimum_mean_brightness', 50)
        self.max_mean_brightness = self.config['exposure'].get('maximum_mean_brightness', 205)
        self.max_shadow_percentage = self.config['exposure'].get('maximum_shadow_percentage', 0.7)
        self.min_highlight_percentage = self.config['exposure'].get('minimum_highlight_percentage', 0.01)
        self.max_iso = self.config['metadata']['maximum_iso']
        self.min_shutter_denominator = self.config['metadata']['minimum_shutter_speed_denominator']
        self.apply_focal_rule = self.config['metadata']['apply_focal_length_rule']
        self.focal_rule_multiplier = self.config['metadata']['focal_length_rule_multiplier']
        
    def analyze_image(self, image_array: np.ndarray, metadata: Dict) -> Dict:
        """
        Perform complete technical analysis on an image
        
        Args:
            image_array: RGB image array
            metadata: Image metadata dictionary
            
        Returns:
            Analysis results dictionary
        """
        results = {
            'passed': True,
            'rejection_reasons': [],
            'scores': {},
            'details': {}
        }
        
        # Analyze sharpness
        sharpness_result = self.analyze_sharpness(image_array)
        results['scores']['sharpness'] = sharpness_result['score']
        results['details']['sharpness'] = sharpness_result
        
        if not sharpness_result['passed']:
            results['passed'] = False
            results['rejection_reasons'].append('blurry')
            
        # Analyze exposure
        exposure_result = self.analyze_exposure(image_array)
        results['scores']['exposure'] = exposure_result['score']
        results['details']['exposure'] = exposure_result
        
        if exposure_result['underexposed']:
            results['passed'] = False
            results['rejection_reasons'].append('underexposed')
        if exposure_result['overexposed']:
            results['passed'] = False
            results['rejection_reasons'].append('overexposed')
            
        # Analyze metadata
        metadata_result = self.analyze_metadata(metadata)
        results['scores']['metadata'] = metadata_result['score']
        results['details']['metadata'] = metadata_result
        
        if not metadata_result['passed']:
            results['passed'] = False
            results['rejection_reasons'].extend(metadata_result['rejection_reasons'])
            
        # Calculate overall technical score (0-100)
        results['overall_score'] = self._calculate_overall_score(results['scores'])
        
        return results
    
    def analyze_sharpness(self, image_array: np.ndarray) -> Dict:
        """
        Analyze image sharpness using Laplacian variance
        
        Args:
            image_array: RGB image array
            
        Returns:
            Sharpness analysis results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance
        variance = laplacian.var()
        
        # Generate sharpness map if debugging
        sharpness_map = None
        if self.config['sharpness'].get('save_debug_heatmaps', False):
            # Calculate local sharpness using sliding window
            sharpness_map = self._generate_sharpness_map(gray)
            
        passed = variance >= self.sharpness_threshold
        
        return {
            'passed': passed,
            'score': variance,
            'threshold': self.sharpness_threshold,
            'sharpness_map': sharpness_map,
            'verdict': 'sharp' if passed else 'blurry'
        }
    
    def _generate_sharpness_map(self, gray_image: np.ndarray, 
                               window_size: int = 50) -> np.ndarray:
        """
        Generate a sharpness heatmap for the image
        
        Args:
            gray_image: Grayscale image
            window_size: Size of sliding window
            
        Returns:
            Sharpness heatmap
        """
        h, w = gray_image.shape
        step = window_size // 2
        sharpness_map = np.zeros((h // step, w // step))
        
        for i in range(0, h - window_size, step):
            for j in range(0, w - window_size, step):
                window = gray_image[i:i+window_size, j:j+window_size]
                laplacian = cv2.Laplacian(window, cv2.CV_64F)
                sharpness_map[i//step, j//step] = laplacian.var()
                
        # Resize to original size
        sharpness_map = cv2.resize(sharpness_map, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return sharpness_map
    
    def analyze_exposure(self, image_array: np.ndarray) -> Dict:
        """
        Analyze image exposure using histogram analysis
        
        Args:
            image_array: RGB image array
            
        Returns:
            Exposure analysis results
        """
        # Calculate histograms
        histograms = {}
        total_pixels = image_array.shape[0] * image_array.shape[1]
        
        # Analyze each channel
        clipping_info = {}
        for i, channel in enumerate(['red', 'green', 'blue']):
            hist, _ = np.histogram(image_array[:, :, i], bins=256, range=(0, 256))
            histograms[channel] = hist
            
            # Check clipping
            black_pixels = hist[0] / total_pixels
            white_pixels = hist[255] / total_pixels
            
            clipping_info[channel] = {
                'black_clipping': black_pixels,
                'white_clipping': white_pixels,
                'black_clipped': black_pixels > self.black_clip_threshold,
                'white_clipped': white_pixels > self.white_clip_threshold
            }
        
        # Calculate luminance histogram
        luminance = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        lum_hist, _ = np.histogram(luminance, bins=256, range=(0, 256))
        histograms['luminance'] = lum_hist
        
        # Calculate mean brightness
        mean_brightness = np.mean(luminance)
        
        # Calculate brightness distribution
        shadows = np.sum(lum_hist[0:64]) / total_pixels  # 0-25% brightness
        midtones = np.sum(lum_hist[64:192]) / total_pixels  # 25-75% brightness
        highlights = np.sum(lum_hist[192:256]) / total_pixels  # 75-100% brightness
        
        # Check luminance clipping
        lum_black_pixels = lum_hist[0] / total_pixels
        lum_white_pixels = lum_hist[255] / total_pixels
        
        # Enhanced exposure detection
        # 1. Check for clipping (original method)
        clipping_underexposed = lum_black_pixels > self.black_clip_threshold
        clipping_overexposed = lum_white_pixels > self.white_clip_threshold
        
        # 2. Check for overall brightness issues
        brightness_underexposed = mean_brightness < self.min_mean_brightness
        brightness_overexposed = mean_brightness > self.max_mean_brightness
        
        # 3. Check for distribution issues
        distribution_underexposed = shadows > self.max_shadow_percentage
        distribution_overexposed = highlights < self.min_highlight_percentage and mean_brightness > 150
        
        # Combine all checks
        underexposed = clipping_underexposed or brightness_underexposed or distribution_underexposed
        overexposed = clipping_overexposed or brightness_overexposed or distribution_overexposed
        
        # Calculate exposure score (0-100)
        score = 100.0
        
        # Penalize for clipping
        if clipping_underexposed:
            score -= min(lum_black_pixels / self.black_clip_threshold, 1.0) * 30
        if clipping_overexposed:
            score -= min(lum_white_pixels / self.white_clip_threshold, 1.0) * 30
            
        # Penalize for brightness issues
        if brightness_underexposed:
            brightness_penalty = (self.min_mean_brightness - mean_brightness) / self.min_mean_brightness * 20
            score -= min(brightness_penalty, 20)
        elif brightness_overexposed:
            brightness_penalty = (mean_brightness - self.max_mean_brightness) / (255 - self.max_mean_brightness) * 20
            score -= min(brightness_penalty, 20)
            
        # Penalize for distribution issues
        if distribution_underexposed:
            distribution_penalty = (shadows - self.max_shadow_percentage) / (1 - self.max_shadow_percentage) * 20
            score -= min(distribution_penalty, 20)
            
        exposure_score = max(score, 0)
        
        return {
            'passed': not (underexposed or overexposed),
            'score': exposure_score,
            'underexposed': underexposed,
            'overexposed': overexposed,
            'histograms': histograms,
            'clipping_info': clipping_info,
            'luminance_black_clipping': lum_black_pixels,
            'luminance_white_clipping': lum_white_pixels,
            'mean_brightness': mean_brightness,
            'shadows': shadows,
            'midtones': midtones,
            'highlights': highlights,
            'underexposure_reasons': {
                'clipping': clipping_underexposed,
                'brightness': brightness_underexposed,
                'distribution': distribution_underexposed
            },
            'overexposure_reasons': {
                'clipping': clipping_overexposed,
                'brightness': brightness_overexposed,
                'distribution': distribution_overexposed
            }
        }
    
    def analyze_metadata(self, metadata: Dict) -> Dict:
        """
        Analyze image metadata for technical issues
        
        Args:
            metadata: Metadata dictionary from RawProcessor
            
        Returns:
            Metadata analysis results
        """
        passed = True
        rejection_reasons = []
        details = {}
        score = 100.0
        
        # Check ISO
        iso = metadata.get('iso')
        if iso and iso > self.max_iso:
            passed = False
            rejection_reasons.append('high_iso')
            details['iso_verdict'] = f'Too high ({iso} > {self.max_iso})'
            # Penalize score based on how much over the limit
            iso_penalty = min(((iso - self.max_iso) / self.max_iso) * 30, 30)
            score -= iso_penalty
        else:
            details['iso_verdict'] = f'Acceptable ({iso})'
            
        # Check shutter speed
        shutter_speed = metadata.get('shutter_speed_numeric')
        if shutter_speed:
            # Check absolute minimum
            if shutter_speed > (1.0 / self.min_shutter_denominator):
                passed = False
                rejection_reasons.append('slow_shutter')
                details['shutter_verdict'] = f'Too slow ({shutter_speed:.4f}s)'
                score -= 20
            else:
                details['shutter_verdict'] = f'Acceptable ({shutter_speed:.4f}s)'
                
            # Apply focal length rule if configured
            if self.apply_focal_rule and metadata.get('focal_length'):
                focal_length = float(str(metadata['focal_length']).replace('mm', ''))
                # Use 35mm equivalent if available
                if metadata.get('focal_length_35mm'):
                    focal_length = float(str(metadata['focal_length_35mm']).replace('mm', ''))
                    
                min_shutter_focal = 1.0 / (focal_length * self.focal_rule_multiplier)
                if shutter_speed > min_shutter_focal:
                    if 'slow_shutter' not in rejection_reasons:
                        passed = False
                        rejection_reasons.append('slow_shutter')
                        score -= 15
                    details['focal_rule_verdict'] = f'Failed 1/{focal_length} rule'
                else:
                    details['focal_rule_verdict'] = 'Passed focal length rule'
                    
        return {
            'passed': passed,
            'score': max(score, 0),
            'rejection_reasons': rejection_reasons,
            'details': details,
            'iso': iso,
            'shutter_speed': shutter_speed,
            'metadata': metadata
        }
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted overall technical score
        
        Args:
            scores: Dictionary of individual scores
            
        Returns:
            Overall score (0-100)
        """
        # Define weights
        weights = {
            'sharpness': 0.4,    # Most important for technical quality
            'exposure': 0.35,    # Very important
            'metadata': 0.25     # Important but can be compensated
        }
        
        # Normalize sharpness score (assuming threshold is minimum acceptable)
        if 'sharpness' in scores:
            # Scale sharpness to 0-100 range
            sharpness_normalized = min(scores['sharpness'] / (self.sharpness_threshold * 2), 1.0) * 100
            scores['sharpness'] = sharpness_normalized
            
        # Calculate weighted score
        total_score = 0
        total_weight = 0
        
        for key, weight in weights.items():
            if key in scores:
                total_score += scores[key] * weight
                total_weight += weight
                
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0