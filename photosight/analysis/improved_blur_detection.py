"""
Improved blur detection methods for PhotoSight
This module contains enhanced blur detection algorithms that better handle
artistic blur (bokeh) vs actual image blur/motion blur
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImprovedSharpnessAnalyzer:
    """Enhanced sharpness analyzer with region-based and subject-aware detection"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        self.base_threshold = config.get('laplacian_variance_minimum', 80.0)
        self.use_regions = config.get('use_region_based_analysis', True)
        self.center_weight = config.get('center_region_weight', 0.7)
        self.edge_weight = config.get('edge_region_weight', 0.3)
        self.subject_priority = config.get('subject_focus_priority', True)
        self.detect_motion = config.get('motion_blur_detection', True)
        
    def analyze_sharpness_improved(self, image_array: np.ndarray, 
                                   metadata: Optional[Dict] = None,
                                   detected_subjects: Optional[List[Dict]] = None) -> Dict:
        """
        Perform improved sharpness analysis with multiple methods
        
        Args:
            image_array: RGB image array
            metadata: Image metadata (for adaptive thresholds)
            detected_subjects: List of detected subjects with bounding boxes
            
        Returns:
            Comprehensive sharpness analysis results
        """
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # 1. Traditional global analysis (for baseline)
        global_result = self._analyze_global_sharpness(gray)
        
        # 2. Region-based analysis
        region_result = self._analyze_regional_sharpness(gray) if self.use_regions else None
        
        # 3. Subject-focused analysis
        subject_result = None
        if self.subject_priority and detected_subjects:
            subject_result = self._analyze_subject_sharpness(gray, detected_subjects)
        
        # 4. Motion blur detection
        motion_result = self._detect_motion_blur(gray) if self.detect_motion else None
        
        # 5. Focus map generation
        focus_map = self._generate_focus_map(gray)
        
        # 6. Gradient-based analysis (NEW)
        gradient_result = self._analyze_gradient_sharpness(gray)
        
        # 7. Uniform blur detection (NEW)
        uniform_blur_result = self._detect_uniform_blur(gray, region_result, gradient_result)
        
        # Calculate adaptive threshold based on metadata
        threshold = self._calculate_adaptive_threshold(metadata) if metadata else self.base_threshold
        
        # Determine if image passes based on all analyses
        passed = self._determine_pass_status(
            global_result, region_result, subject_result, motion_result, 
            threshold, gradient_result, uniform_blur_result
        )
        
        # Calculate comprehensive score
        score = self._calculate_comprehensive_score(
            global_result, region_result, subject_result, motion_result, gradient_result
        )
        
        return {
            'passed': passed,
            'score': score,
            'threshold': threshold,
            'global_analysis': global_result,
            'regional_analysis': region_result,
            'subject_analysis': subject_result,
            'motion_blur_analysis': motion_result,
            'gradient_analysis': gradient_result,
            'uniform_blur': uniform_blur_result,
            'focus_map': focus_map,
            'verdict': self._generate_verdict(passed, motion_result, uniform_blur_result)
        }
    
    def _analyze_global_sharpness(self, gray_image: np.ndarray) -> Dict:
        """Traditional global Laplacian variance analysis"""
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        variance = laplacian.var()
        
        return {
            'variance': variance,
            'passed': variance >= self.base_threshold
        }
    
    def _analyze_regional_sharpness(self, gray_image: np.ndarray) -> Dict:
        """Analyze sharpness using 3x3 grid for better granularity"""
        h, w = gray_image.shape
        
        # Create 3x3 grid regions
        regions = {}
        region_scores = {}
        
        grid_names = [
            ['top_left', 'top_center', 'top_right'],
            ['mid_left', 'center', 'mid_right'], 
            ['bottom_left', 'bottom_center', 'bottom_right']
        ]
        
        for row in range(3):
            for col in range(3):
                # Calculate region boundaries
                y1 = (h * row) // 3
                y2 = (h * (row + 1)) // 3
                x1 = (w * col) // 3
                x2 = (w * (col + 1)) // 3
                
                region_name = grid_names[row][col]
                region = gray_image[y1:y2, x1:x2]
                
                # Calculate sharpness for this region
                laplacian = cv2.Laplacian(region, cv2.CV_64F)
                variance = laplacian.var()
                
                regions[region_name] = region
                region_scores[region_name] = variance
        
        # Calculate weighted score with center emphasis
        center_score = region_scores['center']
        
        # Adjacent to center regions get medium weight
        adjacent_regions = ['top_center', 'mid_left', 'mid_right', 'bottom_center']
        adjacent_scores = [region_scores[name] for name in adjacent_regions]
        avg_adjacent_score = np.mean(adjacent_scores) if adjacent_scores else 0
        
        # Corner regions get lower weight
        corner_regions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        corner_scores = [region_scores[name] for name in corner_regions]
        avg_corner_score = np.mean(corner_scores) if corner_scores else 0
        
        # Weighted average: center=50%, adjacent=35%, corners=15%
        weighted_score = (center_score * 0.5 + 
                         avg_adjacent_score * 0.35 + 
                         avg_corner_score * 0.15)
        
        # Find the sharpest region for rescue logic
        max_region_name = max(region_scores, key=region_scores.get)
        max_sharpness = region_scores[max_region_name]
        
        return {
            'region_scores': region_scores,
            'grid_layout': grid_names,
            'center_variance': center_score,
            'adjacent_avg_variance': avg_adjacent_score,
            'corner_avg_variance': avg_corner_score,
            'weighted_score': weighted_score,
            'max_region': max_region_name,
            'max_sharpness': max_sharpness,
            'passed': center_score >= self.base_threshold * 0.8  # More lenient for center
        }
    
    def _extract_center_region(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract the center region of the image"""
        h, w = gray_image.shape
        center_y1, center_y2 = int(h * 0.3), int(h * 0.7)
        center_x1, center_x2 = int(w * 0.3), int(w * 0.7)
        return gray_image[center_y1:center_y2, center_x1:center_x2]
    
    def _analyze_subject_sharpness(self, gray_image: np.ndarray, 
                                  subjects: List[Dict]) -> Dict:
        """Analyze sharpness specifically on detected subjects"""
        subject_scores = []
        subject_details = []
        
        for i, subject in enumerate(subjects):
            bbox = subject.get('bbox', subject.get('coordinates', []))
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                # Ensure bounds are valid
                x1, x2 = max(0, x1), min(gray_image.shape[1], x2)
                y1, y2 = max(0, y1), min(gray_image.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    subject_region = gray_image[y1:y2, x1:x2]
                    laplacian = cv2.Laplacian(subject_region, cv2.CV_64F)
                    variance = laplacian.var()
                    
                    subject_scores.append(variance)
                    subject_details.append({
                        'subject_id': i,
                        'bbox': [x1, y1, x2, y2],
                        'sharpness': variance,
                        'sharp': variance >= self.base_threshold
                    })
        
        if not subject_scores:
            return None
        
        # Use the sharpest subject as reference
        max_sharpness = max(subject_scores)
        avg_sharpness = np.mean(subject_scores)
        
        return {
            'max_subject_sharpness': max_sharpness,
            'avg_subject_sharpness': avg_sharpness,
            'num_subjects': len(subject_scores),
            'subject_details': subject_details,
            'passed': max_sharpness >= self.base_threshold
        }
    
    def _detect_motion_blur(self, gray_image: np.ndarray) -> Dict:
        """Detect motion blur using frequency domain analysis"""
        # Apply FFT
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Analyze directional patterns in frequency domain
        h, w = magnitude_spectrum.shape
        center = (h // 2, w // 2)
        
        # Create radial sampling lines
        angles = np.linspace(0, np.pi, 36)  # Sample every 5 degrees
        radial_profiles = []
        
        for angle in angles:
            # Sample along a line through the center
            num_samples = min(h, w) // 2
            x_samples = center[1] + np.cos(angle) * np.linspace(0, num_samples, num_samples)
            y_samples = center[0] + np.sin(angle) * np.linspace(0, num_samples, num_samples)
            
            # Ensure coordinates are within bounds
            valid_mask = (x_samples >= 0) & (x_samples < w) & (y_samples >= 0) & (y_samples < h)
            x_samples = x_samples[valid_mask].astype(int)
            y_samples = y_samples[valid_mask].astype(int)
            
            if len(x_samples) > 0:
                profile = magnitude_spectrum[y_samples, x_samples]
                radial_profiles.append(np.mean(profile))
        
        if not radial_profiles:
            return {'has_motion_blur': False, 'confidence': 0.0}
        
        # Motion blur creates elongation in specific directions
        radial_std = np.std(radial_profiles)
        radial_mean = np.mean(radial_profiles)
        
        # High variance in radial profiles indicates directional blur
        motion_blur_score = radial_std / (radial_mean + 1e-6)
        has_motion_blur = motion_blur_score > 0.3  # Threshold for motion blur
        
        # Find dominant direction if motion blur detected
        blur_angle = None
        if has_motion_blur and len(radial_profiles) > 0:
            max_idx = np.argmax(radial_profiles)
            blur_angle = angles[max_idx] * 180 / np.pi
        
        return {
            'has_motion_blur': has_motion_blur,
            'motion_blur_score': motion_blur_score,
            'blur_angle': blur_angle,
            'confidence': min(motion_blur_score / 0.5, 1.0)  # Normalize confidence
        }
    
    def _generate_focus_map(self, gray_image: np.ndarray, window_size: int = 32) -> np.ndarray:
        """Generate a focus/sharpness heat map of the image"""
        h, w = gray_image.shape
        
        # Reduce computation by using stride
        stride = window_size // 2
        
        # Initialize focus map
        map_h = (h - window_size) // stride + 1
        map_w = (w - window_size) // stride + 1
        focus_map = np.zeros((map_h, map_w))
        
        for i in range(0, h - window_size, stride):
            for j in range(0, w - window_size, stride):
                window = gray_image[i:i+window_size, j:j+window_size]
                
                # Use gradient magnitude for better edge detection
                grad_x = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # Use standard deviation of gradient as sharpness measure
                focus_map[i//stride, j//stride] = gradient_magnitude.std()
        
        # Resize back to original dimensions
        focus_map_resized = cv2.resize(focus_map, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to 0-1 range
        if focus_map_resized.max() > 0:
            focus_map_resized = focus_map_resized / focus_map_resized.max()
        
        return focus_map_resized
    
    def _calculate_adaptive_threshold(self, metadata: Dict) -> float:
        """Calculate adaptive threshold based on shooting conditions"""
        threshold = self.base_threshold
        
        # Adjust for focal length (telephoto is harder to keep sharp)
        focal_length_str = metadata.get('focal_length', '50mm')
        try:
            focal_length = float(focal_length_str.replace('mm', ''))
            if focal_length > 200:
                threshold *= 0.85  # More lenient for telephoto
            elif focal_length < 35:
                threshold *= 1.1   # Stricter for wide angle
        except:
            pass
        
        # Adjust for aperture (wide apertures have shallower DOF)
        aperture = metadata.get('aperture')
        if aperture:
            try:
                f_number = float(aperture)
                if f_number < 2.8:
                    threshold *= 0.9  # More lenient for shallow DOF
                elif f_number > 8:
                    threshold *= 1.1  # Stricter for deep DOF
            except:
                pass
        
        # Adjust for ISO (high ISO images are noisier)
        iso = metadata.get('iso')
        if iso and iso > 6400:
            threshold *= 0.95  # Slightly more lenient for high ISO
        
        return threshold
    
    def _analyze_gradient_sharpness(self, gray_image: np.ndarray) -> Dict:
        """Analyze image sharpness using gradient information"""
        # Calculate gradients
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Calculate statistics
        gradient_variance = gradient_magnitude.var()
        gradient_mean = gradient_magnitude.mean()
        gradient_max = gradient_magnitude.max()
        
        # Calculate focus peak ratio
        # High gradient areas indicate sharp edges
        high_gradient_threshold = np.percentile(gradient_magnitude, 90)
        focus_peaks = np.sum(gradient_magnitude > high_gradient_threshold)
        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        focus_peak_ratio = focus_peaks / total_pixels
        
        # Determine if gradient analysis passes
        # Photos with very low gradient variance are uniformly blurry
        gradient_passed = gradient_variance > 100 and focus_peak_ratio > 0.001
        
        return {
            'variance': gradient_variance,
            'mean': gradient_mean,
            'max': gradient_max,
            'focus_peak_ratio': focus_peak_ratio,
            'passed': gradient_passed
        }
    
    def _detect_uniform_blur(self, gray_image: np.ndarray, 
                           region_result: Optional[Dict],
                           gradient_result: Optional[Dict]) -> Dict:
        """Detect uniform blur (camera shake or overall focus miss)"""
        h, w = gray_image.shape
        
        # Calculate local variance map
        window_size = 64
        stride = 32
        local_variances = []
        
        for y in range(0, h - window_size, stride):
            for x in range(0, w - window_size, stride):
                window = gray_image[y:y+window_size, x:x+window_size]
                window_lap = cv2.Laplacian(window, cv2.CV_64F)
                local_variances.append(window_lap.var())
        
        if not local_variances:
            return {'has_uniform_blur': False, 'confidence': 0.0}
        
        # Calculate uniformity metrics
        local_var_std = np.std(local_variances)
        local_var_mean = np.mean(local_variances)
        variance_uniformity = local_var_std / (local_var_mean + 1e-6)
        
        # Check for uniform blur indicators
        has_uniform_blur = False
        confidence = 0.0
        
        # Low variance uniformity indicates consistent blur across image
        if variance_uniformity < 0.3:
            has_uniform_blur = True
            confidence += 0.4
        
        # Similar center and edge blur (from region analysis)
        if region_result:
            center_var = region_result.get('center_variance', 0)
            edge_avg = region_result.get('edge_avg_variance', 0)
            if edge_avg > 0:
                blur_ratio = center_var / edge_avg
                if 0.8 <= blur_ratio <= 1.2:
                    has_uniform_blur = True
                    confidence += 0.3
        
        # Low gradient variance indicates lack of sharp edges anywhere
        if gradient_result and gradient_result.get('variance', float('inf')) < 100:
            has_uniform_blur = True
            confidence += 0.3
        
        confidence = min(confidence, 1.0)
        
        return {
            'has_uniform_blur': has_uniform_blur,
            'variance_uniformity': variance_uniformity,
            'confidence': confidence,
            'local_variance_mean': local_var_mean,
            'local_variance_std': local_var_std
        }
    
    def _determine_pass_status(self, global_result: Dict, 
                              region_result: Optional[Dict],
                              subject_result: Optional[Dict], 
                              motion_result: Optional[Dict],
                              threshold: float,
                              gradient_result: Optional[Dict] = None,
                              uniform_blur_result: Optional[Dict] = None) -> bool:
        """Determine if image passes based on all analyses"""
        
        # If motion blur is detected with high confidence, fail
        if motion_result and motion_result.get('has_motion_blur') and motion_result.get('confidence', 0) > 0.7:
            return False
        
        # Check for uniform blur with high confidence
        if uniform_blur_result and uniform_blur_result.get('has_uniform_blur') and uniform_blur_result.get('confidence', 0) > 0.6:
            return False
        
        # NEW: Check for any region with exceptional sharpness (selective focus)
        if region_result and region_result.get('region_scores'):
            region_scores = region_result['region_scores']
            max_region_sharpness = max(region_scores.values())
            
            # If ANY region is exceptionally sharp, accept the photo
            # This handles cases like wine glass being tack sharp while background is soft
            high_sharpness_threshold = max(threshold * 0.4, 25.0)  # More lenient threshold
            if max_region_sharpness >= high_sharpness_threshold:
                logger.info(f"Sharp region detected: max={max_region_sharpness:.1f}, threshold={high_sharpness_threshold:.1f}")
                return True
        
        # Check gradient analysis - fail if gradients are too low
        if gradient_result and not gradient_result.get('passed', True):
            # Only fail on gradient if no subjects are detected or subjects aren't sharp
            if not (subject_result and subject_result.get('passed')):
                return False
        
        # If subjects are detected and sharp, pass regardless of background
        if subject_result and subject_result.get('passed'):
            return True
        
        # If center region is sharp (for images without detected subjects)
        if region_result and region_result.get('passed'):
            return True
        
        # Fall back to global analysis
        return global_result.get('passed', False)
    
    def _calculate_comprehensive_score(self, global_result: Dict,
                                     region_result: Optional[Dict],
                                     subject_result: Optional[Dict],
                                     motion_result: Optional[Dict],
                                     gradient_result: Optional[Dict] = None) -> float:
        """Calculate a comprehensive sharpness score (0-100)"""
        scores = []
        weights = []
        
        # Global score (normalized)
        if global_result:
            global_score = min(global_result['variance'] / (self.base_threshold * 2), 1.0) * 100
            scores.append(global_score)
            weights.append(0.3)
        
        # Regional score
        if region_result:
            regional_score = min(region_result['center_variance'] / (self.base_threshold * 2), 1.0) * 100
            scores.append(regional_score)
            weights.append(0.4)
        
        # Subject score (highest priority)
        if subject_result:
            subject_score = min(subject_result['max_subject_sharpness'] / (self.base_threshold * 2), 1.0) * 100
            scores.append(subject_score)
            weights.append(0.5)
        
        # Motion blur penalty
        if motion_result and motion_result.get('has_motion_blur'):
            motion_penalty = motion_result.get('confidence', 0) * 30
            scores.append(100 - motion_penalty)
            weights.append(0.3)
        
        # Gradient score
        if gradient_result:
            # Normalize gradient variance (typical range 0-500)
            gradient_score = min(gradient_result['variance'] / 500, 1.0) * 100
            scores.append(gradient_score)
            weights.append(0.2)
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weights[:len(scores)])
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            return max(0, min(100, weighted_score))
        
        return 0
    
    def _generate_verdict(self, passed: bool, motion_result: Optional[Dict],
                         uniform_blur_result: Optional[Dict] = None) -> str:
        """Generate a descriptive verdict"""
        if not passed:
            if motion_result and motion_result.get('has_motion_blur'):
                return 'motion_blur_detected'
            if uniform_blur_result and uniform_blur_result.get('has_uniform_blur'):
                return 'uniform_blur_detected'
            return 'insufficient_sharpness'
        return 'sharp'


def integrate_with_technical_analyzer(technical_analyzer_instance, improved_analyzer: ImprovedSharpnessAnalyzer):
    """
    Monkey-patch or extend the existing TechnicalAnalyzer to use improved sharpness detection
    
    This is a temporary integration method. In production, you would properly
    refactor the TechnicalAnalyzer class.
    """
    original_analyze_sharpness = technical_analyzer_instance.analyze_sharpness
    
    def new_analyze_sharpness(image_array: np.ndarray, metadata: Optional[Dict] = None, 
                             detected_subjects: Optional[List[Dict]] = None) -> Dict:
        # Use improved analysis
        improved_result = improved_analyzer.analyze_sharpness_improved(
            image_array, metadata, detected_subjects
        )
        
        # Format result to match expected structure
        return {
            'passed': improved_result['passed'],
            'score': improved_result['score'],
            'threshold': improved_result['threshold'],
            'sharpness_map': improved_result.get('focus_map'),
            'verdict': improved_result['verdict'],
            # Include additional details for debugging
            'improved_analysis': improved_result
        }
    
    # Replace the method
    technical_analyzer_instance.analyze_sharpness = new_analyze_sharpness