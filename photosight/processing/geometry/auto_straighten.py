"""
Automatic image straightening based on detected reference lines
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import logging
from .horizon_detector import HorizonDetector
from ..scene_classifier import SceneClassifier

logger = logging.getLogger(__name__)


class AutoStraightener:
    """
    Automatically straighten images based on horizon or architectural lines
    with scene-aware detection prioritization
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 max_rotation: float = 10.0,
                 edge_fill_method: str = 'reflect'):
        """
        Initialize auto straightener
        
        Args:
            confidence_threshold: Minimum confidence to apply auto-straightening
            max_rotation: Maximum rotation angle to apply (degrees)
            edge_fill_method: Method to fill edges after rotation ('reflect', 'replicate', 'constant')
        """
        self.confidence_threshold = confidence_threshold
        self.max_rotation = max_rotation
        self.edge_fill_method = edge_fill_method
        self.horizon_detector = HorizonDetector()
        self.scene_classifier = SceneClassifier()
        
    def analyze_image(self, image: np.ndarray) -> Dict:
        """
        Analyze image and suggest straightening angle with scene-aware detection
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary with analysis results and suggested rotation
        """
        results = {
            'needs_straightening': False,
            'suggested_angle': 0.0,
            'confidence': 0.0,
            'detection_method': None,
            'detection_details': {},
            'scene_info': {}
        }
        
        # Step 1: Classify scene to prioritize detection methods
        scene_result = self.scene_classifier.classify_scene(image)
        processing_hints = self.scene_classifier.get_processing_hints(scene_result)
        
        results['scene_info'] = {
            'classification': scene_result['classification'],
            'confidence': scene_result['confidence'],
            'processing_hints': processing_hints
        }
        
        # Step 2: Apply scene-aware confidence adjustments
        scene_type = scene_result['classification']
        base_threshold = self.confidence_threshold
        
        # Apply confidence boost for indoor architectural detection
        if scene_type == 'indoor' and processing_hints['leveling'].get('confidence_boost', 0) > 0:
            adjusted_threshold = base_threshold - processing_hints['leveling']['confidence_boost']
        else:
            adjusted_threshold = base_threshold
        
        # Step 3: Prioritize detection methods based on scene
        priority_methods = processing_hints['leveling']['priority_methods']
        avoid_methods = processing_hints['leveling']['avoid_methods']
        prefer_architectural = processing_hints['leveling'].get('prefer_architectural', False)
        
        # Method 1: Try priority methods first
        if 'vertical_reference' in priority_methods or prefer_architectural:
            # For indoor scenes, start with architectural detection
            horizon_result = self.horizon_detector.detect_horizon(image)
            
            # Check if we got vertical reference detection with good confidence
            if (horizon_result['horizon_detected'] and 
                horizon_result['method'] == 'vertical_reference' and
                horizon_result['confidence'] > adjusted_threshold):
                
                angle = horizon_result['angle']
                if abs(angle) <= self.max_rotation:
                    results.update({
                        'needs_straightening': abs(angle) > 0.5,
                        'suggested_angle': -angle,
                        'confidence': horizon_result['confidence'],
                        'detection_method': horizon_result['method'],
                        'detection_details': horizon_result
                    })
                    return results
        
        # Method 2: Standard horizon detection (if not avoided)
        if 'horizon_line' not in avoid_methods:
            horizon_result = self.horizon_detector.detect_horizon(image)
            
            if (horizon_result['horizon_detected'] and 
                horizon_result['confidence'] > adjusted_threshold):
                
                angle = horizon_result['angle']
                if abs(angle) <= self.max_rotation:
                    results.update({
                        'needs_straightening': abs(angle) > 0.5,
                        'suggested_angle': -angle,
                        'confidence': horizon_result['confidence'],
                        'detection_method': horizon_result['method'],
                        'detection_details': horizon_result
                    })
                    return results
        
        # Method 3: Grid line detection (especially for indoor architectural scenes)
        if 'grid_lines' in priority_methods or prefer_architectural:
            grid_result = self.horizon_detector.detect_grid_lines(image)
            
            grid_threshold = adjusted_threshold * 0.9  # Slightly lower for grid detection
            if grid_result['grid_detected'] and grid_result['confidence'] > grid_threshold:
                angle = grid_result['dominant_angle']
                
                if abs(angle) <= self.max_rotation:
                    results.update({
                        'needs_straightening': abs(angle) > 0.5,
                        'suggested_angle': -angle,
                        'confidence': grid_result['confidence'],
                        'detection_method': 'grid_lines',
                        'detection_details': grid_result
                    })
                    return results
        
        # Method 4: Fallback to subtle tilt detection
        subtle_angle = self._detect_subtle_tilt(image)
        if subtle_angle is not None and abs(subtle_angle) > 1.0:
            results.update({
                'needs_straightening': True,
                'suggested_angle': -subtle_angle,
                'confidence': 0.5,
                'detection_method': 'image_moments',
                'detection_details': {'subtle_angle': subtle_angle}
            })
        
        return results
    
    def apply_straightening(self, 
                          image: np.ndarray, 
                          angle: float,
                          crop_to_valid: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Apply rotation to straighten image
        
        Args:
            image: RGB image array
            angle: Rotation angle in degrees
            crop_to_valid: Whether to crop out invalid edge regions
            
        Returns:
            Tuple of (straightened image, transformation info)
        """
        if angle == 0:
            return image, {'applied_angle': 0, 'cropped': False}
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image bounds
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for new image size
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        # Apply rotation with specified edge fill method
        border_mode = {
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE,
            'constant': cv2.BORDER_CONSTANT
        }.get(self.edge_fill_method, cv2.BORDER_REFLECT)
        
        rotated = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (new_w, new_h),
            borderMode=border_mode,
            borderValue=(128, 128, 128) if border_mode == cv2.BORDER_CONSTANT else None
        )
        
        transform_info = {
            'applied_angle': angle,
            'original_size': (w, h),
            'rotated_size': (new_w, new_h),
            'rotation_matrix': rotation_matrix.tolist()
        }
        
        # Optionally crop to largest valid rectangle
        if crop_to_valid and abs(angle) > 0.1:
            cropped, crop_bounds = self._get_largest_valid_rectangle(rotated, angle, (w, h))
            transform_info['cropped'] = True
            transform_info['crop_bounds'] = crop_bounds
            return cropped, transform_info
        
        transform_info['cropped'] = False
        return rotated, transform_info
    
    def _detect_subtle_tilt(self, image: np.ndarray) -> Optional[float]:
        """
        Detect subtle tilts using image moments and symmetry
        
        Args:
            image: RGB image array
            
        Returns:
            Detected tilt angle or None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate moments
        moments = cv2.moments(edges)
        
        if moments['m00'] == 0:
            return None
        
        # Calculate centroid
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        
        # Calculate orientation using second-order moments
        mu20 = moments['mu20'] / moments['m00']
        mu02 = moments['mu02'] / moments['m00']
        mu11 = moments['mu11'] / moments['m00']
        
        # Calculate angle
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        angle_degrees = np.degrees(angle)
        
        # Only return if angle is significant but not too large
        if 1.0 < abs(angle_degrees) < 5.0:
            return angle_degrees
        
        return None
    
    def _get_largest_valid_rectangle(self, 
                                   rotated_image: np.ndarray, 
                                   angle: float,
                                   original_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Calculate and extract the largest valid rectangle after rotation
        
        Args:
            rotated_image: Rotated image
            angle: Rotation angle that was applied
            original_size: Original image dimensions (w, h)
            
        Returns:
            Tuple of (cropped image, crop bounds)
        """
        h, w = rotated_image.shape[:2]
        orig_w, orig_h = original_size
        
        # Convert angle to radians
        angle_rad = np.radians(abs(angle))
        
        # Calculate the largest rectangle that fits
        if angle_rad == 0:
            return rotated_image, (0, 0, w, h)
        
        # Use the formula for largest inscribed rectangle
        sin_a = np.sin(angle_rad)
        cos_a = np.cos(angle_rad)
        
        if orig_w <= orig_h:
            if 2 * sin_a * cos_a <= (orig_h / orig_w) - 1:
                x = orig_w * cos_a
                y = orig_w * sin_a
            else:
                x = (orig_h - orig_w * sin_a) / cos_a
                y = (orig_h - orig_w * cos_a) / sin_a
        else:
            if 2 * sin_a * cos_a <= (orig_w / orig_h) - 1:
                x = orig_h * sin_a
                y = orig_h * cos_a
            else:
                x = (orig_w - orig_h * cos_a) / sin_a
                y = (orig_w - orig_h * sin_a) / cos_a
        
        # Calculate crop bounds
        cx, cy = w // 2, h // 2
        crop_w, crop_h = int(x), int(y)
        
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        cropped = rotated_image[y1:y2, x1:x2]
        
        return cropped, (x1, y1, x2 - x1, y2 - y1)