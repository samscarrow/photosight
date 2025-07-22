"""
Scene classification for intelligent RAW processing
Determines indoor vs outdoor scenes to optimize leveling and color grading
"""

import numpy as np
import cv2
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SceneClassifier:
    """Classify scenes as indoor/outdoor for scene-aware processing"""
    
    def __init__(self, 
                 sky_threshold: float = 0.1,
                 color_temp_threshold: float = 1.1,
                 brightness_std_threshold: float = 50,
                 edge_density_threshold: float = 0.005):
        """
        Initialize scene classifier
        
        Args:
            sky_threshold: Minimum blue area ratio in upper third to detect sky
            color_temp_threshold: B/R ratio threshold for cool outdoor lighting
            brightness_std_threshold: Brightness variation threshold for dynamic range
            edge_density_threshold: Edge density threshold for geometric indoor scenes
        """
        self.sky_threshold = sky_threshold
        self.color_temp_threshold = color_temp_threshold
        self.brightness_std_threshold = brightness_std_threshold
        self.edge_density_threshold = edge_density_threshold
        
    def classify_scene(self, image: np.ndarray) -> Dict:
        """
        Classify scene as indoor or outdoor
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary with classification results
        """
        h, w = image.shape[:2]
        
        # 1. Sky detection - look for blue areas in upper portion
        sky_score = self._detect_sky(image)
        
        # 2. Color temperature analysis - outdoor tends to be cooler
        color_temp_score = self._analyze_color_temperature(image)
        
        # 3. Edge analysis - indoor scenes have more geometric structures
        edge_score = self._analyze_edges(image)
        
        # 4. Dynamic range analysis - outdoor typically has higher contrast
        dynamic_range_score = self._analyze_dynamic_range(image)
        
        # 5. Lighting analysis - indoor has more uniform, artificial lighting
        lighting_score = self._analyze_lighting(image)
        
        # Weighted scoring for final classification
        outdoor_indicators = {
            'sky_detected': sky_score > self.sky_threshold,
            'cool_color_temp': color_temp_score > self.color_temp_threshold,
            'high_dynamic_range': dynamic_range_score > self.brightness_std_threshold,
            'natural_lighting': lighting_score > 0.5
        }
        
        indoor_indicators = {
            'geometric_edges': edge_score > self.edge_density_threshold,
            'warm_color_temp': color_temp_score < 0.9,
            'uniform_lighting': lighting_score < 0.3,
            'low_dynamic_range': dynamic_range_score < 30
        }
        
        # Calculate scores
        outdoor_score = sum([
            sky_score * 3.0,  # Strong indicator
            (color_temp_score > self.color_temp_threshold) * 1.5,
            (dynamic_range_score > self.brightness_std_threshold) * 1.0,
            lighting_score * 1.0
        ])
        
        indoor_score = sum([
            (edge_score > self.edge_density_threshold) * 2.0,
            (color_temp_score < 0.9) * 1.5,
            (lighting_score < 0.3) * 1.0,
            (dynamic_range_score < 30) * 1.0
        ])
        
        # Final classification
        is_outdoor = outdoor_score > indoor_score and outdoor_score > 2.0
        confidence = abs(outdoor_score - indoor_score) / max(outdoor_score + indoor_score, 1.0)
        
        classification = 'outdoor' if is_outdoor else 'indoor'
        
        return {
            'classification': classification,
            'confidence': confidence,
            'outdoor_score': outdoor_score,
            'indoor_score': indoor_score,
            'indicators': {
                'outdoor': outdoor_indicators,
                'indoor': indoor_indicators
            },
            'metrics': {
                'sky_score': sky_score,
                'color_temp_ratio': color_temp_score,
                'edge_density': edge_score,
                'brightness_std': dynamic_range_score,
                'lighting_uniformity': lighting_score
            }
        }
    
    def _detect_sky(self, image: np.ndarray) -> float:
        """Detect sky areas in upper portion of image"""
        h, w = image.shape[:2]
        upper_third = image[:h//3, :]
        
        # Convert to HSV for better blue detection
        hsv = cv2.cvtColor(upper_third, cv2.COLOR_RGB2HSV)
        
        # Define blue sky range in HSV
        lower_blue = np.array([100, 50, 50])   # Darker blue
        upper_blue = np.array([130, 255, 255]) # Lighter blue
        
        # Create mask for blue areas
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / (upper_third.shape[0] * upper_third.shape[1])
        
        return blue_ratio
    
    def _analyze_color_temperature(self, image: np.ndarray) -> float:
        """Analyze overall color temperature (B/R ratio)"""
        r_mean = np.mean(image[:, :, 0])
        b_mean = np.mean(image[:, :, 2])
        
        # Avoid division by zero
        if r_mean < 1:
            r_mean = 1
            
        color_temp_ratio = b_mean / r_mean
        return color_temp_ratio
    
    def _analyze_edges(self, image: np.ndarray) -> float:
        """Analyze edge density for geometric structures"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        total_pixels = gray.shape[0] * gray.shape[1]
        edge_density = np.sum(edges > 0) / total_pixels
        
        return edge_density
    
    def _analyze_dynamic_range(self, image: np.ndarray) -> float:
        """Analyze brightness variation"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        brightness_std = np.std(gray)
        
        return brightness_std
    
    def _analyze_lighting(self, image: np.ndarray) -> float:
        """Analyze lighting uniformity and characteristics"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Divide into 9 regions and analyze brightness variation
        regions = []
        for i in range(3):
            for j in range(3):
                y1, y2 = i * h // 3, (i + 1) * h // 3
                x1, x2 = j * w // 3, (j + 1) * w // 3
                region_mean = np.mean(gray[y1:y2, x1:x2])
                regions.append(region_mean)
        
        # Calculate coefficient of variation
        region_std = np.std(regions)
        region_mean = np.mean(regions)
        
        if region_mean > 0:
            lighting_uniformity = 1.0 - (region_std / region_mean)
        else:
            lighting_uniformity = 0.5
        
        return max(0, min(1, lighting_uniformity))
    
    def get_processing_hints(self, classification_result: Dict) -> Dict:
        """
        Get processing hints based on scene classification
        
        Args:
            classification_result: Result from classify_scene()
            
        Returns:
            Dictionary with processing recommendations
        """
        scene_type = classification_result['classification']
        confidence = classification_result['confidence']
        
        if scene_type == 'indoor':
            return {
                'leveling': {
                    'priority_methods': ['vertical_reference', 'grid_lines'],
                    'avoid_methods': ['horizon_line'],
                    'confidence_boost': 0.1,  # Boost architectural detection confidence
                    'prefer_architectural': True
                },
                'color_grading': {
                    'white_balance_hint': 'tungsten_fluorescent',
                    'temperature_range': (2700, 4000),  # Warmer indoor lighting
                    'shadow_priority': 'high',  # Indoor scenes often need shadow lift
                    'contrast_boost': 0.15,  # Combat flat indoor lighting
                    'skin_tone_protection': True,  # Important for indoor portraits
                    'vibrance_modifier': 0.0  # Neutral vibrance for indoor
                },
                'scene_confidence': confidence
            }
        else:  # outdoor
            return {
                'leveling': {
                    'priority_methods': ['horizon_line', 'vertical_reference'],
                    'avoid_methods': [],
                    'confidence_boost': 0.0,
                    'prefer_architectural': False
                },
                'color_grading': {
                    'white_balance_hint': 'daylight_cloudy',
                    'temperature_range': (5000, 7000),  # Cooler outdoor lighting
                    'highlight_priority': 'high',  # Outdoor scenes often need highlight recovery
                    'contrast_boost': 0.0,  # Natural outdoor contrast
                    'skin_tone_protection': False,
                    'vibrance_modifier': 0.1  # Enhance landscape colors
                },
                'scene_confidence': confidence
            }