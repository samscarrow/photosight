"""
Advanced composition analysis for photo curation
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)


class CompositionAnalyzer:
    """Analyze photo composition using computer vision techniques"""
    
    def __init__(self):
        """Initialize composition analyzer"""
        pass
        
    def analyze_composition(self, image: np.ndarray) -> Dict:
        """
        Analyze overall composition of image
        
        Args:
            image: Image array (RGB)
            
        Returns:
            Dictionary with composition analysis
        """
        results = {
            'rule_of_thirds': self._check_rule_of_thirds(image),
            'symmetry': self._check_symmetry(image),
            'leading_lines': self._detect_leading_lines(image),
            'depth_layers': self._analyze_depth_layers(image),
            'color_harmony': self._analyze_color_harmony(image),
            'visual_balance': self._check_visual_balance(image),
            'golden_ratio': self._check_golden_ratio(image),
            'composition_score': 0.0
        }
        
        # Calculate overall composition score
        results['composition_score'] = self._calculate_composition_score(results)
        
        return results
        
    def _check_rule_of_thirds(self, image: np.ndarray) -> Dict:
        """Check if image follows rule of thirds"""
        h, w = image.shape[:2]
        
        # Define rule of thirds lines
        vertical_lines = [w // 3, 2 * w // 3]
        horizontal_lines = [h // 3, 2 * h // 3]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density near rule of thirds lines
        line_strength = []
        
        # Check vertical lines
        for x in vertical_lines:
            region = edges[:, max(0, x-10):min(w, x+10)]
            strength = np.sum(region) / (h * 20)
            line_strength.append(strength)
            
        # Check horizontal lines
        for y in horizontal_lines:
            region = edges[max(0, y-10):min(h, y+10), :]
            strength = np.sum(region) / (w * 20)
            line_strength.append(strength)
            
        # Check intersection points (power points)
        power_points = []
        for x in vertical_lines:
            for y in horizontal_lines:
                # Check region around intersection
                region = edges[max(0, y-20):min(h, y+20), 
                             max(0, x-20):min(w, x+20)]
                strength = np.sum(region) / (40 * 40)
                power_points.append(strength)
                
        # Normalize strengths
        avg_line_strength = np.mean(line_strength) / 255
        avg_point_strength = np.mean(power_points) / 255
        
        return {
            'follows_rule': avg_line_strength > 0.1 or avg_point_strength > 0.15,
            'line_strength': avg_line_strength,
            'point_strength': avg_point_strength,
            'score': min(1.0, (avg_line_strength + avg_point_strength) * 2)
        }
        
    def _check_symmetry(self, image: np.ndarray) -> Dict:
        """Check image symmetry (vertical and horizontal)"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Vertical symmetry
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        right_flipped = cv2.flip(right_half, 1)
        
        # Resize to same width if odd width
        if left_half.shape[1] != right_flipped.shape[1]:
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_flipped = right_flipped[:, :min_width]
            
        vertical_diff = np.mean(np.abs(left_half - right_flipped)) / 255
        vertical_symmetry = 1.0 - vertical_diff
        
        # Horizontal symmetry
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        bottom_flipped = cv2.flip(bottom_half, 0)
        
        # Resize to same height if odd height
        if top_half.shape[0] != bottom_flipped.shape[0]:
            min_height = min(top_half.shape[0], bottom_flipped.shape[0])
            top_half = top_half[:min_height, :]
            bottom_flipped = bottom_flipped[:min_height, :]
            
        horizontal_diff = np.mean(np.abs(top_half - bottom_flipped)) / 255
        horizontal_symmetry = 1.0 - horizontal_diff
        
        return {
            'vertical_symmetry': vertical_symmetry,
            'horizontal_symmetry': horizontal_symmetry,
            'is_symmetric': max(vertical_symmetry, horizontal_symmetry) > 0.7,
            'score': max(vertical_symmetry, horizontal_symmetry)
        }
        
    def _detect_leading_lines(self, image: np.ndarray) -> Dict:
        """Detect leading lines using Hough transform"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                               minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return {
                'has_leading_lines': False,
                'line_count': 0,
                'convergence_point': None,
                'score': 0.0
            }
            
        # Analyze line directions
        line_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            line_angles.append(angle)
            
        # Check for converging lines
        convergence_points = []
        h, w = image.shape[:2]
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                # Calculate intersection point
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) > 0.01:  # Lines not parallel
                    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                    
                    # Intersection point
                    px = x1 + t * (x2 - x1)
                    py = y1 + t * (y2 - y1)
                    
                    # Check if point is within or near image
                    if -w*0.5 < px < w*1.5 and -h*0.5 < py < h*1.5:
                        convergence_points.append((px, py))
                        
        has_convergence = len(convergence_points) > 3
        
        return {
            'has_leading_lines': len(lines) > 2,
            'line_count': len(lines),
            'has_convergence': has_convergence,
            'score': min(1.0, len(lines) / 10.0) * (1.5 if has_convergence else 1.0)
        }
        
    def _analyze_depth_layers(self, image: np.ndarray) -> Dict:
        """Analyze depth layers using edge detection and blur"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Divide image into thirds (foreground, middle, background)
        top_third = gray[:h//3, :]
        middle_third = gray[h//3:2*h//3, :]
        bottom_third = gray[2*h//3:, :]
        
        # Calculate sharpness for each region
        def calculate_sharpness(region):
            laplacian = cv2.Laplacian(region, cv2.CV_64F)
            return laplacian.var()
            
        top_sharp = calculate_sharpness(top_third)
        middle_sharp = calculate_sharpness(middle_third)
        bottom_sharp = calculate_sharpness(bottom_third)
        
        # Typical depth patterns
        has_depth = False
        depth_pattern = "flat"
        
        if bottom_sharp > middle_sharp > top_sharp:
            has_depth = True
            depth_pattern = "foreground_focus"
        elif top_sharp < middle_sharp and bottom_sharp < middle_sharp:
            has_depth = True
            depth_pattern = "middle_focus"
        elif top_sharp > middle_sharp > bottom_sharp:
            has_depth = True
            depth_pattern = "background_focus"
            
        # Calculate depth score
        sharpness_values = [top_sharp, middle_sharp, bottom_sharp]
        sharpness_range = max(sharpness_values) - min(sharpness_values)
        depth_score = min(1.0, sharpness_range / 1000.0)
        
        return {
            'has_depth': has_depth,
            'depth_pattern': depth_pattern,
            'sharpness_distribution': {
                'top': top_sharp,
                'middle': middle_sharp,
                'bottom': bottom_sharp
            },
            'score': depth_score
        }
        
    def _analyze_color_harmony(self, image: np.ndarray) -> Dict:
        """Analyze color harmony and palette"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract dominant colors using k-means
        pixels = hsv.reshape(-1, 3)
        
        # Sample pixels for efficiency
        sample_size = min(5000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]
        
        # Simple k-means with 5 colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(sample_pixels)
        
        # Get dominant colors
        colors = kmeans.cluster_centers_
        color_percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_)
        
        # Analyze color relationships
        hues = colors[:, 0]  # Hue channel
        saturations = colors[:, 1]  # Saturation channel
        values = colors[:, 2]  # Value channel
        
        # Check for complementary colors (180° apart)
        complementary = False
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                hue_diff = abs(hues[i] - hues[j])
                if 170 < hue_diff < 190 or 170 < (360 - hue_diff) < 190:
                    complementary = True
                    
        # Check for analogous colors (within 30°)
        analogous = False
        hue_range = np.ptp(hues)
        if hue_range < 60:
            analogous = True
            
        # Calculate color diversity
        hue_diversity = hue_range / 180.0  # Normalize to 0-1
        saturation_avg = np.mean(saturations) / 255.0
        
        # Harmony score
        harmony_score = 0.0
        if complementary:
            harmony_score += 0.4
        if analogous:
            harmony_score += 0.3
        harmony_score += 0.3 * saturation_avg
        
        return {
            'has_complementary': complementary,
            'has_analogous': analogous,
            'color_diversity': hue_diversity,
            'saturation_level': saturation_avg,
            'dominant_colors': len(colors),
            'score': min(1.0, harmony_score)
        }
        
    def _check_visual_balance(self, image: np.ndarray) -> Dict:
        """Check visual balance using visual weight distribution"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Calculate visual weight (brightness + edges)
        edges = cv2.Canny(gray, 50, 150)
        visual_weight = gray.astype(float) / 255.0 + edges.astype(float) / 255.0
        
        # Divide into quadrants
        mid_x, mid_y = w // 2, h // 2
        
        quadrants = {
            'top_left': visual_weight[:mid_y, :mid_x],
            'top_right': visual_weight[:mid_y, mid_x:],
            'bottom_left': visual_weight[mid_y:, :mid_x],
            'bottom_right': visual_weight[mid_y:, mid_x:]
        }
        
        # Calculate weight for each quadrant
        weights = {k: np.sum(v) for k, v in quadrants.items()}
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            weight_distribution = {k: v / total_weight for k, v in weights.items()}
        else:
            weight_distribution = {k: 0.25 for k in weights.keys()}
            
        # Check balance
        # Perfect balance would be 0.25 for each quadrant
        balance_deviation = sum(abs(w - 0.25) for w in weight_distribution.values())
        balance_score = 1.0 - (balance_deviation / 2.0)  # Normalize
        
        # Check diagonal balance
        diagonal1 = weight_distribution['top_left'] + weight_distribution['bottom_right']
        diagonal2 = weight_distribution['top_right'] + weight_distribution['bottom_left']
        diagonal_balance = 1.0 - abs(diagonal1 - diagonal2)
        
        return {
            'weight_distribution': weight_distribution,
            'is_balanced': balance_score > 0.7,
            'balance_score': balance_score,
            'diagonal_balance': diagonal_balance,
            'score': (balance_score + diagonal_balance) / 2.0
        }
        
    def _check_golden_ratio(self, image: np.ndarray) -> Dict:
        """Check if image follows golden ratio composition"""
        h, w = image.shape[:2]
        
        # Golden ratio
        phi = 1.618
        
        # Calculate golden ratio lines
        golden_vertical = [int(w / phi), int(w - w / phi)]
        golden_horizontal = [int(h / phi), int(h - h / phi)]
        
        # Similar to rule of thirds but with golden ratio
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check edge density near golden lines
        line_strength = []
        
        for x in golden_vertical:
            region = edges[:, max(0, x-10):min(w, x+10)]
            strength = np.sum(region) / (h * 20)
            line_strength.append(strength)
            
        for y in golden_horizontal:
            region = edges[max(0, y-10):min(h, y+10), :]
            strength = np.sum(region) / (w * 20)
            line_strength.append(strength)
            
        avg_strength = np.mean(line_strength) / 255
        
        return {
            'follows_golden_ratio': avg_strength > 0.1,
            'strength': avg_strength,
            'score': min(1.0, avg_strength * 3)
        }
        
    def _calculate_composition_score(self, results: Dict) -> float:
        """Calculate overall composition score"""
        # Weight different aspects
        weights = {
            'rule_of_thirds': 0.2,
            'symmetry': 0.15,
            'leading_lines': 0.15,
            'depth_layers': 0.15,
            'color_harmony': 0.15,
            'visual_balance': 0.15,
            'golden_ratio': 0.05
        }
        
        score = 0.0
        for aspect, weight in weights.items():
            if aspect in results and 'score' in results[aspect]:
                score += weight * results[aspect]['score']
                
        return min(1.0, score)