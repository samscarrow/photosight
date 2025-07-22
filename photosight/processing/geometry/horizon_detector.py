"""
Horizon and reference line detection for automatic straightening
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class HorizonDetector:
    """Detect horizon lines and other reference lines for straightening"""
    
    def __init__(self, 
                 edge_threshold_low: int = 30,
                 edge_threshold_high: int = 100,
                 hough_threshold: int = 60,
                 min_line_length: int = 100,
                 max_line_gap: int = 15):
        """
        Initialize horizon detector
        
        Args:
            edge_threshold_low: Lower threshold for Canny edge detection
            edge_threshold_high: Upper threshold for Canny edge detection
            hough_threshold: Minimum votes for Hough line detection
            min_line_length: Minimum line length to consider
            max_line_gap: Maximum gap between line segments to merge
        """
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        
    def detect_horizon(self, image: np.ndarray) -> Dict:
        """
        Detect horizon line in image with improved indoor scene handling
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary with horizon detection results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Detect edges
        edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
        
        # FIRST: Check for strong vertical references (picture frames, doors, etc.)
        # These are more reliable than horizontal lines in indoor scenes
        vertical_result = self._detect_vertical_references(edges)
        
        if vertical_result['reliable_vertical_detected']:
            # If we have strong vertical references, use them as the primary guide
            return {
                'horizon_detected': True,
                'angle': vertical_result['suggested_rotation'],
                'confidence': vertical_result['confidence'],
                'method': 'vertical_reference',
                'vertical_lines': vertical_result['vertical_lines'][:3],
                'note': 'Using vertical references (frames, architecture) as primary guide'
            }
        
        # SECOND: Traditional horizon detection, but with stricter requirements
        # Focus on middle third of image where horizon is likely
        horizon_region = edges[h//3:2*h//3, :]
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            horizon_region, 
            1, 
            np.pi/180, 
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length, 
            maxLineGap=self.max_line_gap
        )
        
        if lines is None or len(lines) == 0:
            return {
                'horizon_detected': False,
                'angle': 0.0,
                'confidence': 0.0,
                'method': 'none'
            }
        
        # Filter for near-horizontal lines (within 15 degrees, stricter than before)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15:  # Stricter horizontal requirement
                horizontal_lines.append((line[0], angle))
        
        if not horizontal_lines:
            # No reliable horizontal lines found
            return {
                'horizon_detected': False,
                'angle': 0.0,
                'confidence': 0.0,
                'method': 'no_reliable_horizon'
            }
        
        # Find dominant angle using RANSAC-like approach
        angles = [angle for _, angle in horizontal_lines]
        
        if len(angles) >= 3:
            # Use median for robustness
            dominant_angle = np.median(angles)
            angle_std = np.std(angles)
            
            # Stricter confidence calculation
            # Only high confidence if angles are consistent AND significant
            confidence = np.exp(-angle_std / 2.0) if angle_std < 1.5 and abs(dominant_angle) > 1.0 else 0.3
        else:
            dominant_angle = angles[0] if angles else 0.0
            confidence = 0.3  # Lower confidence for few lines
        
        # Additional check: if the angle is very small, don't bother
        if abs(dominant_angle) < 1.0:
            return {
                'horizon_detected': False,
                'angle': 0.0,
                'confidence': 0.0,
                'method': 'angle_too_small',
                'detected_angle': dominant_angle
            }
        
        return {
            'horizon_detected': True,
            'angle': dominant_angle,
            'confidence': confidence,
            'method': 'horizon_line',
            'line_count': len(horizontal_lines),
            'lines': horizontal_lines[:5]  # Top 5 lines for visualization
        }
    
    def _detect_vertical_references(self, edges: np.ndarray) -> Dict:
        """
        Detect vertical reference lines (picture frames, doors, architecture)
        These are often more reliable than horizon lines in indoor scenes
        
        Args:
            edges: Edge detection result
            
        Returns:
            Dictionary with vertical reference detection results
        """
        h, w = edges.shape
        
        # Detect lines using Hough transform optimized for verticals
        lines = cv2.HoughLinesP(
            edges, 
            1, 
            np.pi/180, 
            threshold=max(25, self.hough_threshold//3),  # Much lower threshold for verticals
            minLineLength=h//8,  # Minimum 1/8 image height for vertical lines (more sensitive)
            maxLineGap=self.max_line_gap*3  # Allow bigger gaps for furniture/frames
        )
        
        if lines is None:
            return {
                'reliable_vertical_detected': False,
                'suggested_rotation': 0.0,
                'confidence': 0.0
            }
        
        # Filter for near-vertical lines and convert to rotation angles
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle from vertical (0° = perfectly vertical)
            if x2 - x1 == 0:  # Perfectly vertical
                angle_from_vertical = 0.0
            else:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Convert to angle from vertical
                if angle > 0:
                    angle_from_vertical = 90 - angle
                else:
                    angle_from_vertical = -(90 + angle)
            
            # Only consider lines that are close to vertical (within 15°)
            if abs(angle_from_vertical) < 15:
                line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                vertical_lines.append({
                    'line': (x1, y1, x2, y2),
                    'angle_from_vertical': angle_from_vertical,
                    'rotation_needed': angle_from_vertical,  # Positive to correct the tilt
                    'length': line_length,
                    'strength': line_length * (15 - abs(angle_from_vertical)) / 15  # Longer & more vertical = stronger
                })
        
        if len(vertical_lines) < 2:
            return {
                'reliable_vertical_detected': False,
                'suggested_rotation': 0.0,
                'confidence': 0.0,
                'vertical_lines': vertical_lines
            }
        
        # Sort by strength (combination of length and verticality)
        vertical_lines.sort(key=lambda x: x['strength'], reverse=True)
        
        # Use the strongest vertical lines to determine rotation
        top_lines = vertical_lines[:min(5, len(vertical_lines))]
        rotation_angles = [line['rotation_needed'] for line in top_lines]
        
        # Calculate weighted average rotation
        weights = [line['strength'] for line in top_lines]
        total_weight = sum(weights)
        
        if total_weight == 0:
            suggested_rotation = 0.0
        else:
            suggested_rotation = sum(angle * weight for angle, weight in zip(rotation_angles, weights)) / total_weight
        
        # Calculate confidence based on consistency and strength
        angle_std = np.std(rotation_angles)
        avg_strength = np.mean([line['strength'] for line in top_lines])
        
        # High confidence if:
        # 1. Multiple strong vertical lines agree on the rotation
        # 2. Standard deviation is low (consistent)
        # 3. Lines are reasonably long and vertical
        if len(top_lines) >= 3 and angle_std < 2.0 and avg_strength > h/8:
            confidence = min(0.95, np.exp(-angle_std / 3.0) * min(1.0, avg_strength / (h/4)))
        elif len(top_lines) >= 2 and angle_std < 1.5 and avg_strength > h/6:
            confidence = min(0.8, np.exp(-angle_std / 2.0) * min(1.0, avg_strength / (h/3)))
        else:
            confidence = 0.4
        
        # Only consider it reliable if we have good confidence and meaningful rotation
        # For high confidence (>0.8), allow smaller rotations for precision
        if confidence > 0.8:
            reliable = abs(suggested_rotation) > 0.5
        else:
            reliable = confidence > 0.7 and abs(suggested_rotation) > 1.0
        
        return {
            'reliable_vertical_detected': reliable,
            'suggested_rotation': suggested_rotation,
            'confidence': confidence,
            'vertical_lines': vertical_lines,
            'top_lines_used': len(top_lines),
            'angle_consistency': angle_std
        }

    def _detect_alternative_references(self, image: np.ndarray, edges: np.ndarray) -> Dict:
        """
        Detect alternative reference lines (buildings, people, etc.)
        
        Args:
            image: RGB image array
            edges: Edge detection result
            
        Returns:
            Dictionary with detection results
        """
        h, w = edges.shape
        
        # Detect strong vertical lines (buildings, poles, people standing)
        lines = cv2.HoughLinesP(
            edges, 
            1, 
            np.pi/180, 
            threshold=self.hough_threshold,
            minLineLength=h//4,  # Longer lines for verticals
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return {
                'horizon_detected': False,
                'angle': 0.0,
                'confidence': 0.0,
                'method': 'none'
            }
        
        # Filter for near-vertical lines
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Vertical lines are at 90 or -90 degrees
            vertical_angle = abs(abs(angle) - 90)
            if vertical_angle < 10:  # Within 10 degrees of vertical
                # Convert to rotation needed (perpendicular to vertical)
                rotation_angle = angle - 90 if angle > 0 else angle + 90
                vertical_lines.append((line[0], rotation_angle))
        
        if not vertical_lines:
            # Try detecting based on image symmetry
            return self._detect_by_symmetry(image)
        
        # Find dominant vertical alignment
        rotation_angles = [angle for _, angle in vertical_lines]
        
        if len(rotation_angles) >= 2:
            dominant_angle = np.median(rotation_angles)
            angle_std = np.std(rotation_angles)
            confidence = np.exp(-angle_std / 5.0) * 0.8  # Slightly lower confidence than horizon
        else:
            dominant_angle = rotation_angles[0] if rotation_angles else 0.0
            confidence = 0.4
        
        return {
            'horizon_detected': True,
            'angle': dominant_angle,
            'confidence': confidence,
            'method': 'vertical_reference',
            'line_count': len(vertical_lines),
            'lines': vertical_lines[:5]
        }
    
    def _detect_by_symmetry(self, image: np.ndarray) -> Dict:
        """
        Detect rotation based on image symmetry
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary with detection results
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Test small rotation angles and measure symmetry
        best_angle = 0.0
        best_symmetry = 0.0
        
        for angle in np.linspace(-5, 5, 21):  # Test -5 to +5 degrees
            # Rotate image
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, rotation_matrix, (w, h))
            
            # Measure vertical symmetry
            left_half = rotated[:, :w//2]
            right_half = rotated[:, w//2:]
            right_flipped = cv2.flip(right_half, 1)
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_flipped = right_flipped[:, :min_width]
            
            # Calculate symmetry score
            diff = np.abs(left_half.astype(float) - right_flipped.astype(float))
            symmetry_score = 1.0 - np.mean(diff) / 255.0
            
            if symmetry_score > best_symmetry:
                best_symmetry = symmetry_score
                best_angle = angle
        
        # Only return if symmetry is significant
        if best_symmetry > 0.7:
            return {
                'horizon_detected': True,
                'angle': best_angle,
                'confidence': best_symmetry * 0.6,  # Lower confidence for symmetry method
                'method': 'symmetry',
                'symmetry_score': best_symmetry
            }
        else:
            return {
                'horizon_detected': False,
                'angle': 0.0,
                'confidence': 0.0,
                'method': 'none'
            }
    
    def detect_grid_lines(self, image: np.ndarray) -> Dict:
        """
        Detect architectural grid lines for straightening
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary with grid line detection
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
        
        # Detect all lines
        lines = cv2.HoughLinesP(
            edges, 
            1, 
            np.pi/180, 
            threshold=self.hough_threshold//2,  # Lower threshold for more lines
            minLineLength=self.min_line_length//2,
            maxLineGap=self.max_line_gap*2
        )
        
        if lines is None:
            return {
                'grid_detected': False,
                'dominant_angle': 0.0,
                'confidence': 0.0
            }
        
        # Classify lines by angle
        horizontal_angles = []
        vertical_angles = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            if abs(angle) < 45:  # More horizontal
                horizontal_angles.append(angle)
            else:  # More vertical
                # Convert to rotation angle needed
                rotation_angle = angle - 90 if angle > 0 else angle + 90
                vertical_angles.append(rotation_angle)
        
        # Find dominant angles
        all_angles = horizontal_angles + vertical_angles
        
        if len(all_angles) >= 5:
            # Use kernel density estimation to find dominant angle
            from scipy.stats import gaussian_kde
            
            if len(all_angles) > 1:
                kde = gaussian_kde(all_angles)
                angle_range = np.linspace(-10, 10, 100)
                density = kde(angle_range)
                dominant_angle = angle_range[np.argmax(density)]
                
                # Confidence based on peak prominence
                confidence = np.max(density) / np.mean(density)
                confidence = min(1.0, confidence / 5.0)
            else:
                dominant_angle = all_angles[0]
                confidence = 0.3
            
            return {
                'grid_detected': True,
                'dominant_angle': dominant_angle,
                'confidence': confidence,
                'horizontal_count': len(horizontal_angles),
                'vertical_count': len(vertical_angles)
            }
        
        return {
            'grid_detected': False,
            'dominant_angle': 0.0,
            'confidence': 0.0
        }