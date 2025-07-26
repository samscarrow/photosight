"""
Composition analysis for photos.

Analyzes compositional elements like rule of thirds, leading lines,
symmetry, and visual balance.
"""

import logging
import numpy as np
import cv2
from typing import Dict, Tuple, List
from scipy import ndimage

logger = logging.getLogger(__name__)


class CompositionAnalyzer:
    """
    Analyzes photographic composition using computer vision techniques.
    
    Evaluates:
    - Rule of thirds
    - Visual balance
    - Leading lines
    - Symmetry
    - Edge placement
    - Subject positioning
    """
    
    def __init__(self, config: Dict):
        """Initialize the composition analyzer."""
        self.config = config
        self.composition_config = config.get('composition_analysis', {})
    
    def analyze_composition(self, img_array: np.ndarray) -> Dict:
        """
        Perform comprehensive composition analysis.
        
        Args:
            img_array: Image as numpy array (RGB)
            
        Returns:
            Dictionary containing composition analysis results
        """
        try:
            height, width = img_array.shape[:2]
            
            results = {
                'image_dimensions': {'width': width, 'height': height}
            }
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Rule of thirds analysis
            rule_of_thirds = self._analyze_rule_of_thirds(gray)
            results.update(rule_of_thirds)
            
            # Visual balance analysis
            balance = self._analyze_visual_balance(gray)
            results.update(balance)
            
            # Leading lines analysis
            leading_lines = self._analyze_leading_lines(gray)
            results.update(leading_lines)
            
            # Symmetry analysis
            symmetry = self._analyze_symmetry(gray)
            results.update(symmetry)
            
            # Edge distribution analysis
            edge_analysis = self._analyze_edge_distribution(gray)
            results.update(edge_analysis)
            
            # Calculate overall composition score
            results['composition_score'] = self._calculate_composition_score(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Composition analysis error: {e}")
            return {
                'composition_score': 0.5,
                'error': str(e)
            }
    
    def _analyze_rule_of_thirds(self, gray: np.ndarray) -> Dict:
        """
        Analyze rule of thirds composition.
        
        Args:
            gray: Grayscale image array
            
        Returns:
            Dictionary with rule of thirds analysis
        """
        try:
            height, width = gray.shape
            
            # Calculate third lines
            third_h = height // 3
            third_w = width // 3
            
            # Define rule of thirds grid intersection points
            intersections = [
                (third_w, third_h),
                (2 * third_w, third_h),
                (third_w, 2 * third_h),
                (2 * third_w, 2 * third_h)
            ]
            
            # Calculate interest points using edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours for interest points
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate centroid of largest contours (main subjects)
            interest_points = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        interest_points.append((cx, cy))
            
            # Calculate distances from interest points to rule of thirds intersections
            min_distances = []
            for point in interest_points:
                distances_to_intersections = [
                    np.sqrt((point[0] - intersection[0])**2 + (point[1] - intersection[1])**2)
                    for intersection in intersections
                ]
                min_distances.append(min(distances_to_intersections))
            
            # Score based on how close interest points are to intersections
            if min_distances:
                avg_distance = np.mean(min_distances)
                # Normalize distance (closer is better)
                max_possible_distance = np.sqrt(width**2 + height**2)
                rule_of_thirds_score = 1.0 - (avg_distance / max_possible_distance)
                rule_of_thirds_score = max(0.0, min(1.0, rule_of_thirds_score * 2))  # Amplify score
            else:
                rule_of_thirds_score = 0.5  # Default if no interest points found
            
            # Analyze line alignment with rule of thirds
            third_lines_strength = self._analyze_third_line_alignment(edges, width, height)
            
            return {
                'rule_of_thirds_score': float(rule_of_thirds_score),
                'interest_points_count': len(interest_points),
                'third_lines_alignment': float(third_lines_strength),
                'intersections': intersections
            }
            
        except Exception as e:
            logger.warning(f"Rule of thirds analysis error: {e}")
            return {
                'rule_of_thirds_score': 0.5,
                'interest_points_count': 0,
                'third_lines_alignment': 0.5
            }
    
    def _analyze_third_line_alignment(self, edges: np.ndarray, width: int, height: int) -> float:
        """Analyze how well lines in the image align with rule of thirds."""
        try:
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=min(width, height)//4, maxLineGap=10)
            
            if lines is None:
                return 0.5
            
            # Rule of thirds line positions
            third_positions = {
                'vertical': [width//3, 2*width//3],
                'horizontal': [height//3, 2*height//3]
            }
            
            alignment_scores = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is more vertical or horizontal
                if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
                    line_pos = (y1 + y2) / 2
                    distances = [abs(line_pos - pos) for pos in third_positions['horizontal']]
                    min_distance = min(distances)
                    alignment_score = 1.0 - (min_distance / height)
                else:  # Vertical line
                    line_pos = (x1 + x2) / 2
                    distances = [abs(line_pos - pos) for pos in third_positions['vertical']]
                    min_distance = min(distances)
                    alignment_score = 1.0 - (min_distance / width)
                
                alignment_scores.append(max(0.0, alignment_score))
            
            return float(np.mean(alignment_scores)) if alignment_scores else 0.5
            
        except Exception as e:
            logger.warning(f"Third line alignment error: {e}")
            return 0.5
    
    def _analyze_visual_balance(self, gray: np.ndarray) -> Dict:
        """
        Analyze visual balance in the image.
        
        Args:
            gray: Grayscale image array
            
        Returns:
            Dictionary with visual balance analysis
        """
        try:
            height, width = gray.shape
            
            # Calculate center of mass
            y_coords, x_coords = np.ogrid[:height, :width]
            
            # Weight each pixel by its intensity
            total_mass = np.sum(gray)
            if total_mass == 0:
                return {'balance_score': 0.5, 'center_of_mass': (width//2, height//2)}
            
            center_x = np.sum(x_coords * gray) / total_mass
            center_y = np.sum(y_coords * gray) / total_mass
            
            # Calculate how far center of mass is from image center
            image_center_x = width / 2
            image_center_y = height / 2
            
            distance_from_center = np.sqrt(
                (center_x - image_center_x)**2 + (center_y - image_center_y)**2
            )
            
            # Normalize distance (closer to center is better balance)
            max_distance = np.sqrt((width/2)**2 + (height/2)**2)
            balance_score = 1.0 - (distance_from_center / max_distance)
            
            # Analyze left-right balance
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            
            left_mass = np.sum(left_half)
            right_mass = np.sum(right_half)
            
            # Convert to float to prevent overflow
            left_mass_f = float(left_mass)
            right_mass_f = float(right_mass)
            denominator = left_mass_f + right_mass_f
            
            if denominator > 0:
                lr_balance = 1.0 - abs(left_mass_f - right_mass_f) / denominator
            else:
                lr_balance = 1.0
            
            # Analyze top-bottom balance
            top_half = gray[:height//2, :]
            bottom_half = gray[height//2:, :]
            
            top_mass = np.sum(top_half)
            bottom_mass = np.sum(bottom_half)
            
            # Convert to float to prevent overflow
            top_mass_f = float(top_mass)
            bottom_mass_f = float(bottom_mass)
            denominator = top_mass_f + bottom_mass_f
            
            if denominator > 0:
                tb_balance = 1.0 - abs(top_mass_f - bottom_mass_f) / denominator
            else:
                tb_balance = 1.0
            
            # Overall balance score
            overall_balance = (balance_score + lr_balance + tb_balance) / 3
            
            return {
                'balance_score': float(overall_balance),
                'center_of_mass': (float(center_x), float(center_y)),
                'lr_balance': float(lr_balance),
                'tb_balance': float(tb_balance),
                'distance_from_center': float(distance_from_center)
            }
            
        except Exception as e:
            logger.warning(f"Visual balance analysis error: {e}")
            return {
                'balance_score': 0.5,
                'center_of_mass': (0.0, 0.0),
                'lr_balance': 0.5,
                'tb_balance': 0.5
            }
    
    def _analyze_leading_lines(self, gray: np.ndarray) -> Dict:
        """
        Analyze leading lines in the composition.
        
        Args:
            gray: Grayscale image array
            
        Returns:
            Dictionary with leading lines analysis
        """
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=min(gray.shape)//3, maxLineGap=20)
            
            if lines is None:
                return {
                    'leading_lines_score': 0.3,
                    'line_count': 0,
                    'convergence_score': 0.0
                }
            
            height, width = gray.shape
            image_center = (width//2, height//2)
            
            # Analyze line convergence toward center or edges
            convergence_scores = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line direction and check if it leads toward center
                line_vector = np.array([x2 - x1, y2 - y1])
                line_length = np.linalg.norm(line_vector)
                
                if line_length > 0:
                    # Check if line points toward center or interesting areas
                    line_midpoint = ((x1 + x2)//2, (y1 + y2)//2)
                    distance_to_center = np.sqrt(
                        (line_midpoint[0] - image_center[0])**2 + 
                        (line_midpoint[1] - image_center[1])**2
                    )
                    
                    # Normalize distance
                    max_distance = np.sqrt((width//2)**2 + (height//2)**2)
                    convergence_score = 1.0 - (distance_to_center / max_distance)
                    convergence_scores.append(convergence_score)
            
            # Calculate overall leading lines score
            if convergence_scores:
                avg_convergence = np.mean(convergence_scores)
                line_density = min(len(lines) / 10.0, 1.0)  # Normalize line count
                leading_lines_score = (avg_convergence + line_density) / 2
            else:
                leading_lines_score = 0.3
            
            return {
                'leading_lines_score': float(leading_lines_score),
                'line_count': len(lines),
                'convergence_score': float(np.mean(convergence_scores)) if convergence_scores else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Leading lines analysis error: {e}")
            return {
                'leading_lines_score': 0.3,
                'line_count': 0,
                'convergence_score': 0.0
            }
    
    def _analyze_symmetry(self, gray: np.ndarray) -> Dict:
        """
        Analyze symmetry in the composition.
        
        Args:
            gray: Grayscale image array
            
        Returns:
            Dictionary with symmetry analysis
        """
        try:
            height, width = gray.shape
            
            # Vertical symmetry (left-right)
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            right_half_flipped = np.fliplr(right_half)
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            
            # Calculate vertical symmetry score
            vertical_diff = np.mean(np.abs(left_half.astype(float) - right_half_flipped.astype(float)))
            vertical_symmetry = 1.0 - (vertical_diff / 255.0)
            
            # Horizontal symmetry (top-bottom)
            top_half = gray[:height//2, :]
            bottom_half = gray[height//2:, :]
            bottom_half_flipped = np.flipud(bottom_half)
            
            # Resize to match if needed
            min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
            top_half = top_half[:min_height, :]
            bottom_half_flipped = bottom_half_flipped[:min_height, :]
            
            # Calculate horizontal symmetry score
            horizontal_diff = np.mean(np.abs(top_half.astype(float) - bottom_half_flipped.astype(float)))
            horizontal_symmetry = 1.0 - (horizontal_diff / 255.0)
            
            # Overall symmetry score (take the better of the two)
            symmetry_score = max(vertical_symmetry, horizontal_symmetry)
            
            return {
                'symmetry_score': float(symmetry_score),
                'vertical_symmetry': float(vertical_symmetry),
                'horizontal_symmetry': float(horizontal_symmetry)
            }
            
        except Exception as e:
            logger.warning(f"Symmetry analysis error: {e}")
            return {
                'symmetry_score': 0.5,
                'vertical_symmetry': 0.5,
                'horizontal_symmetry': 0.5
            }
    
    def _analyze_edge_distribution(self, gray: np.ndarray) -> Dict:
        """
        Analyze distribution of edges and details.
        
        Args:
            gray: Grayscale image array
            
        Returns:
            Dictionary with edge distribution analysis
        """
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            height, width = edges.shape
            
            # Divide image into grid and analyze edge density
            grid_size = 3
            h_step = height // grid_size
            w_step = width // grid_size
            
            edge_densities = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    y1 = i * h_step
                    y2 = min((i + 1) * h_step, height)
                    x1 = j * w_step
                    x2 = min((j + 1) * w_step, width)
                    
                    grid_cell = edges[y1:y2, x1:x2]
                    edge_density = np.sum(grid_cell) / (grid_cell.size * 255)
                    edge_densities.append(edge_density)
            
            # Calculate distribution statistics
            edge_densities = np.array(edge_densities)
            mean_density = np.mean(edge_densities)
            std_density = np.std(edge_densities)
            
            # Good distribution has moderate density with some variation
            distribution_score = mean_density * (1.0 - min(std_density * 2, 1.0))
            
            return {
                'edge_distribution_score': float(distribution_score),
                'mean_edge_density': float(mean_density),
                'edge_density_variation': float(std_density),
                'grid_densities': edge_densities.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Edge distribution analysis error: {e}")
            return {
                'edge_distribution_score': 0.5,
                'mean_edge_density': 0.0,
                'edge_density_variation': 0.0
            }
    
    def _calculate_composition_score(self, analysis: Dict) -> float:
        """
        Calculate overall composition score from individual analyses.
        
        Args:
            analysis: Dictionary containing all composition analysis results
            
        Returns:
            Overall composition score (0.0-1.0)
        """
        try:
            # Weight factors for different composition aspects
            weights = {
                'rule_of_thirds': 0.30,
                'balance': 0.25,
                'leading_lines': 0.20,
                'symmetry': 0.15,
                'edge_distribution': 0.10
            }
            
            # Get individual scores
            rule_of_thirds_score = analysis.get('rule_of_thirds_score', 0.5)
            balance_score = analysis.get('balance_score', 0.5)
            leading_lines_score = analysis.get('leading_lines_score', 0.5)
            symmetry_score = analysis.get('symmetry_score', 0.5)
            edge_distribution_score = analysis.get('edge_distribution_score', 0.5)
            
            # Calculate weighted composition score
            composition_score = (
                rule_of_thirds_score * weights['rule_of_thirds'] +
                balance_score * weights['balance'] +
                leading_lines_score * weights['leading_lines'] +
                symmetry_score * weights['symmetry'] +
                edge_distribution_score * weights['edge_distribution']
            )
            
            return max(0.0, min(1.0, composition_score))
            
        except Exception as e:
            logger.warning(f"Composition score calculation error: {e}")
            return 0.5