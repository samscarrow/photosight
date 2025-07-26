"""
Aesthetic analysis for photos.

Analyzes aesthetic qualities like color harmony, visual appeal,
and artistic merit using computer vision and AI techniques.
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
import colorsys

logger = logging.getLogger(__name__)


class AestheticAnalyzer:
    """
    Analyzes aesthetic qualities of photographs.
    
    Evaluates:
    - Color harmony and palette
    - Contrast and visual impact
    - Saturation and vibrancy
    - Overall aesthetic appeal
    - Mood and atmosphere
    """
    
    def __init__(self, config: Dict):
        """Initialize the aesthetic analyzer."""
        self.config = config
        self.aesthetic_config = config.get('aesthetic_analysis', {})
    
    def analyze_aesthetics(self, img_array: np.ndarray) -> Dict:
        """
        Perform comprehensive aesthetic analysis.
        
        Args:
            img_array: Image as numpy array (RGB)
            
        Returns:
            Dictionary containing aesthetic analysis results
        """
        try:
            results = {}
            
            # Color harmony analysis
            color_analysis = self._analyze_color_harmony(img_array)
            results.update(color_analysis)
            
            # Contrast and visual impact
            contrast_analysis = self._analyze_visual_impact(img_array)
            results.update(contrast_analysis)
            
            # Saturation and vibrancy
            saturation_analysis = self._analyze_saturation(img_array)
            results.update(saturation_analysis)
            
            # Mood and atmosphere
            mood_analysis = self._analyze_mood(img_array)
            results.update(mood_analysis)
            
            # Overall aesthetic appeal
            results['overall_appeal'] = self._calculate_aesthetic_score(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Aesthetic analysis error: {e}")
            return {
                'overall_appeal': 0.5,
                'color_harmony': 0.5,
                'contrast_score': 0.5,
                'saturation_score': 0.5,
                'error': str(e)
            }
    
    def _analyze_color_harmony(self, img_array: np.ndarray) -> Dict:
        """
        Analyze color harmony and palette quality.
        
        Args:
            img_array: RGB image array
            
        Returns:
            Dictionary with color harmony analysis
        """
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Extract dominant colors using K-means clustering
            pixels = img_array.reshape(-1, 3)
            
            # Sample pixels for efficiency
            sample_size = min(10000, len(pixels))
            sampled_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
            
            # K-means clustering to find dominant colors
            n_colors = 5
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(sampled_pixels)
            
            dominant_colors = kmeans.cluster_centers_.astype(int)
            
            # Convert dominant colors to HSV for harmony analysis
            dominant_hsv = []
            for color in dominant_colors:
                color_rgb = color.reshape(1, 1, 3)
                color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)[0, 0]
                dominant_hsv.append(color_hsv)
            
            # Analyze color relationships
            harmony_score = self._calculate_color_harmony_score(dominant_hsv)
            
            # Analyze color temperature
            temperature_analysis = self._analyze_color_temperature(img_array)
            
            # Color distribution analysis
            distribution_analysis = self._analyze_color_distribution(hsv)
            
            return {
                'color_harmony': float(harmony_score),
                'dominant_colors': dominant_colors.tolist(),
                'color_temperature': temperature_analysis,
                'color_distribution': distribution_analysis
            }
            
        except Exception as e:
            logger.warning(f"Color harmony analysis error: {e}")
            return {
                'color_harmony': 0.5,
                'dominant_colors': [],
                'color_temperature': {'score': 0.5, 'temperature': 'neutral'},
                'color_distribution': {'score': 0.5}
            }
    
    def _calculate_color_harmony_score(self, dominant_hsv: List) -> float:
        """
        Calculate color harmony score based on color theory.
        
        Args:
            dominant_hsv: List of dominant colors in HSV format
            
        Returns:
            Color harmony score (0.0-1.0)
        """
        try:
            if len(dominant_hsv) < 2:
                return 0.5
            
            hues = [color[0] for color in dominant_hsv]
            
            # Check for common harmony types
            harmony_scores = []
            
            # Monochromatic harmony (similar hues)
            hue_differences = []
            for i in range(len(hues)):
                for j in range(i + 1, len(hues)):
                    diff = min(abs(hues[i] - hues[j]), 360 - abs(hues[i] - hues[j]))
                    hue_differences.append(diff)
            
            if hue_differences:
                avg_hue_diff = np.mean(hue_differences)
                
                # Monochromatic (0-30 degrees)
                if avg_hue_diff <= 30:
                    harmony_scores.append(0.8)
                
                # Analogous (30-60 degrees)
                elif avg_hue_diff <= 60:
                    harmony_scores.append(0.9)
                
                # Complementary (150-210 degrees)
                elif 150 <= avg_hue_diff <= 210:
                    harmony_scores.append(0.85)
                
                # Triadic (around 120 degrees)
                elif 100 <= avg_hue_diff <= 140:
                    harmony_scores.append(0.8)
                
                # Split complementary or tetradic
                else:
                    harmony_scores.append(0.6)
            
            # Consider saturation harmony
            saturations = [color[1] for color in dominant_hsv]
            saturation_variance = np.var(saturations)
            saturation_harmony = 1.0 - min(saturation_variance / 10000.0, 1.0)
            harmony_scores.append(saturation_harmony)
            
            # Consider value (brightness) harmony
            values = [color[2] for color in dominant_hsv]
            value_variance = np.var(values)
            value_harmony = 1.0 - min(value_variance / 10000.0, 1.0)
            harmony_scores.append(value_harmony)
            
            return float(np.mean(harmony_scores)) if harmony_scores else 0.5
            
        except Exception as e:
            logger.warning(f"Color harmony calculation error: {e}")
            return 0.5
    
    def _analyze_color_temperature(self, img_array: np.ndarray) -> Dict:
        """
        Analyze color temperature and warmth.
        
        Args:
            img_array: RGB image array
            
        Returns:
            Dictionary with color temperature analysis
        """
        try:
            # Calculate average color values
            avg_r = np.mean(img_array[:, :, 0])
            avg_g = np.mean(img_array[:, :, 1])
            avg_b = np.mean(img_array[:, :, 2])
            
            # Simple color temperature estimation
            if avg_r > avg_b + 10:
                if avg_r > avg_b + 30:
                    temperature = 'very_warm'
                    score = 0.8  # Warm can be appealing
                else:
                    temperature = 'warm'
                    score = 0.9
            elif avg_b > avg_r + 10:
                if avg_b > avg_r + 30:
                    temperature = 'very_cool'
                    score = 0.7
                else:
                    temperature = 'cool'
                    score = 0.8
            else:
                temperature = 'neutral'
                score = 0.85  # Neutral is often good
            
            # Calculate warmth ratio
            warmth_ratio = (avg_r - avg_b) / 255.0
            
            return {
                'score': float(score),
                'temperature': temperature,
                'warmth_ratio': float(warmth_ratio),
                'avg_rgb': {'r': float(avg_r), 'g': float(avg_g), 'b': float(avg_b)}
            }
            
        except Exception as e:
            logger.warning(f"Color temperature analysis error: {e}")
            return {
                'score': 0.5,
                'temperature': 'neutral',
                'warmth_ratio': 0.0
            }
    
    def _analyze_color_distribution(self, hsv: np.ndarray) -> Dict:
        """
        Analyze color distribution across the image.
        
        Args:
            hsv: HSV image array
            
        Returns:
            Dictionary with color distribution analysis
        """
        try:
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            
            # Calculate histograms
            h_hist, _ = np.histogram(h_channel, bins=36, range=(0, 180))  # OpenCV uses 0-179 for hue
            s_hist, _ = np.histogram(s_channel, bins=32, range=(0, 256))
            v_hist, _ = np.histogram(v_channel, bins=32, range=(0, 256))
            
            # Normalize histograms
            h_hist = h_hist / np.sum(h_hist)
            s_hist = s_hist / np.sum(s_hist)
            v_hist = v_hist / np.sum(v_hist)
            
            # Calculate distribution scores
            # Good distribution has some variety but not too chaotic
            h_entropy = -np.sum(h_hist * np.log(h_hist + 1e-10))
            s_entropy = -np.sum(s_hist * np.log(s_hist + 1e-10))
            v_entropy = -np.sum(v_hist * np.log(v_hist + 1e-10))
            
            # Normalize entropy scores
            max_h_entropy = np.log(36)  # Maximum possible entropy for hue
            max_s_entropy = np.log(32)  # Maximum possible entropy for saturation
            max_v_entropy = np.log(32)  # Maximum possible entropy for value
            
            h_diversity = h_entropy / max_h_entropy
            s_diversity = s_entropy / max_s_entropy
            v_diversity = v_entropy / max_v_entropy
            
            # Good distribution has moderate diversity (not too uniform, not too chaotic)
            optimal_diversity = 0.7
            h_score = 1.0 - abs(h_diversity - optimal_diversity) / optimal_diversity
            s_score = 1.0 - abs(s_diversity - optimal_diversity) / optimal_diversity
            v_score = 1.0 - abs(v_diversity - optimal_diversity) / optimal_diversity
            
            distribution_score = (h_score + s_score + v_score) / 3
            
            return {
                'score': float(distribution_score),
                'hue_diversity': float(h_diversity),
                'saturation_diversity': float(s_diversity),
                'value_diversity': float(v_diversity)
            }
            
        except Exception as e:
            logger.warning(f"Color distribution analysis error: {e}")
            return {
                'score': 0.5,
                'hue_diversity': 0.5,
                'saturation_diversity': 0.5,
                'value_diversity': 0.5
            }
    
    def _analyze_visual_impact(self, img_array: np.ndarray) -> Dict:
        """
        Analyze visual impact and contrast.
        
        Args:
            img_array: RGB image array
            
        Returns:
            Dictionary with visual impact analysis
        """
        try:
            # Convert to grayscale for contrast analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate various contrast metrics
            
            # RMS contrast
            mean_val = np.mean(gray)
            rms_contrast = np.sqrt(np.mean((gray - mean_val) ** 2))
            rms_score = min(rms_contrast / 64.0, 1.0)  # Normalize
            
            # Michelson contrast
            min_val = np.min(gray)
            max_val = np.max(gray)
            if max_val + min_val > 0:
                michelson_contrast = (max_val - min_val) / (max_val + min_val)
            else:
                michelson_contrast = 0.0
            
            # Local contrast (edge strength)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (edges.size * 255)
            
            # Color contrast
            color_contrast = self._calculate_color_contrast(img_array)
            
            # Overall contrast score
            contrast_score = (rms_score + michelson_contrast + edge_density + color_contrast) / 4
            
            return {
                'contrast_score': float(contrast_score),
                'rms_contrast': float(rms_contrast),
                'michelson_contrast': float(michelson_contrast),
                'edge_density': float(edge_density),
                'color_contrast': float(color_contrast)
            }
            
        except Exception as e:
            logger.warning(f"Visual impact analysis error: {e}")
            return {
                'contrast_score': 0.5,
                'rms_contrast': 0.0,
                'michelson_contrast': 0.0,
                'edge_density': 0.0,
                'color_contrast': 0.5
            }
    
    def _calculate_color_contrast(self, img_array: np.ndarray) -> float:
        """
        Calculate color contrast in the image.
        
        Args:
            img_array: RGB image array
            
        Returns:
            Color contrast score (0.0-1.0)
        """
        try:
            # Convert to LAB color space for perceptual contrast
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Calculate color differences in LAB space
            lab_float = lab.astype(float)
            
            # Sample random pixel pairs for efficiency
            h, w = lab_float.shape[:2]
            n_samples = min(1000, h * w // 10)
            
            random_y1 = np.random.randint(0, h, n_samples)
            random_x1 = np.random.randint(0, w, n_samples)
            random_y2 = np.random.randint(0, h, n_samples)
            random_x2 = np.random.randint(0, w, n_samples)
            
            pixels1 = lab_float[random_y1, random_x1]
            pixels2 = lab_float[random_y2, random_x2]
            
            # Calculate Delta E color differences
            delta_e = np.sqrt(np.sum((pixels1 - pixels2) ** 2, axis=1))
            
            # Normalize and calculate score
            mean_delta_e = np.mean(delta_e)
            color_contrast_score = min(mean_delta_e / 100.0, 1.0)  # Normalize Delta E
            
            return float(color_contrast_score)
            
        except Exception as e:
            logger.warning(f"Color contrast calculation error: {e}")
            return 0.5
    
    def _analyze_saturation(self, img_array: np.ndarray) -> Dict:
        """
        Analyze saturation and vibrancy.
        
        Args:
            img_array: RGB image array
            
        Returns:
            Dictionary with saturation analysis
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            
            # Calculate saturation statistics
            mean_saturation = np.mean(saturation) / 255.0
            std_saturation = np.std(saturation) / 255.0
            
            # Calculate saturation distribution
            sat_hist, _ = np.histogram(saturation, bins=32, range=(0, 256))
            sat_hist_norm = sat_hist / np.sum(sat_hist)
            
            # Good saturation has moderate mean with some variation
            optimal_saturation = 0.6
            saturation_score = 1.0 - abs(mean_saturation - optimal_saturation) / optimal_saturation
            
            # Boost score if there's good variation
            if std_saturation > 0.1:
                saturation_score = min(saturation_score * 1.2, 1.0)
            
            # Check for oversaturation
            high_sat_pixels = np.sum(saturation > 200) / saturation.size
            if high_sat_pixels > 0.1:  # More than 10% highly saturated
                saturation_score *= 0.8  # Penalize oversaturation
            
            # Vibrancy calculation (saturation in already saturated areas)
            vibrancy = self._calculate_vibrancy(img_array)
            
            return {
                'saturation_score': float(saturation_score),
                'mean_saturation': float(mean_saturation),
                'saturation_std': float(std_saturation),
                'vibrancy': float(vibrancy),
                'high_saturation_percentage': float(high_sat_pixels)
            }
            
        except Exception as e:
            logger.warning(f"Saturation analysis error: {e}")
            return {
                'saturation_score': 0.5,
                'mean_saturation': 0.5,
                'saturation_std': 0.0,
                'vibrancy': 0.5,
                'high_saturation_percentage': 0.0
            }
    
    def _calculate_vibrancy(self, img_array: np.ndarray) -> float:
        """
        Calculate image vibrancy (smart saturation enhancement).
        
        Args:
            img_array: RGB image array
            
        Returns:
            Vibrancy score (0.0-1.0)
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1].astype(float) / 255.0
            
            # Vibrancy selectively enhances less saturated colors
            # and protects already highly saturated colors
            
            # Create vibrancy mask (focus on mid-saturation areas)
            vibrancy_mask = np.where(saturation < 0.5, 1.0 - saturation, 0.5)
            
            # Calculate weighted saturation
            vibrancy_weighted_sat = saturation * vibrancy_mask
            vibrancy_score = np.mean(vibrancy_weighted_sat) * 2  # Scale up
            
            return min(vibrancy_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Vibrancy calculation error: {e}")
            return 0.5
    
    def _analyze_mood(self, img_array: np.ndarray) -> Dict:
        """
        Analyze mood and atmosphere of the image.
        
        Args:
            img_array: RGB image array
            
        Returns:
            Dictionary with mood analysis
        """
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Analyze brightness for mood
            mean_brightness = np.mean(gray) / 255.0
            
            # Analyze color temperature for mood
            avg_r = np.mean(img_array[:, :, 0])
            avg_b = np.mean(img_array[:, :, 2])
            warmth = (avg_r - avg_b) / 255.0
            
            # Analyze saturation for mood
            mean_saturation = np.mean(hsv[:, :, 1]) / 255.0
            
            # Determine mood characteristics
            mood_scores = {}
            
            # Bright and cheerful
            if mean_brightness > 0.6 and warmth > 0.1:
                mood_scores['cheerful'] = 0.8
                mood_scores['energetic'] = 0.7
            
            # Dark and moody
            elif mean_brightness < 0.4:
                mood_scores['dramatic'] = 0.8
                mood_scores['mysterious'] = 0.7
            
            # Cool and calm
            elif warmth < -0.1:
                mood_scores['calm'] = 0.8
                mood_scores['serene'] = 0.7
            
            # Warm and cozy
            elif warmth > 0.1:
                mood_scores['warm'] = 0.8
                mood_scores['cozy'] = 0.7
            
            # Vibrant and lively
            if mean_saturation > 0.6:
                mood_scores['vibrant'] = 0.9
                mood_scores['lively'] = 0.8
            
            # Muted and subtle
            elif mean_saturation < 0.3:
                mood_scores['subtle'] = 0.8
                mood_scores['elegant'] = 0.7
            
            # Calculate overall mood score
            mood_score = np.mean(list(mood_scores.values())) if mood_scores else 0.5
            
            return {
                'mood_score': float(mood_score),
                'mood_characteristics': mood_scores,
                'brightness_level': float(mean_brightness),
                'warmth_level': float(warmth),
                'saturation_level': float(mean_saturation)
            }
            
        except Exception as e:
            logger.warning(f"Mood analysis error: {e}")
            return {
                'mood_score': 0.5,
                'mood_characteristics': {},
                'brightness_level': 0.5,
                'warmth_level': 0.0,
                'saturation_level': 0.5
            }
    
    def _calculate_aesthetic_score(self, analysis: Dict) -> float:
        """
        Calculate overall aesthetic score.
        
        Args:
            analysis: Dictionary containing all aesthetic analysis results
            
        Returns:
            Overall aesthetic score (0.0-1.0)
        """
        try:
            # Weight factors for different aesthetic aspects
            weights = {
                'color_harmony': 0.30,
                'contrast': 0.25,
                'saturation': 0.20,
                'mood': 0.25
            }
            
            # Get individual scores
            color_harmony = analysis.get('color_harmony', 0.5)
            contrast_score = analysis.get('contrast_score', 0.5)
            saturation_score = analysis.get('saturation_score', 0.5)
            mood_score = analysis.get('mood_score', 0.5)
            
            # Calculate weighted aesthetic score
            aesthetic_score = (
                color_harmony * weights['color_harmony'] +
                contrast_score * weights['contrast'] +
                saturation_score * weights['saturation'] +
                mood_score * weights['mood']
            )
            
            return max(0.0, min(1.0, aesthetic_score))
            
        except Exception as e:
            logger.warning(f"Aesthetic score calculation error: {e}")
            return 0.5