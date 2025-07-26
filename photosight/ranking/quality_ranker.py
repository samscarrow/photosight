"""
Quality-based photo ranking system.

Implements comprehensive photo quality assessment using multiple AI models
and technical analysis to provide accurate photo rankings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image
import json

logger = logging.getLogger(__name__)


class QualityRanker:
    """
    Advanced photo quality ranking system.
    
    Combines multiple quality assessment techniques:
    - Technical quality (sharpness, exposure, noise)
    - Composition analysis (rule of thirds, leading lines)
    - AI-powered aesthetic scoring
    - YOLO-based subject detection and analysis
    """
    
    def __init__(self, config: Dict):
        """Initialize the quality ranker with configuration."""
        self.config = config
        self.ranking_config = config.get('ranking', {})
        
        # Weight factors for different quality aspects
        self.weights = self.ranking_config.get('weights', {
            'technical': 0.3,
            'composition': 0.25,
            'aesthetic': 0.25,
            'subject': 0.2
        })
        
        # Initialize analyzers lazily
        self._technical_analyzer = None
        self._composition_analyzer = None
        self._aesthetic_analyzer = None
        self._subject_analyzer = None
        
    @property
    def technical_analyzer(self):
        """Lazy loading of technical analyzer."""
        if self._technical_analyzer is None:
            from ..analysis.technical_analyzer import TechnicalAnalyzer
            self._technical_analyzer = TechnicalAnalyzer(self.config)
        return self._technical_analyzer
    
    @property
    def composition_analyzer(self):
        """Lazy loading of composition analyzer."""
        if self._composition_analyzer is None:
            from ..analysis.composition_analyzer import CompositionAnalyzer
            self._composition_analyzer = CompositionAnalyzer(self.config)
        return self._composition_analyzer
    
    @property
    def aesthetic_analyzer(self):
        """Lazy loading of aesthetic analyzer."""
        if self._aesthetic_analyzer is None:
            from ..analysis.aesthetic_analyzer import AestheticAnalyzer
            self._aesthetic_analyzer = AestheticAnalyzer(self.config)
        return self._aesthetic_analyzer
    
    @property
    def subject_analyzer(self):
        """Lazy loading of subject analyzer."""
        if self._subject_analyzer is None:
            from ..processing.production_yolo_blur_processor import ProductionYOLOBlurProcessor
            self._subject_analyzer = ProductionYOLOBlurProcessor(self.config)
        return self._subject_analyzer
    
    def rank_photo(self, photo_path: Union[str, Path]) -> float:
        """
        Rank a single photo and return a quality score (0.0-1.0).
        
        Args:
            photo_path: Path to the photo file
            
        Returns:
            Quality score between 0.0 and 1.0 (higher is better)
        """
        photo_path = Path(photo_path)
        
        if not photo_path.exists():
            logger.error(f"Photo not found: {photo_path}")
            return 0.0
        
        try:
            # Load image for analysis
            image = Image.open(photo_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            scores = {}
            
            # Technical quality analysis
            try:
                technical_score = self._analyze_technical_quality(image, photo_path)
                scores['technical'] = technical_score
            except Exception as e:
                logger.warning(f"Technical analysis failed for {photo_path}: {e}")
                scores['technical'] = 0.5  # Default score
            
            # Composition analysis
            try:
                composition_score = self._analyze_composition(image)
                scores['composition'] = composition_score
            except Exception as e:
                logger.warning(f"Composition analysis failed for {photo_path}: {e}")
                scores['composition'] = 0.5
            
            # Aesthetic analysis
            try:
                aesthetic_score = self._analyze_aesthetics(image)
                scores['aesthetic'] = aesthetic_score
            except Exception as e:
                logger.warning(f"Aesthetic analysis failed for {photo_path}: {e}")
                scores['aesthetic'] = 0.5
            
            # Subject analysis
            try:
                subject_score = self._analyze_subjects(image, photo_path)
                scores['subject'] = subject_score
            except Exception as e:
                logger.warning(f"Subject analysis failed for {photo_path}: {e}")
                scores['subject'] = 0.5
            
            # Calculate weighted overall score
            overall_score = sum(
                scores[aspect] * self.weights[aspect]
                for aspect in scores.keys()
                if aspect in self.weights
            )
            
            # Ensure score is in valid range
            overall_score = max(0.0, min(1.0, overall_score))
            
            logger.debug(f"Ranked {photo_path.name}: {overall_score:.3f} "
                        f"(T:{scores['technical']:.2f}, C:{scores['composition']:.2f}, "
                        f"A:{scores['aesthetic']:.2f}, S:{scores['subject']:.2f})")
            
            return overall_score
            
        except Exception as e:
            logger.error(f"Error ranking photo {photo_path}: {e}")
            return 0.0
    
    def rank_photos_batch(self, photo_paths: List[Union[str, Path]], 
                         progress_callback: Optional[callable] = None) -> List[Tuple[Path, float]]:
        """
        Rank multiple photos and return sorted results.
        
        Args:
            photo_paths: List of photo file paths
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of (path, score) tuples sorted by score (descending)
        """
        results = []
        
        for i, photo_path in enumerate(photo_paths):
            score = self.rank_photo(photo_path)
            results.append((Path(photo_path), score))
            
            if progress_callback:
                progress_callback(i + 1, len(photo_paths))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _analyze_technical_quality(self, image: Image.Image, photo_path: Path) -> float:
        """Analyze technical quality aspects."""
        try:
            analysis = self.technical_analyzer.analyze_photo(photo_path)
            
            # Combine technical metrics
            sharpness = analysis.get('sharpness', 0) / 1000.0  # Normalize
            sharpness = min(1.0, sharpness)  # Cap at 1.0
            
            exposure_quality = analysis.get('exposure_quality', 0.5)
            noise_quality = 1.0 - analysis.get('noise_level', 0.5)  # Invert noise
            
            # Weight technical aspects
            technical_score = (
                sharpness * 0.4 +
                exposure_quality * 0.4 +
                noise_quality * 0.2
            )
            
            return max(0.0, min(1.0, technical_score))
            
        except Exception as e:
            logger.warning(f"Technical analysis error: {e}")
            return 0.5
    
    def _analyze_composition(self, image: Image.Image) -> float:
        """Analyze composition quality."""
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            analysis = self.composition_analyzer.analyze_composition(img_array)
            
            # Extract composition scores
            rule_of_thirds = analysis.get('rule_of_thirds_score', 0.5)
            balance = analysis.get('balance_score', 0.5)
            leading_lines = analysis.get('leading_lines_score', 0.5)
            symmetry = analysis.get('symmetry_score', 0.5)
            
            # Weight composition aspects
            composition_score = (
                rule_of_thirds * 0.3 +
                balance * 0.3 +
                leading_lines * 0.2 +
                symmetry * 0.2
            )
            
            return max(0.0, min(1.0, composition_score))
            
        except Exception as e:
            logger.warning(f"Composition analysis error: {e}")
            return 0.5
    
    def _analyze_aesthetics(self, image: Image.Image) -> float:
        """Analyze aesthetic quality using AI models."""
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            analysis = self.aesthetic_analyzer.analyze_aesthetics(img_array)
            
            # Extract aesthetic scores
            color_harmony = analysis.get('color_harmony', 0.5)
            contrast = analysis.get('contrast_score', 0.5)
            saturation = analysis.get('saturation_score', 0.5)
            overall_appeal = analysis.get('overall_appeal', 0.5)
            
            # Weight aesthetic aspects
            aesthetic_score = (
                color_harmony * 0.3 +
                contrast * 0.2 +
                saturation * 0.2 +
                overall_appeal * 0.3
            )
            
            return max(0.0, min(1.0, aesthetic_score))
            
        except Exception as e:
            logger.warning(f"Aesthetic analysis error: {e}")
            return 0.5
    
    def _analyze_subjects(self, image: Image.Image, photo_path: Path) -> float:
        """Analyze subject quality and focus."""
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            analysis = self.subject_analyzer.analyze_photo(img_array)
            
            # Extract subject scores
            subject_clarity = analysis.get('subject_clarity', 0.5)
            subject_prominence = analysis.get('subject_prominence', 0.5)
            background_quality = analysis.get('background_blur_quality', 0.5)
            face_quality = analysis.get('face_quality', 0.5)
            
            # Adjust weights based on photo type
            if analysis.get('has_faces', False):
                # Portrait-style weighting
                subject_score = (
                    subject_clarity * 0.3 +
                    face_quality * 0.4 +
                    background_quality * 0.2 +
                    subject_prominence * 0.1
                )
            else:
                # General photo weighting
                subject_score = (
                    subject_clarity * 0.4 +
                    subject_prominence * 0.3 +
                    background_quality * 0.3
                )
            
            return max(0.0, min(1.0, subject_score))
            
        except Exception as e:
            logger.warning(f"Subject analysis error: {e}")
            return 0.5
    
    def get_detailed_analysis(self, photo_path: Union[str, Path]) -> Dict:
        """
        Get detailed quality analysis for a photo.
        
        Returns:
            Dictionary with detailed scores and analysis
        """
        photo_path = Path(photo_path)
        
        try:
            image = Image.open(photo_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get detailed analysis from each component
            technical = self.technical_analyzer.analyze_photo(photo_path)
            composition = self.composition_analyzer.analyze_composition(np.array(image))
            aesthetic = self.aesthetic_analyzer.analyze_aesthetics(np.array(image))
            subject = self.subject_analyzer.analyze_photo(np.array(image))
            
            # Calculate component scores
            technical_score = self._analyze_technical_quality(image, photo_path)
            composition_score = self._analyze_composition(image)
            aesthetic_score = self._analyze_aesthetics(image)
            subject_score = self._analyze_subjects(image, photo_path)
            
            # Overall score
            overall_score = sum(
                score * self.weights[aspect]
                for aspect, score in [
                    ('technical', technical_score),
                    ('composition', composition_score),
                    ('aesthetic', aesthetic_score),
                    ('subject', subject_score)
                ]
            )
            
            return {
                'overall_score': overall_score,
                'component_scores': {
                    'technical': technical_score,
                    'composition': composition_score,
                    'aesthetic': aesthetic_score,
                    'subject': subject_score
                },
                'detailed_analysis': {
                    'technical': technical,
                    'composition': composition,
                    'aesthetic': aesthetic,
                    'subject': subject
                },
                'weights': self.weights,
                'file_path': str(photo_path),
                'file_name': photo_path.name
            }
            
        except Exception as e:
            logger.error(f"Error in detailed analysis for {photo_path}: {e}")
            return {
                'overall_score': 0.0,
                'error': str(e),
                'file_path': str(photo_path),
                'file_name': photo_path.name
            }