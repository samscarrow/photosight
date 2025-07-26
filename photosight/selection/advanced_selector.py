"""
Advanced photo selection system.

Provides sophisticated photo selection capabilities with multiple criteria
and filtering options for curating photo collections.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class AdvancedSelector:
    """
    Advanced photo selection system with multiple criteria support.
    
    Supports selection based on:
    - Technical quality metrics
    - Composition analysis
    - Subject detection and analysis
    - Aesthetic qualities
    - Custom criteria combinations
    """
    
    def __init__(self, config: Dict):
        """Initialize the advanced selector."""
        self.config = config
        self.selection_config = config.get('selection', {})
        
        # Default thresholds
        self.default_thresholds = {
            'sharpness': 100.0,
            'exposure_quality': 0.7,
            'composition_score': 0.6,
            'aesthetic_score': 0.6,
            'technical_quality': 0.7,
            'face_count_min': 1,
            'face_count_max': 10
        }
        
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
    
    def select_photos(self, photo_paths: List[Union[str, Path]], 
                     criteria: Dict[str, Any],
                     progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Select photos based on specified criteria.
        
        Args:
            photo_paths: List of photo file paths
            criteria: Dictionary of selection criteria
            progress_callback: Optional progress callback function
            
        Returns:
            List of selection results with photo info and selection status
        """
        results = []
        
        for i, photo_path in enumerate(photo_paths):
            try:
                result = self._evaluate_photo(Path(photo_path), criteria)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(photo_paths))
                    
            except Exception as e:
                logger.error(f"Error evaluating photo {photo_path}: {e}")
                results.append({
                    'file_path': str(photo_path),
                    'file_name': Path(photo_path).name,
                    'selected': False,
                    'error': str(e),
                    'reasons': [f'Evaluation error: {e}']
                })
        
        return results
    
    def _evaluate_photo(self, photo_path: Path, criteria: Dict[str, Any]) -> Dict:
        """
        Evaluate a single photo against selection criteria.
        
        Args:
            photo_path: Path to photo file
            criteria: Selection criteria dictionary
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Load image
            image = Image.open(photo_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Perform comprehensive analysis
            analysis = self._analyze_photo_comprehensive(img_array, photo_path)
            
            # Apply selection criteria
            selection_result = self._apply_selection_criteria(analysis, criteria)
            
            return {
                'file_path': str(photo_path),
                'file_name': photo_path.name,
                'selected': selection_result['selected'],
                'reasons': selection_result['reasons'],
                'analysis': analysis,
                'criteria_results': selection_result['criteria_results']
            }
            
        except Exception as e:
            logger.error(f"Error evaluating photo {photo_path}: {e}")
            return {
                'file_path': str(photo_path),
                'file_name': photo_path.name,
                'selected': False,
                'error': str(e),
                'reasons': [f'Evaluation error: {e}']
            }
    
    def _analyze_photo_comprehensive(self, img_array: np.ndarray, photo_path: Path) -> Dict:
        """
        Perform comprehensive photo analysis.
        
        Args:
            img_array: Image as numpy array
            photo_path: Path to photo file
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        analysis = {}
        
        try:
            # Technical analysis
            technical = self.technical_analyzer.analyze_photo(photo_path)
            analysis['technical'] = technical
            
            # Composition analysis
            composition = self.composition_analyzer.analyze_composition(img_array)
            analysis['composition'] = composition
            
            # Aesthetic analysis
            aesthetic = self.aesthetic_analyzer.analyze_aesthetics(img_array)
            analysis['aesthetic'] = aesthetic
            
            # Subject analysis
            subject = self.subject_analyzer.analyze_photo(img_array)
            analysis['subject'] = subject
            
            # Extract key metrics for easy access
            analysis['metrics'] = {
                'sharpness': technical.get('sharpness', 0),
                'exposure_quality': technical.get('exposure_quality', 0.5),
                'technical_quality': technical.get('technical_quality', 0.5),
                'composition_score': composition.get('composition_score', 0.5),
                'aesthetic_score': aesthetic.get('overall_appeal', 0.5),
                'face_count': subject.get('face_count', 0),
                'subject_clarity': subject.get('subject_clarity', 0.5),
                'noise_level': technical.get('noise_level', 0.5),
                'contrast_quality': technical.get('contrast_quality', 0.5),
                'color_harmony': aesthetic.get('color_harmony', 0.5),
                'rule_of_thirds_score': composition.get('rule_of_thirds_score', 0.5),
                'balance_score': composition.get('balance_score', 0.5)
            }
            
        except Exception as e:
            logger.warning(f"Analysis error for {photo_path}: {e}")
            analysis['error'] = str(e)
            analysis['metrics'] = {}
        
        return analysis
    
    def _apply_selection_criteria(self, analysis: Dict, criteria: Dict[str, Any]) -> Dict:
        """
        Apply selection criteria to photo analysis.
        
        Args:
            analysis: Photo analysis results
            criteria: Selection criteria
            
        Returns:
            Dictionary with selection results
        """
        selected = True
        reasons = []
        criteria_results = {}
        
        metrics = analysis.get('metrics', {})
        
        # Apply each criterion
        for criterion, threshold in criteria.items():
            try:
                result = self._evaluate_criterion(criterion, threshold, metrics, analysis)
                criteria_results[criterion] = result
                
                if not result['passed']:
                    selected = False
                    reasons.append(result['reason'])
                    
            except Exception as e:
                logger.warning(f"Error evaluating criterion {criterion}: {e}")
                criteria_results[criterion] = {
                    'passed': False,
                    'reason': f"Evaluation error: {e}",
                    'value': None
                }
                selected = False
                reasons.append(f"Error evaluating {criterion}")
        
        if selected and not reasons:
            reasons.append("Meets all selection criteria")
        
        return {
            'selected': selected,
            'reasons': reasons,
            'criteria_results': criteria_results
        }
    
    def _evaluate_criterion(self, criterion: str, threshold: Any, 
                          metrics: Dict, analysis: Dict) -> Dict:
        """
        Evaluate a single selection criterion.
        
        Args:
            criterion: Name of the criterion
            threshold: Threshold value or configuration
            metrics: Extracted metrics dictionary
            analysis: Full analysis results
            
        Returns:
            Dictionary with criterion evaluation result
        """
        
        # Handle different criterion types
        if criterion == 'sharpness':
            value = metrics.get('sharpness', 0)
            passed = value >= threshold
            reason = f"Sharpness {value:.1f} {'≥' if passed else '<'} {threshold}"
            
        elif criterion == 'exposure_quality':
            value = metrics.get('exposure_quality', 0.5)
            passed = value >= threshold
            reason = f"Exposure quality {value:.2f} {'≥' if passed else '<'} {threshold}"
            
        elif criterion == 'technical_quality':
            value = metrics.get('technical_quality', 0.5)
            passed = value >= threshold
            reason = f"Technical quality {value:.2f} {'≥' if passed else '<'} {threshold}"
            
        elif criterion == 'composition_score':
            value = metrics.get('composition_score', 0.5)
            passed = value >= threshold
            reason = f"Composition score {value:.2f} {'≥' if passed else '<'} {threshold}"
            
        elif criterion == 'aesthetic_score':
            value = metrics.get('aesthetic_score', 0.5)
            passed = value >= threshold
            reason = f"Aesthetic score {value:.2f} {'≥' if passed else '<'} {threshold}"
            
        elif criterion == 'face_count_min':
            value = metrics.get('face_count', 0)
            passed = value >= threshold
            reason = f"Face count {value} {'≥' if passed else '<'} {threshold}"
            
        elif criterion == 'face_count_max':
            value = metrics.get('face_count', 0)
            passed = value <= threshold
            reason = f"Face count {value} {'≤' if passed else '>'} {threshold}"
            
        elif criterion == 'face_count_exact':
            value = metrics.get('face_count', 0)
            passed = value == threshold
            reason = f"Face count {value} {'==' if passed else '!='} {threshold}"
            
        elif criterion == 'noise_level_max':
            value = metrics.get('noise_level', 0.5)
            passed = value <= threshold
            reason = f"Noise level {value:.2f} {'≤' if passed else '>'} {threshold}"
            
        elif criterion == 'contrast_quality':
            value = metrics.get('contrast_quality', 0.5)
            passed = value >= threshold
            reason = f"Contrast quality {value:.2f} {'≥' if passed else '<'} {threshold}"
            
        elif criterion == 'color_harmony':
            value = metrics.get('color_harmony', 0.5)
            passed = value >= threshold
            reason = f"Color harmony {value:.2f} {'≥' if passed else '<'} {threshold}"
            
        elif criterion == 'rule_of_thirds':
            value = metrics.get('rule_of_thirds_score', 0.5)
            passed = value >= threshold
            reason = f"Rule of thirds {value:.2f} {'≥' if passed else '<'} {threshold}"
            
        elif criterion == 'balance':
            value = metrics.get('balance_score', 0.5)
            passed = value >= threshold
            reason = f"Visual balance {value:.2f} {'≥' if passed else '<'} {threshold}"
            
        elif criterion == 'subject_clarity':
            value = metrics.get('subject_clarity', 0.5)
            passed = value >= threshold
            reason = f"Subject clarity {value:.2f} {'≥' if passed else '<'} {threshold}"
            
        elif criterion == 'has_faces':
            value = metrics.get('face_count', 0) > 0
            passed = value == threshold
            reason = f"Has faces: {value} {'==' if passed else '!='} {threshold}"
            
        elif criterion == 'orientation':
            # Check image orientation
            technical = analysis.get('technical', {})
            dimensions = technical.get('image_dimensions', {})
            width = dimensions.get('width', 1)
            height = dimensions.get('height', 1)
            
            if threshold == 'landscape':
                passed = width > height
                reason = f"Orientation: {'landscape' if passed else 'portrait'} (expected landscape)"
            elif threshold == 'portrait':
                passed = height > width
                reason = f"Orientation: {'portrait' if passed else 'landscape'} (expected portrait)"
            elif threshold == 'square':
                aspect_ratio = width / height
                passed = 0.9 <= aspect_ratio <= 1.1
                reason = f"Aspect ratio: {aspect_ratio:.2f} ({'square' if passed else 'not square'})"
            else:
                passed = True
                reason = f"Unknown orientation criteria: {threshold}"
            
            value = f"{width}x{height}"
            
        else:
            # Unknown criterion
            logger.warning(f"Unknown selection criterion: {criterion}")
            passed = True
            reason = f"Unknown criterion: {criterion}"
            value = None
        
        return {
            'passed': passed,
            'reason': reason,
            'value': value,
            'threshold': threshold
        }
    
    def create_selection_preset(self, name: str, criteria: Dict[str, Any], 
                              description: str = "") -> Dict:
        """
        Create a selection preset for reuse.
        
        Args:
            name: Preset name
            criteria: Selection criteria dictionary
            description: Optional description
            
        Returns:
            Preset dictionary
        """
        return {
            'name': name,
            'description': description,
            'criteria': criteria,
            'created_at': str(Path(__file__).stat().st_mtime)  # Placeholder timestamp
        }
    
    def get_default_presets(self) -> Dict[str, Dict]:
        """
        Get built-in selection presets.
        
        Returns:
            Dictionary of preset configurations
        """
        return {
            'high_quality': self.create_selection_preset(
                'High Quality',
                {
                    'sharpness': 150.0,
                    'exposure_quality': 0.8,
                    'technical_quality': 0.8,
                    'composition_score': 0.7,
                    'noise_level_max': 0.3
                },
                'Selects only high-quality photos with excellent technical merit'
            ),
            
            'portraits': self.create_selection_preset(
                'Portrait Photos',
                {
                    'face_count_min': 1,
                    'face_count_max': 5,
                    'sharpness': 120.0,
                    'subject_clarity': 0.7,
                    'orientation': 'portrait'
                },
                'Selects portrait photos with clear faces'
            ),
            
            'landscapes': self.create_selection_preset(
                'Landscape Photos',
                {
                    'orientation': 'landscape',
                    'composition_score': 0.7,
                    'rule_of_thirds': 0.6,
                    'technical_quality': 0.7,
                    'has_faces': False
                },
                'Selects landscape photos with good composition'
            ),
            
            'well_composed': self.create_selection_preset(
                'Well Composed',
                {
                    'composition_score': 0.8,
                    'rule_of_thirds': 0.7,
                    'balance': 0.7,
                    'aesthetic_score': 0.7
                },
                'Selects photos with excellent composition and aesthetic appeal'
            ),
            
            'technical_excellence': self.create_selection_preset(
                'Technical Excellence',
                {
                    'sharpness': 200.0,
                    'exposure_quality': 0.9,
                    'noise_level_max': 0.2,
                    'contrast_quality': 0.8,
                    'technical_quality': 0.9
                },
                'Selects technically perfect photos'
            ),
            
            'color_vibrant': self.create_selection_preset(
                'Vibrant Colors',
                {
                    'color_harmony': 0.8,
                    'aesthetic_score': 0.8,
                    'technical_quality': 0.7
                },
                'Selects photos with excellent color and vibrant appeal'
            )
        }
    
    def apply_preset(self, preset_name: str, photo_paths: List[Union[str, Path]], 
                    progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Apply a selection preset to photos.
        
        Args:
            preset_name: Name of the preset to apply
            photo_paths: List of photo file paths
            progress_callback: Optional progress callback
            
        Returns:
            List of selection results
        """
        presets = self.get_default_presets()
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        preset = presets[preset_name]
        criteria = preset['criteria']
        
        logger.info(f"Applying preset '{preset_name}': {preset['description']}")
        
        return self.select_photos(photo_paths, criteria, progress_callback)