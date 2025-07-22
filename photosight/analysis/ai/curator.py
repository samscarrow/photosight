"""
AI Curator - Main module integrating all AI analysis components
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from .person_detection import PersonDetector
from .face_analysis import FaceAnalyzer
from .composition import CompositionAnalyzer

logger = logging.getLogger(__name__)


class AICurator:
    """
    Main AI curator that integrates all analysis components
    for advanced photo curation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize AI Curator with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('ai_curation', {})
        
        # Initialize components based on config
        self.person_detector = None
        self.face_analyzer = None
        self.composition_analyzer = None
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize AI components based on configuration"""
        # Person detection
        if self.config.get('enable_person_detection', True):
            try:
                self.person_detector = PersonDetector(
                    model_size=self.config.get('yolo_model', 'yolov8n.pt'),
                    confidence_threshold=self.config.get('person_confidence', 0.5)
                )
                logger.info("Initialized person detector")
            except Exception as e:
                logger.error(f"Failed to initialize person detector: {e}")
                
        # Face analysis
        if self.config.get('enable_face_analysis', True):
            try:
                self.face_analyzer = FaceAnalyzer(
                    min_detection_confidence=self.config.get('face_confidence', 0.5)
                )
                logger.info("Initialized face analyzer")
            except Exception as e:
                logger.error(f"Failed to initialize face analyzer: {e}")
                
        # Composition analysis
        if self.config.get('enable_composition_analysis', True):
            try:
                self.composition_analyzer = CompositionAnalyzer()
                logger.info("Initialized composition analyzer")
            except Exception as e:
                logger.error(f"Failed to initialize composition analyzer: {e}")
                
    def analyze_image(self, image: np.ndarray, metadata: Dict = None) -> Dict:
        """
        Perform complete AI analysis on image
        
        Args:
            image: Image array (RGB)
            metadata: Optional metadata from camera
            
        Returns:
            Comprehensive analysis results
        """
        results = {
            'ai_score': 0.0,
            'ai_passed': False,
            'rejection_reasons': [],
            'person_analysis': None,
            'face_analysis': None,
            'composition_analysis': None,
            'recommendations': []
        }
        
        # Person detection
        if self.person_detector:
            try:
                person_results = self.person_detector.detect_people(image)
                results['person_analysis'] = person_results
                
                # Additional composition analysis with people
                if person_results['people_count'] > 0:
                    person_composition = self.person_detector.analyze_composition(
                        image, person_results
                    )
                    person_results['composition'] = person_composition
                    
            except Exception as e:
                logger.error(f"Person detection failed: {e}")
                
        # Face analysis
        if self.face_analyzer and results['person_analysis']:
            if results['person_analysis']['people_count'] > 0:
                try:
                    face_results = self.face_analyzer.analyze_faces(image)
                    results['face_analysis'] = face_results
                except Exception as e:
                    logger.error(f"Face analysis failed: {e}")
                    
        # Composition analysis
        if self.composition_analyzer:
            try:
                comp_results = self.composition_analyzer.analyze_composition(image)
                results['composition_analysis'] = comp_results
            except Exception as e:
                logger.error(f"Composition analysis failed: {e}")
                
        # Calculate overall AI score and make curation decision
        results['ai_score'] = self._calculate_ai_score(results)
        results['ai_passed'], results['rejection_reasons'] = self._make_curation_decision(results)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
        
    def _calculate_ai_score(self, results: Dict) -> float:
        """Calculate overall AI curation score"""
        score = 0.0
        weights_sum = 0.0
        
        # Person detection score
        if results['person_analysis']:
            person_score = 0.0
            
            # Scene type scoring
            scene_type = results['person_analysis']['scene_type']
            if scene_type == 'portrait':
                person_score += 0.4
            elif scene_type == 'small_group':
                person_score += 0.3
            elif scene_type == 'no_people':
                person_score += 0.2  # Landscape/architecture can be good too
                
            # Composition with people
            if 'composition' in results['person_analysis']:
                comp = results['person_analysis']['composition']
                if comp['rule_of_thirds']:
                    person_score += 0.3
                elif comp['people_centered']:
                    person_score += 0.2
                if comp['balanced']:
                    person_score += 0.1
                    
            score += person_score * self.config.get('person_weight', 0.3)
            weights_sum += self.config.get('person_weight', 0.3)
            
        # Face analysis score
        if results['face_analysis'] and results['face_analysis']['face_count'] > 0:
            face_score = results['face_analysis']['overall_quality']
            
            # Bonus for good expressions
            if results['face_analysis']['has_eyes_open']:
                face_score = min(1.0, face_score + 0.1)
            if results['face_analysis']['has_smiles']:
                face_score = min(1.0, face_score + 0.1)
                
            score += face_score * self.config.get('face_weight', 0.3)
            weights_sum += self.config.get('face_weight', 0.3)
            
        # Composition score
        if results['composition_analysis']:
            comp_score = results['composition_analysis']['composition_score']
            score += comp_score * self.config.get('composition_weight', 0.4)
            weights_sum += self.config.get('composition_weight', 0.4)
            
        # Normalize by weights
        if weights_sum > 0:
            score = score / weights_sum
            
        return min(1.0, score)
        
    def _make_curation_decision(self, results: Dict) -> Tuple[bool, List[str]]:
        """Make final curation decision based on AI analysis"""
        rejection_reasons = []
        
        # Check thresholds
        min_score = self.config.get('min_ai_score', 0.5)
        
        if results['ai_score'] < min_score:
            rejection_reasons.append('low_ai_score')
            
        # Specific checks for people photos
        if results['person_analysis'] and results['person_analysis']['people_count'] > 0:
            # Face analysis checks only when faces are actually detected
            if results['face_analysis'] and results['face_analysis']['face_count'] > 0:
                # Reject if eyes closed in portrait
                if (results['person_analysis']['scene_type'] == 'portrait' and 
                    not results['face_analysis']['has_eyes_open']):
                    rejection_reasons.append('eyes_closed')
                    
                # Reject if face quality too low
                if results['face_analysis']['overall_quality'] < 0.3:
                    rejection_reasons.append('poor_face_quality')
                    
        # Composition checks
        if results['composition_analysis']:
            comp = results['composition_analysis']
            
            # Reject if severely unbalanced
            if comp['visual_balance']['balance_score'] < 0.3:
                rejection_reasons.append('poor_balance')
                
        passed = len(rejection_reasons) == 0
        
        return passed, rejection_reasons
        
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate improvement recommendations based on analysis"""
        recommendations = []
        
        # Person-based recommendations
        if results['person_analysis']:
            if results['person_analysis']['scene_type'] == 'portrait':
                main_subject = results['person_analysis'].get('main_subject')
                if main_subject:
                    if main_subject['size_ratio'] < 0.1:
                        recommendations.append("Move closer to subject for stronger portrait")
                    if main_subject['position'][1] > 0.7:
                        recommendations.append("Consider centering subject vertically")
                        
        # Face recommendations
        if results['face_analysis'] and results['face_analysis']['face_count'] > 0:
            if not results['face_analysis']['has_eyes_open']:
                recommendations.append("Ensure all subjects have eyes open")
            if not results['face_analysis']['has_smiles']:
                recommendations.append("Capture genuine expressions or smiles")
                
        # Composition recommendations
        if results['composition_analysis']:
            comp = results['composition_analysis']
            
            if not comp['rule_of_thirds']['follows_rule']:
                recommendations.append("Try placing subjects on rule of thirds lines")
                
            if comp['visual_balance']['balance_score'] < 0.5:
                recommendations.append("Improve visual balance in composition")
                
            if not comp['leading_lines']['has_leading_lines']:
                recommendations.append("Look for leading lines to guide the eye")
                
        return recommendations