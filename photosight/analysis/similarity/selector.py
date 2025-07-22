"""
Intelligent selection of the best image from duplicate groups
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DuplicateSelector:
    """
    Select the best image from groups of similar/duplicate photos
    using technical quality, AI analysis, and smart heuristics
    """
    
    def __init__(self, config: Dict):
        """
        Initialize duplicate selector
        
        Args:
            config: Configuration dictionary with selection criteria
        """
        self.config = config.get('similarity', {})
        
        # Selection weights
        self.weights = {
            'technical_quality': self.config.get('technical_weight', 0.4),
            'ai_score': self.config.get('ai_weight', 0.3), 
            'sharpness': self.config.get('sharpness_weight', 0.2),
            'preference_heuristics': self.config.get('heuristics_weight', 0.1)
        }
        
        logger.info(f"Duplicate selector initialized with weights: {self.weights}")
        
    def select_best_from_group(self, 
                              group: List[Path], 
                              analysis_results: Dict[Path, Dict]) -> Tuple[Path, List[Path]]:
        """
        Select the best image from a group of similar photos
        
        Args:
            group: List of similar image paths
            analysis_results: Dictionary mapping paths to their analysis results
            
        Returns:
            Tuple of (best_image_path, rejected_images_paths)
        """
        if len(group) <= 1:
            return group[0] if group else None, []
            
        logger.debug(f"Selecting best from group of {len(group)} similar images")
        
        # Score each image
        scores = {}
        for path in group:
            scores[path] = self._compute_selection_score(path, analysis_results.get(path, {}))
            
        # Sort by score (highest first)
        sorted_images = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        best_image = sorted_images[0][0]
        rejected_images = [img for img, _ in sorted_images[1:]]
        
        logger.debug(f"Selected {best_image.name} (score: {scores[best_image]:.3f}) "
                    f"from {len(group)} candidates")
                    
        # Log detailed scoring for top candidates
        for i, (path, score) in enumerate(sorted_images[:min(3, len(sorted_images))]):
            logger.debug(f"  #{i+1}: {path.name} - score: {score:.3f}")
            
        return best_image, rejected_images
    
    def _compute_selection_score(self, image_path: Path, analysis: Dict) -> float:
        """
        Compute overall selection score for an image
        
        Args:
            image_path: Path to the image
            analysis: Analysis results for the image
            
        Returns:
            Selection score (higher is better)
        """
        score = 0.0
        
        # Technical quality score
        technical_score = self._get_technical_score(analysis)
        score += technical_score * self.weights['technical_quality']
        
        # AI curation score  
        ai_score = analysis.get('ai_score', 0.0)
        score += ai_score * self.weights['ai_score']
        
        # Sharpness/focus quality
        sharpness_score = self._get_sharpness_score(analysis)
        score += sharpness_score * self.weights['sharpness']
        
        # Preference heuristics (middle shots, expression quality, etc.)
        heuristic_score = self._get_heuristic_score(image_path, analysis)
        score += heuristic_score * self.weights['preference_heuristics']
        
        return score
    
    def _get_technical_score(self, analysis: Dict) -> float:
        """Get normalized technical quality score"""
        technical = analysis.get('technical_analysis', {})
        if not technical:
            return 0.5  # Neutral score if no analysis
            
        score = 0.0
        components = 0
        
        # ISO performance (lower is better for noise)
        if 'iso' in technical:
            iso_score = min(1.0, 6400 / max(technical['iso'], 100))  # Normalize around ISO 6400
            score += iso_score
            components += 1
            
        # Exposure quality
        if 'exposure_quality' in technical:
            score += technical['exposure_quality']
            components += 1
            
        # Focus quality
        if 'focus_quality' in technical:
            score += technical['focus_quality'] 
            components += 1
            
        return score / max(components, 1)
    
    def _get_sharpness_score(self, analysis: Dict) -> float:
        """Get normalized sharpness score"""
        # From blur detection analysis
        blur_analysis = analysis.get('blur_analysis', {})
        if 'sharpness_score' in blur_analysis:
            return blur_analysis['sharpness_score']
            
        # From face analysis sharpness
        face_analysis = analysis.get('face_analysis', {})
        if 'faces' in face_analysis and face_analysis['faces']:
            face_sharpness = np.mean([face['sharpness'] for face in face_analysis['faces']])
            return face_sharpness
            
        # Fallback to laplacian variance if available
        if 'laplacian_variance' in analysis:
            # Normalize around typical sharp photo variance
            return min(1.0, analysis['laplacian_variance'] / 100.0)
            
        return 0.5  # Neutral if no sharpness data
    
    def _get_heuristic_score(self, image_path: Path, analysis: Dict) -> float:
        """
        Get score based on selection heuristics and preferences
        
        Common patterns:
        - Middle shots in a burst are often better (avoid start/end clicks)
        - Better expressions (eyes open, smiling)
        - Good composition scores
        - Avoid motion blur in people photos
        """
        score = 0.5  # Start neutral
        
        # Expression quality for people photos
        face_analysis = analysis.get('face_analysis', {})
        if face_analysis.get('face_count', 0) > 0:
            # Prefer photos with eyes open
            if face_analysis.get('has_eyes_open', False):
                score += 0.2
                
            # Prefer photos with good expressions
            if face_analysis.get('good_expressions', False):
                score += 0.1
                
            # Prefer photos with smiles in appropriate contexts
            if face_analysis.get('has_smiles', False):
                score += 0.1
        
        # Composition quality
        comp_analysis = analysis.get('composition_analysis', {})
        if comp_analysis:
            rule_of_thirds = comp_analysis.get('rule_of_thirds', {})
            if rule_of_thirds.get('score', 0) > 0.7:
                score += 0.1
                
        # Motion blur penalty for people photos
        if face_analysis.get('face_count', 0) > 0:
            blur_analysis = analysis.get('blur_analysis', {})
            if blur_analysis.get('motion_blur_detected', False):
                score -= 0.2
        
        # File naming heuristics (prefer middle shots in sequences)
        score += self._get_sequence_preference_score(image_path)
        
        return np.clip(score, 0.0, 1.0)
    
    def _get_sequence_preference_score(self, image_path: Path) -> float:
        """
        Get preference score based on position in filename sequence
        
        In burst shots, middle images are often better than first/last
        """
        try:
            filename = image_path.stem
            # Extract sequence number (e.g., DSC04123 -> 4123)
            sequence_num = int(''.join(filter(str.isdigit, filename)))
            
            # For burst sequences, we'd ideally know the full sequence
            # For now, slightly prefer even numbers (arbitrary heuristic)
            if sequence_num % 2 == 0:
                return 0.05
            else:
                return 0.0
                
        except:
            return 0.0
    
    def process_similarity_groups(self, 
                                 similarity_groups: List[List[Path]], 
                                 analysis_results: Dict[Path, Dict]) -> Dict[str, List[Path]]:
        """
        Process all similarity groups and categorize results
        
        Args:
            similarity_groups: List of groups of similar images
            analysis_results: Analysis results for all images
            
        Returns:
            Dictionary with 'selected', 'duplicates', and 'groups' keys
        """
        selected_images = []
        duplicate_images = []
        group_info = []
        
        for i, group in enumerate(similarity_groups):
            best_image, rejected_images = self.select_best_from_group(group, analysis_results)
            
            if best_image:
                selected_images.append(best_image)
                duplicate_images.extend(rejected_images)
                
                group_info.append({
                    'group_id': i,
                    'selected': str(best_image),
                    'rejected': [str(p) for p in rejected_images],
                    'count': len(group)
                })
        
        logger.info(f"Processed {len(similarity_groups)} similarity groups:")
        logger.info(f"  Selected: {len(selected_images)} images")
        logger.info(f"  Duplicates: {len(duplicate_images)} images")
        
        return {
            'selected': selected_images,
            'duplicates': duplicate_images, 
            'groups': group_info
        }
    
    def create_selection_report(self, 
                              results: Dict[str, List[Path]], 
                              output_path: Path) -> None:
        """
        Create detailed report of similarity detection and selection
        
        Args:
            results: Results from process_similarity_groups
            output_path: Path to write report JSON
        """
        report = {
            'summary': {
                'total_groups': len(results['groups']),
                'selected_images': len(results['selected']),
                'duplicate_images': len(results['duplicates']),
                'space_saved': len(results['duplicates'])
            },
            'groups': results['groups'],
            'selection_criteria': {
                'weights': self.weights,
                'similarity_threshold': self.config.get('similarity_threshold', 0.99)
            }
        }
        
        # Convert Path objects to strings for JSON serialization
        report_json = json.dumps(report, indent=2, default=str)
        
        output_path.write_text(report_json)
        logger.info(f"Selection report written to {output_path}")
    
    def get_duplicate_statistics(self, results: Dict[str, List[Path]]) -> Dict[str, any]:
        """Get statistics about duplicate detection results"""
        total_images_in_groups = sum(group['count'] for group in results['groups'])
        
        return {
            'similarity_groups_found': len(results['groups']),
            'images_in_groups': total_images_in_groups,
            'images_selected': len(results['selected']),
            'images_marked_duplicate': len(results['duplicates']),
            'reduction_percentage': len(results['duplicates']) / max(total_images_in_groups, 1) * 100,
            'average_group_size': total_images_in_groups / max(len(results['groups']), 1)
        }