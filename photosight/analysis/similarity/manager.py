"""
Main similarity management system for PhotoSight pipeline integration
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
from .detector import SimilarityDetector
from .selector import DuplicateSelector

logger = logging.getLogger(__name__)


class SimilarityManager:
    """
    Main interface for similarity detection and duplicate management
    Integrates with the PhotoSight pipeline for intelligent photo curation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize similarity manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('similarity', {})
        
        # Initialize components
        self.detector = SimilarityDetector(
            similarity_threshold=self.config.get('similarity_threshold', 0.99),
            hash_size=self.config.get('hash_size', 16),
            enable_metadata_check=self.config.get('enable_metadata_check', True)
        )
        
        self.selector = DuplicateSelector(config)
        
        # Configuration flags
        self.enabled = self.config.get('enable_similarity_detection', True)
        self.auto_reject_duplicates = self.config.get('auto_reject_duplicates', True)
        self.create_duplicate_links = self.config.get('create_duplicate_links', False)
        
        logger.info(f"SimilarityManager initialized: enabled={self.enabled}, "
                   f"auto_reject={self.auto_reject_duplicates}")
    
    def analyze_batch(self, 
                     image_paths: List[Path], 
                     analysis_results: Dict[Path, Dict]) -> Dict[str, any]:
        """
        Analyze a batch of images for similarity and select best representatives
        
        Args:
            image_paths: List of image paths to analyze
            analysis_results: Existing analysis results for the images
            
        Returns:
            Dictionary with similarity analysis results
        """
        if not self.enabled or len(image_paths) < 2:
            return {
                'similarity_groups': [],
                'selected_images': image_paths,
                'duplicate_images': [],
                'statistics': {}
            }
            
        logger.info(f"Starting similarity analysis for {len(image_paths)} images")
        
        # Step 1: Find similarity groups
        similarity_groups = self.detector.find_similar_groups(image_paths)
        
        if not similarity_groups:
            logger.info("No similar image groups found")
            return {
                'similarity_groups': [],
                'selected_images': image_paths,
                'duplicate_images': [],
                'statistics': {}
            }
        
        # Step 2: Select best images from each group
        selection_results = self.selector.process_similarity_groups(
            similarity_groups, analysis_results
        )
        
        # Step 3: Create final image lists
        # Images not in any similarity group remain selected
        images_in_groups = set()
        for group in similarity_groups:
            images_in_groups.update(group)
            
        images_not_in_groups = [p for p in image_paths if p not in images_in_groups]
        
        final_selected = selection_results['selected'] + images_not_in_groups
        final_duplicates = selection_results['duplicates']
        
        # Step 4: Generate statistics
        statistics = self.selector.get_duplicate_statistics(selection_results)
        statistics.update({
            'total_input_images': len(image_paths),
            'images_not_in_groups': len(images_not_in_groups),
            'final_selected_count': len(final_selected),
            'final_duplicate_count': len(final_duplicates)
        })
        
        logger.info(f"Similarity analysis complete:")
        logger.info(f"  Input images: {len(image_paths)}")
        logger.info(f"  Similarity groups: {len(similarity_groups)}")
        logger.info(f"  Final selected: {len(final_selected)}")
        logger.info(f"  Marked as duplicates: {len(final_duplicates)}")
        logger.info(f"  Reduction: {statistics.get('reduction_percentage', 0):.1f}%")
        
        return {
            'similarity_groups': similarity_groups,
            'selected_images': final_selected,
            'duplicate_images': final_duplicates,
            'statistics': statistics,
            'selection_results': selection_results
        }
    
    def create_duplicate_report(self, 
                               similarity_results: Dict[str, any],
                               output_dir: Path) -> Path:
        """
        Create detailed report of similarity detection results
        
        Args:
            similarity_results: Results from analyze_batch
            output_dir: Directory to write report
            
        Returns:
            Path to the created report file
        """
        report_path = output_dir / 'similarity_report.json'
        
        # Create comprehensive report
        report = {
            'summary': similarity_results['statistics'],
            'similarity_groups': [],
            'configuration': {
                'similarity_threshold': self.detector.similarity_threshold,
                'hash_size': self.detector.hash_size,
                'selection_weights': self.selector.weights
            }
        }
        
        # Add detailed group information
        for i, group in enumerate(similarity_results['similarity_groups']):
            group_info = {
                'group_id': i,
                'image_count': len(group),
                'images': [str(path) for path in group]
            }
            
            # Add selection information if available
            selection_results = similarity_results.get('selection_results', {})
            for sel_group in selection_results.get('groups', []):
                if sel_group['group_id'] == i:
                    group_info.update({
                        'selected': sel_group['selected'],
                        'rejected': sel_group['rejected']
                    })
                    break
                    
            report['similarity_groups'].append(group_info)
        
        # Write report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Detailed similarity report written to {report_path}")
        return report_path
    
    def integrate_with_pipeline(self, 
                              accepted_images: List[Path],
                              analysis_results: Dict[Path, Dict],
                              output_structure: Dict[str, Path]) -> Tuple[List[Path], Dict[str, List[Path]]]:
        """
        Integrate similarity detection with PhotoSight pipeline
        
        Args:
            accepted_images: List of images that passed initial curation
            analysis_results: Analysis results for all images
            output_structure: Output directory structure
            
        Returns:
            Tuple of (final_selected_images, categorized_duplicates)
        """
        if not self.enabled:
            return accepted_images, {}
            
        logger.info("Integrating similarity detection with PhotoSight pipeline")
        
        # Analyze accepted images for duplicates
        similarity_results = self.analyze_batch(accepted_images, analysis_results)
        
        # Create duplicate report
        if 'reports' in output_structure:
            self.create_duplicate_report(similarity_results, output_structure['reports'])
        
        final_selected = similarity_results['selected_images']
        duplicates = similarity_results['duplicate_images']
        
        # Categorize duplicates
        categorized_duplicates = {}
        if self.auto_reject_duplicates and duplicates:
            # Move duplicates to a separate category
            categorized_duplicates['duplicates'] = duplicates
            logger.info(f"Auto-rejecting {len(duplicates)} duplicate images")
        else:
            # Keep duplicates in a review category
            categorized_duplicates['duplicates_review'] = duplicates
            logger.info(f"Marking {len(duplicates)} images for duplicate review")
        
        return final_selected, categorized_duplicates
    
    def create_duplicate_links(self, 
                             similarity_groups: List[List[Path]], 
                             output_dir: Path) -> None:
        """
        Create symbolic links grouping duplicate images for review
        
        Args:
            similarity_groups: Groups of similar images
            output_dir: Output directory for link organization
        """
        if not self.create_duplicate_links:
            return
            
        links_dir = output_dir / 'duplicate_groups'
        links_dir.mkdir(parents=True, exist_ok=True)
        
        for i, group in enumerate(similarity_groups):
            group_dir = links_dir / f'group_{i+1:03d}'
            group_dir.mkdir(exist_ok=True)
            
            for j, image_path in enumerate(group):
                link_name = f"{j+1:02d}_{image_path.name}"
                link_path = group_dir / link_name
                
                try:
                    if not link_path.exists():
                        link_path.symlink_to(image_path)
                except OSError as e:
                    logger.warning(f"Could not create symlink {link_path}: {e}")
        
        logger.info(f"Created duplicate group links in {links_dir}")
    
    def get_similarity_summary(self, similarity_results: Dict[str, any]) -> str:
        """
        Get a human-readable summary of similarity detection results
        
        Args:
            similarity_results: Results from analyze_batch
            
        Returns:
            Formatted summary string
        """
        stats = similarity_results['statistics']
        
        summary_lines = [
            "=== Similarity Detection Summary ===",
            f"Total input images: {stats.get('total_input_images', 0)}",
            f"Similarity groups found: {stats.get('similarity_groups_found', 0)}",
            f"Images in similarity groups: {stats.get('images_in_groups', 0)}",
            f"Final selected images: {stats.get('final_selected_count', 0)}",
            f"Duplicate images: {stats.get('final_duplicate_count', 0)}",
            f"Reduction percentage: {stats.get('reduction_percentage', 0):.1f}%",
            f"Average group size: {stats.get('average_group_size', 0):.1f}",
        ]
        
        return '\n'.join(summary_lines)