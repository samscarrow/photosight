#!/usr/bin/env python3
"""
Full analysis and ranking for enneagram workshop photos.

Uses PhotoSight's performance-optimized pipeline to:
- Analyze all photos with vision LLM
- Extract technical and aesthetic qualities
- Rank photos by multiple criteria
- Generate comprehensive report
"""

import sys
import os
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# PhotoSight imports - only what's available
from photosight.analysis.vision_llm_analyzer import VisionLLMAnalyzer
from photosight.analysis.aesthetic_analyzer import AestheticAnalyzer
from photosight.analysis.technical_analyzer import TechnicalAnalyzer
from photosight.ranking.quality_ranker import QualityRanker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enneagram_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnneagramAnalyzer:
    """Comprehensive analyzer for enneagram workshop photos."""
    
    def __init__(self, output_dir: Path):
        """Initialize the analyzer with output directory."""
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.analysis_results = {}
        self.rankings = {}
        self.processing_stats = {
            'total_photos': 0,
            'processed': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Initialize configuration
        self.config = self._create_config()
        
        # Initialize analyzers
        logger.info("Initializing analyzers...")
        self.vision_analyzer = VisionLLMAnalyzer(self.config)
        self.aesthetic_analyzer = AestheticAnalyzer(self.config)
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        self.quality_ranker = QualityRanker(self.config)
        
        # Cache for performance
        self._cache = {}
    
    def _create_config(self) -> Dict:
        """Create configuration for analyzers."""
        return {
            'vision_llm': {
                'enabled': True,
                'provider': 'gemini',
                'gemini': {
                    'model': 'gemini-1.5-flash',
                    'temperature': 0.4,
                    'max_output_tokens': 1024,
                    'safety_settings': 'medium'
                },
                'processing': {
                    'use_previews': True,
                    'preview_size': 1024,
                    'cache_results': True,
                    'timeout': 30
                }
            },
            'aesthetic': {
                'color_analysis': {'enabled': True},
                'mood_detection': {'enabled': True},
                'composition_rules': {'enabled': True}
            },
            'technical': {
                'sharpness_analysis': {'enabled': True},
                'exposure_analysis': {'enabled': True},
                'noise_analysis': {'enabled': True}
            },
            'ranking': {
                'weights': {
                    'technical': 0.3,
                    'aesthetic': 0.4,
                    'emotional': 0.3
                }
            }
        }
    
    async def analyze_photo(self, photo_path: Path) -> Dict:
        """
        Perform comprehensive analysis on a single photo.
        
        Args:
            photo_path: Path to photo file
            
        Returns:
            Dictionary with all analysis results
        """
        try:
            # Check cache first
            cache_key = str(photo_path)
            if cache_key in self._cache:
                logger.debug(f"Using cached results for {photo_path.name}")
                return self._cache[cache_key]
            
            logger.info(f"Analyzing {photo_path.name}...")
            
            results = {
                'filename': photo_path.name,
                'path': str(photo_path),
                'timestamp': datetime.now().isoformat(),
                'errors': []
            }
            
            # Vision LLM Analysis
            try:
                # Scene analysis
                scene_result = self.vision_analyzer.analyze_scene(str(photo_path))
                results['scene'] = {
                    'type': scene_result.get('scene_type', 'unknown'),
                    'environment': scene_result.get('environment', 'unknown'),
                    'confidence': scene_result.get('confidence', 0)
                }
                
                # Composition analysis
                comp_result = self.vision_analyzer.analyze_composition(str(photo_path))
                results['composition'] = {
                    'balance_score': comp_result.get('balance_score', 0),
                    'leading_lines': comp_result.get('leading_lines', False),
                    'rule_of_thirds': comp_result.get('rule_of_thirds_alignment', 0),
                    'depth': comp_result.get('depth_perception', 'unknown')
                }
                
                # Quality assessment
                quality_result = self.vision_analyzer.assess_quality(str(photo_path))
                results['vision_quality'] = {
                    'technical_score': quality_result.get('technical_score', 0),
                    'artistic_score': quality_result.get('artistic_score', 0),
                    'emotional_impact': quality_result.get('emotional_impact', 0),
                    'overall_score': quality_result.get('overall_score', 0)
                }
                
                # Decisive moment
                moment_result = self.vision_analyzer.detect_moment(str(photo_path))
                results['decisive_moment'] = {
                    'is_decisive': moment_result.get('is_decisive_moment', False),
                    'confidence': moment_result.get('confidence', 0),
                    'reason': moment_result.get('reason', '')
                }
                
            except Exception as e:
                logger.warning(f"Vision LLM analysis failed for {photo_path.name}: {e}")
                results['errors'].append(f"Vision analysis: {str(e)}")
            
            # Technical Analysis
            try:
                tech_result = self.technical_analyzer.analyze(str(photo_path))
                results['technical'] = {
                    'sharpness': tech_result.get('sharpness_score', 0),
                    'exposure': tech_result.get('exposure_quality', 0),
                    'noise': tech_result.get('noise_level', 0),
                    'overall_technical': tech_result.get('overall_technical_score', 0)
                }
            except Exception as e:
                logger.warning(f"Technical analysis failed for {photo_path.name}: {e}")
                results['errors'].append(f"Technical analysis: {str(e)}")
                results['technical'] = {
                    'sharpness': 0.5,
                    'exposure': 0.5,
                    'noise': 0.5,
                    'overall_technical': 0.5
                }
            
            # Aesthetic Analysis
            try:
                aesthetic_result = self.aesthetic_analyzer.analyze(str(photo_path))
                results['aesthetic'] = {
                    'color_harmony': aesthetic_result.get('color_harmony_score', 0),
                    'visual_appeal': aesthetic_result.get('visual_appeal_score', 0),
                    'mood': aesthetic_result.get('dominant_mood', 'neutral'),
                    'overall_aesthetic': aesthetic_result.get('overall_aesthetic_score', 0)
                }
            except Exception as e:
                logger.warning(f"Aesthetic analysis failed for {photo_path.name}: {e}")
                results['errors'].append(f"Aesthetic analysis: {str(e)}")
                results['aesthetic'] = {
                    'color_harmony': 0.5,
                    'visual_appeal': 0.5,
                    'mood': 'unknown',
                    'overall_aesthetic': 0.5
                }
            
            # Calculate combined scores
            results['scores'] = self._calculate_combined_scores(results)
            
            # Cache results
            self._cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze {photo_path.name}: {e}")
            return {
                'filename': photo_path.name,
                'path': str(photo_path),
                'error': str(e),
                'scores': {
                    'overall': 0,
                    'technical': 0,
                    'artistic': 0,
                    'emotional': 0
                }
            }
    
    def _calculate_combined_scores(self, results: Dict) -> Dict:
        """Calculate combined scores from all analyses."""
        # Technical score (average of sharpness, exposure, noise)
        technical_score = (
            results.get('technical', {}).get('overall_technical', 0) * 0.5 +
            results.get('vision_quality', {}).get('technical_score', 0) * 0.5
        )
        
        # Artistic score (composition + aesthetic)
        artistic_score = (
            results.get('composition', {}).get('balance_score', 0) * 0.3 +
            results.get('aesthetic', {}).get('overall_aesthetic', 0) * 0.4 +
            results.get('vision_quality', {}).get('artistic_score', 0) * 0.3
        )
        
        # Emotional impact score
        emotional_score = results.get('vision_quality', {}).get('emotional_impact', 0)
        
        # Decisive moment bonus
        if results.get('decisive_moment', {}).get('is_decisive', False):
            emotional_score = min(1.0, emotional_score + 0.2)
        
        # Overall score with configured weights
        weights = self.config['ranking']['weights']
        overall_score = (
            technical_score * weights['technical'] +
            artistic_score * weights['aesthetic'] +
            emotional_score * weights['emotional']
        )
        
        return {
            'overall': overall_score,
            'technical': technical_score,
            'artistic': artistic_score,
            'emotional': emotional_score
        }
    
    async def process_directory(self, photo_dir: Path, limit: Optional[int] = None):
        """
        Process all photos in a directory.
        
        Args:
            photo_dir: Directory containing photos
            limit: Optional limit on number of photos to process
        """
        # Get all photo files
        photo_files = list(photo_dir.glob("*.jpg")) + list(photo_dir.glob("*.jpeg"))
        photo_files.sort()
        
        if limit:
            photo_files = photo_files[:limit]
        
        self.processing_stats['total_photos'] = len(photo_files)
        self.processing_stats['start_time'] = time.time()
        
        logger.info(f"Found {len(photo_files)} photos to process")
        
        # Process in batches for better performance
        batch_size = 5
        for i in range(0, len(photo_files), batch_size):
            batch = photo_files[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.analyze_photo(photo) for photo in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            for photo, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {photo.name}: {result}")
                    self.processing_stats['failed'] += 1
                else:
                    self.analysis_results[photo.name] = result
                    self.processing_stats['processed'] += 1
            
            # Progress update
            progress = (i + len(batch)) / len(photo_files) * 100
            logger.info(f"Progress: {progress:.1f}% ({i + len(batch)}/{len(photo_files)})")
        
        self.processing_stats['end_time'] = time.time()
        
        # Generate rankings
        self._generate_rankings()
    
    def _generate_rankings(self):
        """Generate rankings based on analysis results."""
        logger.info("Generating rankings...")
        
        # Create ranking lists
        rankings = {
            'overall': [],
            'technical': [],
            'artistic': [],
            'emotional': [],
            'decisive_moments': []
        }
        
        for filename, results in self.analysis_results.items():
            if 'scores' in results:
                entry = {
                    'filename': filename,
                    'scores': results['scores'],
                    'scene': results.get('scene', {}).get('type', 'unknown'),
                    'mood': results.get('aesthetic', {}).get('mood', 'unknown')
                }
                
                rankings['overall'].append(entry)
                rankings['technical'].append(entry)
                rankings['artistic'].append(entry)
                rankings['emotional'].append(entry)
                
                # Decisive moments
                if results.get('decisive_moment', {}).get('is_decisive', False):
                    rankings['decisive_moments'].append({
                        'filename': filename,
                        'confidence': results['decisive_moment']['confidence'],
                        'reason': results['decisive_moment']['reason']
                    })
        
        # Sort rankings
        rankings['overall'].sort(key=lambda x: x['scores']['overall'], reverse=True)
        rankings['technical'].sort(key=lambda x: x['scores']['technical'], reverse=True)
        rankings['artistic'].sort(key=lambda x: x['scores']['artistic'], reverse=True)
        rankings['emotional'].sort(key=lambda x: x['scores']['emotional'], reverse=True)
        rankings['decisive_moments'].sort(key=lambda x: x['confidence'], reverse=True)
        
        self.rankings = rankings
    
    def save_results(self):
        """Save all results to files."""
        logger.info("Saving results...")
        
        # Save detailed analysis results
        results_file = self.output_dir / 'enneagram_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_photos': self.processing_stats['total_photos'],
                    'processed': self.processing_stats['processed'],
                    'failed': self.processing_stats['failed'],
                    'processing_time': self.processing_stats['end_time'] - self.processing_stats['start_time']
                },
                'results': self.analysis_results,
                'rankings': self.rankings
            }, f, indent=2)
        
        # Save rankings as CSV for easy viewing
        rankings_csv = self.output_dir / 'enneagram_rankings.csv'
        with open(rankings_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Filename', 'Overall Score', 'Technical', 'Artistic', 'Emotional', 'Scene', 'Mood'])
            
            for i, entry in enumerate(self.rankings['overall'][:50], 1):  # Top 50
                writer.writerow([
                    i,
                    entry['filename'],
                    f"{entry['scores']['overall']:.3f}",
                    f"{entry['scores']['technical']:.3f}",
                    f"{entry['scores']['artistic']:.3f}",
                    f"{entry['scores']['emotional']:.3f}",
                    entry['scene'],
                    entry['mood']
                ])
        
        # Save decisive moments
        if self.rankings['decisive_moments']:
            moments_file = self.output_dir / 'enneagram_decisive_moments.json'
            with open(moments_file, 'w') as f:
                json.dump(self.rankings['decisive_moments'], f, indent=2)
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate a human-readable summary report."""
        report_file = self.output_dir / 'enneagram_analysis_report.txt'
        
        processing_time = self.processing_stats['end_time'] - self.processing_stats['start_time']
        
        with open(report_file, 'w') as f:
            f.write("PHOTOSIGHT ENNEAGRAM WORKSHOP ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Photos Analyzed: {self.processing_stats['processed']}\n")
            f.write(f"Failed: {self.processing_stats['failed']}\n")
            f.write(f"Processing Time: {processing_time:.1f} seconds\n")
            f.write(f"Average Time per Photo: {processing_time/max(self.processing_stats['processed'], 1):.2f} seconds\n\n")
            
            # Top 10 Overall
            f.write("TOP 10 PHOTOS BY OVERALL SCORE\n")
            f.write("-" * 60 + "\n")
            for i, entry in enumerate(self.rankings['overall'][:10], 1):
                f.write(f"{i:2d}. {entry['filename']:<30} Score: {entry['scores']['overall']:.3f}\n")
                f.write(f"    Technical: {entry['scores']['technical']:.3f} | ")
                f.write(f"Artistic: {entry['scores']['artistic']:.3f} | ")
                f.write(f"Emotional: {entry['scores']['emotional']:.3f}\n")
                f.write(f"    Scene: {entry['scene']} | Mood: {entry['mood']}\n\n")
            
            # Decisive Moments
            if self.rankings['decisive_moments']:
                f.write("\nDECISIVE MOMENTS CAPTURED\n")
                f.write("-" * 60 + "\n")
                for i, moment in enumerate(self.rankings['decisive_moments'][:5], 1):
                    f.write(f"{i}. {moment['filename']} (Confidence: {moment['confidence']:.2f})\n")
                    f.write(f"   {moment['reason']}\n\n")
            
            # Scene Distribution
            f.write("\nSCENE TYPE DISTRIBUTION\n")
            f.write("-" * 60 + "\n")
            scene_counts = defaultdict(int)
            for result in self.analysis_results.values():
                scene = result.get('scene', {}).get('type', 'unknown')
                scene_counts[scene] += 1
            
            for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{scene:<20} {count:3d} photos\n")
            
            # Average Scores
            f.write("\nAVERAGE SCORES ACROSS ALL PHOTOS\n")
            f.write("-" * 60 + "\n")
            
            avg_scores = {'overall': 0, 'technical': 0, 'artistic': 0, 'emotional': 0}
            count = 0
            
            for result in self.analysis_results.values():
                if 'scores' in result:
                    for key in avg_scores:
                        avg_scores[key] += result['scores'][key]
                    count += 1
            
            if count > 0:
                for key in avg_scores:
                    avg_scores[key] /= count
                    f.write(f"{key.capitalize():<15} {avg_scores[key]:.3f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Analysis complete. See CSV file for detailed rankings.\n")
    
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*60)
        print("ENNEAGRAM WORKSHOP ANALYSIS COMPLETE")
        print("="*60)
        
        print(f"\n📊 Processing Statistics:")
        print(f"   Total Photos: {self.processing_stats['total_photos']}")
        print(f"   Processed: {self.processing_stats['processed']}")
        print(f"   Failed: {self.processing_stats['failed']}")
        
        if self.processing_stats['end_time'] and self.processing_stats['start_time']:
            duration = self.processing_stats['end_time'] - self.processing_stats['start_time']
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Speed: {self.processing_stats['processed']/duration:.1f} photos/second")
        
        print(f"\n🏆 Top 5 Photos:")
        for i, entry in enumerate(self.rankings['overall'][:5], 1):
            print(f"   {i}. {entry['filename']} (Score: {entry['scores']['overall']:.3f})")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"   - enneagram_analysis_results.json (full details)")
        print(f"   - enneagram_rankings.csv (sortable rankings)")
        print(f"   - enneagram_analysis_report.txt (summary report)")
        
        if self.rankings['decisive_moments']:
            print(f"   - enneagram_decisive_moments.json ({len(self.rankings['decisive_moments'])} moments)")


async def main():
    """Run the full enneagram analysis."""
    # Check for photos directory
    photo_dir = Path("/Users/sam/Desktop/photosight_output/enneagram_workshop/accepted")
    
    if not photo_dir.exists():
        print(f"❌ Photo directory not found: {photo_dir}")
        print("Please update the path to your enneagram photos.")
        return
    
    # Create output directory
    output_dir = Path("enneagram_analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = EnneagramAnalyzer(output_dir)
    
    # Process photos (limit to first 20 for demo, remove limit for full analysis)
    await analyzer.process_directory(photo_dir, limit=20)
    
    # Save results
    analyzer.save_results()
    
    # Print summary
    analyzer.print_summary()


if __name__ == "__main__":
    asyncio.run(main())