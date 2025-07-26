"""
Photo collection statistics generator.

Generates comprehensive statistics and reports for photo collections
including technical metrics, composition analysis, and collection insights.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter
from PIL import Image, ExifTags
import csv

logger = logging.getLogger(__name__)


class StatsGenerator:
    """
    Generates comprehensive statistics for photo collections.
    
    Provides insights into:
    - Technical quality metrics
    - Composition and aesthetic analysis
    - EXIF data statistics
    - Collection metadata
    - Quality distribution
    - Recommendations for improvement
    """
    
    def __init__(self, config: Dict):
        """Initialize the stats generator."""
        self.config = config
        self.stats_config = config.get('stats', {})
        
        # Initialize analyzers lazily
        self._quality_ranker = None
        self._technical_analyzer = None
        self._composition_analyzer = None
        self._aesthetic_analyzer = None
    
    @property
    def quality_ranker(self):
        """Lazy loading of quality ranker."""
        if self._quality_ranker is None:
            from ..ranking.quality_ranker import QualityRanker
            self._quality_ranker = QualityRanker(self.config)
        return self._quality_ranker
    
    @property
    def technical_analyzer(self):
        """Lazy loading of technical analyzer."""
        if self._technical_analyzer is None:
            from .technical_analyzer import TechnicalAnalyzer
            self._technical_analyzer = TechnicalAnalyzer(self.config)
        return self._technical_analyzer
    
    @property
    def composition_analyzer(self):
        """Lazy loading of composition analyzer."""
        if self._composition_analyzer is None:
            from .composition_analyzer import CompositionAnalyzer
            self._composition_analyzer = CompositionAnalyzer(self.config)
        return self._composition_analyzer
    
    @property
    def aesthetic_analyzer(self):
        """Lazy loading of aesthetic analyzer."""
        if self._aesthetic_analyzer is None:
            from .aesthetic_analyzer import AestheticAnalyzer
            self._aesthetic_analyzer = AestheticAnalyzer(self.config)
        return self._aesthetic_analyzer
    
    def generate_stats(self, photo_paths: List[Union[str, Path]], 
                      include_exif: bool = True, 
                      include_analysis: bool = True,
                      group_by: Optional[str] = None,
                      progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for photo collection.
        
        Args:
            photo_paths: List of photo file paths
            include_exif: Whether to include EXIF data analysis
            include_analysis: Whether to include quality analysis
            group_by: Grouping criteria ('date', 'camera', 'rating', 'none')
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary containing comprehensive statistics
        """
        try:
            stats = {
                'collection_info': self._get_collection_info(photo_paths),
                'file_stats': self._analyze_file_stats(photo_paths),
                'generated_at': datetime.now().isoformat()
            }
            
            # EXIF analysis
            if include_exif:
                logger.info("Analyzing EXIF data...")
                stats['exif_stats'] = self._analyze_exif_data(photo_paths, progress_callback)
            
            # Quality analysis
            if include_analysis:
                logger.info("Performing quality analysis...")
                stats['quality_stats'] = self._analyze_quality(photo_paths, progress_callback)
            
            # Grouping analysis
            if group_by and group_by != 'none':
                logger.info(f"Generating grouped statistics by {group_by}...")
                stats['grouped_stats'] = self._generate_grouped_stats(photo_paths, group_by)
            
            # Generate recommendations
            stats['recommendations'] = self._generate_recommendations(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat(),
                'collection_info': {'total_photos': len(photo_paths)}
            }
    
    def _get_collection_info(self, photo_paths: List[Union[str, Path]]) -> Dict:
        """Get basic collection information."""
        try:
            total_photos = len(photo_paths)
            
            # Analyze file extensions
            extensions = Counter()
            total_size = 0
            
            for photo_path in photo_paths:
                path = Path(photo_path)
                extensions[path.suffix.lower()] += 1
                try:
                    total_size += path.stat().st_size
                except OSError:
                    pass  # File might not exist
            
            return {
                'total_photos': total_photos,
                'file_extensions': dict(extensions),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'average_file_size_mb': round(total_size / (1024 * 1024 * total_photos), 2) if total_photos > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Error getting collection info: {e}")
            return {'total_photos': len(photo_paths), 'error': str(e)}
    
    def _analyze_file_stats(self, photo_paths: List[Union[str, Path]]) -> Dict:
        """Analyze file-level statistics."""
        try:
            file_stats = {
                'creation_dates': [],
                'modification_dates': [],
                'file_sizes': [],
                'directories': Counter()
            }
            
            for photo_path in photo_paths:
                try:
                    path = Path(photo_path)
                    stat = path.stat()
                    
                    file_stats['creation_dates'].append(stat.st_ctime)
                    file_stats['modification_dates'].append(stat.st_mtime)
                    file_stats['file_sizes'].append(stat.st_size)
                    file_stats['directories'][str(path.parent)] += 1
                    
                except OSError as e:
                    logger.debug(f"Could not stat {photo_path}: {e}")
            
            # Calculate statistics
            if file_stats['file_sizes']:
                file_stats['size_stats'] = {
                    'min_mb': round(min(file_stats['file_sizes']) / (1024 * 1024), 2),
                    'max_mb': round(max(file_stats['file_sizes']) / (1024 * 1024), 2),
                    'avg_mb': round(np.mean(file_stats['file_sizes']) / (1024 * 1024), 2),
                    'median_mb': round(np.median(file_stats['file_sizes']) / (1024 * 1024), 2)
                }
            
            # Date range
            if file_stats['creation_dates']:
                creation_times = [datetime.fromtimestamp(t) for t in file_stats['creation_dates']]
                file_stats['date_range'] = {
                    'earliest': min(creation_times).isoformat(),
                    'latest': max(creation_times).isoformat(),
                    'span_days': (max(creation_times) - min(creation_times)).days
                }
            
            # Top directories
            file_stats['top_directories'] = dict(file_stats['directories'].most_common(10))
            
            # Clean up raw data
            del file_stats['creation_dates']
            del file_stats['modification_dates']
            del file_stats['file_sizes']
            del file_stats['directories']
            
            return file_stats
            
        except Exception as e:
            logger.warning(f"Error analyzing file stats: {e}")
            return {'error': str(e)}
    
    def _analyze_exif_data(self, photo_paths: List[Union[str, Path]], 
                          progress_callback: Optional[callable] = None) -> Dict:
        """Analyze EXIF data across the collection."""
        try:
            exif_stats = {
                'cameras': Counter(),
                'lenses': Counter(),
                'focal_lengths': [],
                'apertures': [],
                'shutter_speeds': [],
                'iso_values': [],
                'flash_usage': Counter(),
                'orientation': Counter(),
                'shooting_modes': Counter()
            }
            
            processed = 0
            
            for i, photo_path in enumerate(photo_paths):
                try:
                    exif_data = self._extract_exif_data(Path(photo_path))
                    
                    if exif_data:
                        # Camera information
                        if exif_data.get('make') and exif_data.get('model'):
                            camera = f"{exif_data['make']} {exif_data['model']}"
                            exif_stats['cameras'][camera] += 1
                        
                        # Lens information
                        if exif_data.get('lens_model'):
                            exif_stats['lenses'][exif_data['lens_model']] += 1
                        
                        # Technical settings
                        if exif_data.get('focal_length'):
                            exif_stats['focal_lengths'].append(exif_data['focal_length'])
                        
                        if exif_data.get('aperture'):
                            exif_stats['apertures'].append(exif_data['aperture'])
                        
                        if exif_data.get('shutter_speed'):
                            exif_stats['shutter_speeds'].append(exif_data['shutter_speed'])
                        
                        if exif_data.get('iso'):
                            exif_stats['iso_values'].append(exif_data['iso'])
                        
                        # Flash and orientation
                        if exif_data.get('flash'):
                            exif_stats['flash_usage'][exif_data['flash']] += 1
                        
                        if exif_data.get('orientation'):
                            exif_stats['orientation'][exif_data['orientation']] += 1
                        
                        if exif_data.get('shooting_mode'):
                            exif_stats['shooting_modes'][exif_data['shooting_mode']] += 1
                        
                        processed += 1
                    
                    if progress_callback:
                        progress_callback(i + 1, len(photo_paths))
                        
                except Exception as e:
                    logger.debug(f"Error extracting EXIF from {photo_path}: {e}")
            
            # Calculate statistics for numerical values
            exif_stats['statistics'] = {
                'photos_with_exif': processed,
                'exif_coverage': round(processed / len(photo_paths), 2) if photo_paths else 0
            }
            
            if exif_stats['focal_lengths']:
                exif_stats['focal_length_stats'] = {
                    'min': min(exif_stats['focal_lengths']),
                    'max': max(exif_stats['focal_lengths']),
                    'avg': round(np.mean(exif_stats['focal_lengths']), 1),
                    'most_common': Counter(exif_stats['focal_lengths']).most_common(5)
                }
            
            if exif_stats['apertures']:
                exif_stats['aperture_stats'] = {
                    'min': min(exif_stats['apertures']),
                    'max': max(exif_stats['apertures']),
                    'avg': round(np.mean(exif_stats['apertures']), 1),
                    'most_common': Counter(exif_stats['apertures']).most_common(5)
                }
            
            if exif_stats['iso_values']:
                exif_stats['iso_stats'] = {
                    'min': min(exif_stats['iso_values']),
                    'max': max(exif_stats['iso_values']),
                    'avg': round(np.mean(exif_stats['iso_values']), 0),
                    'most_common': Counter(exif_stats['iso_values']).most_common(5)
                }
            
            # Convert counters to dictionaries and get top items
            exif_stats['top_cameras'] = dict(exif_stats['cameras'].most_common(10))
            exif_stats['top_lenses'] = dict(exif_stats['lenses'].most_common(10))
            
            # Clean up raw lists
            del exif_stats['focal_lengths']
            del exif_stats['apertures']
            del exif_stats['shutter_speeds']
            del exif_stats['iso_values']
            
            return exif_stats
            
        except Exception as e:
            logger.warning(f"Error analyzing EXIF data: {e}")
            return {'error': str(e)}
    
    def _extract_exif_data(self, photo_path: Path) -> Optional[Dict]:
        """Extract EXIF data from a photo."""
        try:
            with Image.open(photo_path) as image:
                exif = image.getexif()
                
                if not exif:
                    return None
                
                exif_data = {}
                
                # Map EXIF tags to human-readable names
                for tag_id in exif:
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    value = exif.get(tag_id)
                    
                    if tag == 'Make':
                        exif_data['make'] = str(value).strip()
                    elif tag == 'Model':
                        exif_data['model'] = str(value).strip()
                    elif tag == 'LensModel':
                        exif_data['lens_model'] = str(value).strip()
                    elif tag == 'FocalLength':
                        if isinstance(value, tuple) and len(value) == 2:
                            exif_data['focal_length'] = round(value[0] / value[1], 1)
                        else:
                            exif_data['focal_length'] = float(value)
                    elif tag == 'FNumber':
                        if isinstance(value, tuple) and len(value) == 2:
                            exif_data['aperture'] = round(value[0] / value[1], 1)
                        else:
                            exif_data['aperture'] = float(value)
                    elif tag == 'ExposureTime':
                        if isinstance(value, tuple) and len(value) == 2:
                            exif_data['shutter_speed'] = f"1/{int(value[1]/value[0])}"
                        else:
                            exif_data['shutter_speed'] = str(value)
                    elif tag == 'ISOSpeedRatings':
                        exif_data['iso'] = int(value)
                    elif tag == 'Flash':
                        flash_modes = {0: 'No Flash', 1: 'Flash', 16: 'No Flash', 24: 'Flash'}
                        exif_data['flash'] = flash_modes.get(value, f'Flash Mode {value}')
                    elif tag == 'Orientation':
                        orientations = {1: 'Normal', 3: 'Rotated 180°', 6: 'Rotated 90° CW', 8: 'Rotated 90° CCW'}
                        exif_data['orientation'] = orientations.get(value, f'Orientation {value}')
                    elif tag == 'ExposureMode':
                        exposure_modes = {0: 'Auto', 1: 'Manual', 2: 'Auto Bracket'}
                        exif_data['shooting_mode'] = exposure_modes.get(value, f'Mode {value}')
                
                return exif_data
                
        except Exception as e:
            logger.debug(f"Error extracting EXIF from {photo_path}: {e}")
            return None
    
    def _analyze_quality(self, photo_paths: List[Union[str, Path]], 
                        progress_callback: Optional[callable] = None) -> Dict:
        """Analyze quality metrics across the collection."""
        try:
            quality_scores = []
            quality_categories = Counter()
            
            for i, photo_path in enumerate(photo_paths):
                try:
                    score = self.quality_ranker.rank_photo(photo_path)
                    quality_scores.append(score)
                    
                    # Categorize quality
                    if score >= 0.9:
                        category = 'Excellent'
                    elif score >= 0.8:
                        category = 'Very Good'
                    elif score >= 0.7:
                        category = 'Good'
                    elif score >= 0.6:
                        category = 'Fair'
                    elif score >= 0.5:
                        category = 'Poor'
                    else:
                        category = 'Very Poor'
                    
                    quality_categories[category] += 1
                    
                    if progress_callback:
                        progress_callback(i + 1, len(photo_paths))
                        
                except Exception as e:
                    logger.debug(f"Error analyzing quality for {photo_path}: {e}")
                    quality_scores.append(0.5)  # Default score
            
            if not quality_scores:
                return {'error': 'No photos could be analyzed'}
            
            # Calculate statistics
            quality_stats = {
                'average_quality': round(np.mean(quality_scores), 3),
                'median_quality': round(np.median(quality_scores), 3),
                'min_quality': round(min(quality_scores), 3),
                'max_quality': round(max(quality_scores), 3),
                'std_quality': round(np.std(quality_scores), 3),
                'quality_distribution': dict(quality_categories),
                'percentiles': {
                    '25th': round(np.percentile(quality_scores, 25), 3),
                    '75th': round(np.percentile(quality_scores, 75), 3),
                    '90th': round(np.percentile(quality_scores, 90), 3),
                    '95th': round(np.percentile(quality_scores, 95), 3)
                }
            }
            
            # Calculate percentage distributions
            total_photos = len(quality_scores)
            quality_stats['quality_percentages'] = {
                category: round(count / total_photos * 100, 1)
                for category, count in quality_categories.items()
            }
            
            return quality_stats
            
        except Exception as e:
            logger.warning(f"Error analyzing quality: {e}")
            return {'error': str(e)}
    
    def _generate_grouped_stats(self, photo_paths: List[Union[str, Path]], 
                               group_by: str) -> Dict:
        """Generate statistics grouped by specified criteria."""
        try:
            groups = defaultdict(list)
            
            for photo_path in photo_paths:
                group_key = self._get_group_key(Path(photo_path), group_by)
                groups[group_key].append(photo_path)
            
            grouped_stats = {}
            
            for group_name, group_photos in groups.items():
                group_stats = {
                    'count': len(group_photos),
                    'percentage': round(len(group_photos) / len(photo_paths) * 100, 1)
                }
                
                # Add quality analysis for each group if possible
                try:
                    quality_scores = []
                    for photo in group_photos[:50]:  # Limit for performance
                        score = self.quality_ranker.rank_photo(photo)
                        quality_scores.append(score)
                    
                    if quality_scores:
                        group_stats['avg_quality'] = round(np.mean(quality_scores), 3)
                        group_stats['quality_range'] = [round(min(quality_scores), 3), 
                                                       round(max(quality_scores), 3)]
                
                except Exception as e:
                    logger.debug(f"Error calculating quality for group {group_name}: {e}")
                
                grouped_stats[group_name] = group_stats
            
            return grouped_stats
            
        except Exception as e:
            logger.warning(f"Error generating grouped stats: {e}")
            return {'error': str(e)}
    
    def _get_group_key(self, photo_path: Path, group_by: str) -> str:
        """Get the grouping key for a photo based on criteria."""
        try:
            if group_by == 'date':
                # Try to get date from EXIF, fall back to file date
                exif_data = self._extract_exif_data(photo_path)
                if exif_data and 'date_taken' in exif_data:
                    return exif_data['date_taken'][:10]  # YYYY-MM-DD
                else:
                    # Use file modification date
                    timestamp = photo_path.stat().st_mtime
                    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            
            elif group_by == 'camera':
                exif_data = self._extract_exif_data(photo_path)
                if exif_data and exif_data.get('make') and exif_data.get('model'):
                    return f"{exif_data['make']} {exif_data['model']}"
                return 'Unknown Camera'
            
            elif group_by == 'rating':
                try:
                    score = self.quality_ranker.rank_photo(photo_path)
                    if score >= 0.8:
                        return 'High Quality (0.8+)'
                    elif score >= 0.6:
                        return 'Medium Quality (0.6-0.8)'
                    else:
                        return 'Lower Quality (<0.6)'
                except:
                    return 'Unrated'
            
            else:
                return 'Unknown'
                
        except Exception as e:
            logger.debug(f"Error getting group key for {photo_path}: {e}")
            return 'Unknown'
    
    def _generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate recommendations based on collection statistics."""
        recommendations = []
        
        try:
            # Quality-based recommendations
            quality_stats = stats.get('quality_stats', {})
            if quality_stats and not quality_stats.get('error'):
                avg_quality = quality_stats.get('average_quality', 0.5)
                
                if avg_quality < 0.6:
                    recommendations.append("Consider reviewing camera settings - overall image quality is below average")
                
                if avg_quality > 0.8:
                    recommendations.append("Excellent collection quality! Consider creating a portfolio from top-rated images")
                
                # Check quality distribution
                distribution = quality_stats.get('quality_distribution', {})
                poor_count = distribution.get('Poor', 0) + distribution.get('Very Poor', 0)
                total_count = sum(distribution.values())
                
                if poor_count / total_count > 0.3:
                    recommendations.append("30%+ of images are rated poor quality - consider using burst mode less frequently")
            
            # EXIF-based recommendations
            exif_stats = stats.get('exif_stats', {})
            if exif_stats and not exif_stats.get('error'):
                coverage = exif_stats.get('statistics', {}).get('exif_coverage', 0)
                
                if coverage < 0.5:
                    recommendations.append("Many images lack EXIF data - ensure camera metadata recording is enabled")
                
                # ISO recommendations
                iso_stats = exif_stats.get('iso_stats', {})
                if iso_stats and iso_stats.get('avg', 0) > 1600:
                    recommendations.append("High average ISO detected - consider using tripod or better lighting when possible")
                
                # Camera diversity
                cameras = exif_stats.get('top_cameras', {})
                if len(cameras) == 1:
                    recommendations.append("Single camera detected - consider experimenting with different perspectives or lenses")
            
            # Collection size recommendations
            collection_info = stats.get('collection_info', {})
            total_photos = collection_info.get('total_photos', 0)
            
            if total_photos > 1000:
                recommendations.append("Large collection detected - consider using PhotoSight's selection tools to curate your best work")
            elif total_photos < 50:
                recommendations.append("Small collection - perfect size for detailed manual review and curation")
            
            # File organization recommendations
            file_stats = stats.get('file_stats', {})
            directories = file_stats.get('top_directories', {})
            if len(directories) > 10:
                recommendations.append("Photos scattered across many directories - consider organizing by date or project")
            
            if not recommendations:
                recommendations.append("Collection looks well-organized with good quality distribution")
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to analysis errors")
        
        return recommendations
    
    def export_stats(self, stats: Dict, output_path: Path, format: str = 'json') -> bool:
        """
        Export statistics to file.
        
        Args:
            stats: Statistics dictionary
            output_path: Output file path
            format: Export format ('json', 'csv', 'txt')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
            
            elif format == 'csv':
                self._export_csv(stats, output_path)
            
            elif format == 'txt':
                self._export_text(stats, output_path)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Statistics exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting statistics: {e}")
            return False
    
    def _export_csv(self, stats: Dict, output_path: Path):
        """Export statistics as CSV format."""
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Collection info
            writer.writerow(['Collection Statistics'])
            writer.writerow(['Metric', 'Value'])
            
            collection_info = stats.get('collection_info', {})
            for key, value in collection_info.items():
                writer.writerow([key, value])
            
            writer.writerow([])  # Empty row
            
            # Quality statistics
            quality_stats = stats.get('quality_stats', {})
            if quality_stats and not quality_stats.get('error'):
                writer.writerow(['Quality Statistics'])
                writer.writerow(['Metric', 'Value'])
                
                for key, value in quality_stats.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            writer.writerow([f"{key}_{sub_key}", sub_value])
                    else:
                        writer.writerow([key, value])
    
    def _export_text(self, stats: Dict, output_path: Path):
        """Export statistics as human-readable text."""
        with open(output_path, 'w') as f:
            f.write("PhotoSight Collection Statistics Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Collection info
            collection_info = stats.get('collection_info', {})
            f.write("Collection Overview:\n")
            f.write("-" * 20 + "\n")
            for key, value in collection_info.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # Quality stats
            quality_stats = stats.get('quality_stats', {})
            if quality_stats and not quality_stats.get('error'):
                f.write("Quality Analysis:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average Quality Score: {quality_stats.get('average_quality', 'N/A')}\n")
                f.write(f"Quality Range: {quality_stats.get('min_quality', 'N/A')} - {quality_stats.get('max_quality', 'N/A')}\n")
                
                distribution = quality_stats.get('quality_distribution', {})
                if distribution:
                    f.write("\nQuality Distribution:\n")
                    for category, count in distribution.items():
                        percentage = quality_stats.get('quality_percentages', {}).get(category, 0)
                        f.write(f"  {category}: {count} photos ({percentage}%)\n")
                f.write("\n")
            
            # Recommendations
            recommendations = stats.get('recommendations', [])
            if recommendations:
                f.write("Recommendations:\n")
                f.write("-" * 20 + "\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
            
            f.write(f"\nReport generated: {stats.get('generated_at', 'Unknown')}\n")