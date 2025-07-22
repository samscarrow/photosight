"""
Gear Advisor - Intelligent photography equipment recommendations.

Analyzes shooting patterns and quality metrics to recommend optimal gear
choices and identify gaps in current equipment coverage.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from sqlalchemy import func, and_, or_

from ...db import get_session
from ...db.models import Photo, AnalysisResult, Project
from ...db.operations import PhotoOperations

logger = logging.getLogger(__name__)


@dataclass
class GearRecommendation:
    """Represents a gear recommendation."""
    type: str  # 'lens', 'camera', 'accessory'
    category: str  # 'focal_length_gap', 'quality_upgrade', 'usage_pattern'
    recommendation: str
    reason: str
    confidence: float  # 0-1 confidence score
    priority: str  # 'high', 'medium', 'low'
    supporting_data: Dict[str, Any]


class GearAdvisor:
    """
    Analyzes photo metadata to provide intelligent gear recommendations.
    
    Features:
    - Focal length gap analysis
    - Quality correlation with equipment
    - Usage pattern identification
    - Upgrade recommendations based on shooting style
    """
    
    # Focal length ranges for analysis
    FOCAL_RANGES = [
        (14, 24, "ultra-wide"),
        (24, 35, "wide"),
        (35, 50, "normal"),
        (50, 85, "short telephoto"),
        (85, 135, "portrait"),
        (135, 200, "telephoto"),
        (200, 400, "long telephoto"),
        (400, 600, "super telephoto")
    ]
    
    # Quality thresholds
    QUALITY_THRESHOLDS = {
        'excellent': 0.8,
        'good': 0.6,
        'acceptable': 0.4,
        'poor': 0.2
    }
    
    def __init__(self):
        """Initialize the gear advisor."""
        self.recommendations = []
        
    def analyze(self, user_preferences: Optional[Dict[str, Any]] = None, 
                project_name: Optional[str] = None) -> List[GearRecommendation]:
        """
        Perform comprehensive gear analysis and generate recommendations.
        
        Args:
            user_preferences: Optional user preferences for recommendations
            project_name: Optional project name to filter analysis
            
        Returns:
            List of gear recommendations
        """
        self.recommendations = []
        self.project_name = project_name
        
        # Analyze focal length coverage
        self._analyze_focal_length_gaps()
        
        # Analyze lens vs camera quality correlation
        self._analyze_quality_by_gear()
        
        # Analyze usage patterns
        self._analyze_usage_patterns()
        
        # Analyze shooting conditions
        self._analyze_shooting_conditions()
        
        # Generate upgrade recommendations
        self._generate_upgrade_recommendations()
        
        # Sort by priority
        self.recommendations.sort(
            key=lambda r: (
                {'high': 0, 'medium': 1, 'low': 2}[r.priority],
                -r.confidence
            )
        )
        
        return self.recommendations
    
    def _analyze_focal_length_gaps(self):
        """Identify gaps in focal length coverage."""
        with get_session() as session:
            # Get focal length distribution
            query = session.query(
                Photo.focal_length,
                func.count(Photo.id).label('count')
            ).filter(
                Photo.focal_length.isnot(None)
            )
            
            # Apply project filter if specified
            if self.project_name:
                query = query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == self.project_name)
            
            focal_stats = query.group_by(Photo.focal_length).all()
            
            # Create focal length histogram
            focal_histogram = defaultdict(int)
            for fl, count in focal_stats:
                focal_histogram[int(fl)] = count
            
            # Check each focal range
            for start, end, range_name in self.FOCAL_RANGES:
                range_count = sum(
                    focal_histogram[fl] 
                    for fl in range(start, end + 1)
                )
                
                if range_count == 0:
                    # Complete gap
                    self._add_recommendation(
                        type='lens',
                        category='focal_length_gap',
                        recommendation=f"Consider a {range_name} lens ({start}-{end}mm)",
                        reason=f"No photos found in the {range_name} range",
                        confidence=1.0,
                        priority='high' if start <= 135 else 'medium',
                        supporting_data={
                            'focal_range': (start, end),
                            'range_name': range_name,
                            'photos_in_range': 0
                        }
                    )
                elif range_count < 10:
                    # Underutilized range
                    self._add_recommendation(
                        type='lens',
                        category='focal_length_gap',
                        recommendation=f"You rarely use {range_name} focal lengths",
                        reason=f"Only {range_count} photos in {start}-{end}mm range",
                        confidence=0.7,
                        priority='low',
                        supporting_data={
                            'focal_range': (start, end),
                            'range_name': range_name,
                            'photos_in_range': range_count
                        }
                    )
    
    def _analyze_quality_by_gear(self):
        """Analyze photo quality correlation with different gear."""
        with get_session() as session:
            # Get quality scores by lens
            query = session.query(
                Photo.lens_model,
                func.avg(AnalysisResult.sharpness_score).label('avg_sharpness'),
                func.avg(AnalysisResult.overall_ai_score).label('avg_ai_score'),
                func.count(Photo.id).label('photo_count')
            ).join(
                AnalysisResult, Photo.id == AnalysisResult.photo_id
            ).filter(
                Photo.lens_model.isnot(None),
                AnalysisResult.analysis_type == 'technical'
            )
            
            # Apply project filter if specified
            if self.project_name:
                query = query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == self.project_name)
            
            lens_quality = query.group_by(Photo.lens_model).all()
            
            # Identify underperforming lenses
            for lens, sharpness, ai_score, count in lens_quality:
                if count < 10:  # Skip lenses with too few samples
                    continue
                    
                avg_quality = (sharpness or 0) * 0.6 + (ai_score or 0) * 0.4
                
                if avg_quality < self.QUALITY_THRESHOLDS['acceptable']:
                    self._add_recommendation(
                        type='lens',
                        category='quality_upgrade',
                        recommendation=f"Consider upgrading {lens}",
                        reason=f"Average quality score is {avg_quality:.2f} (below acceptable)",
                        confidence=0.8,
                        priority='high',
                        supporting_data={
                            'current_lens': lens,
                            'avg_quality': avg_quality,
                            'photo_count': count
                        }
                    )
                    
                    # Suggest specific alternatives based on focal length
                    self._suggest_lens_alternatives(lens, avg_quality)
    
    def _analyze_usage_patterns(self):
        """Analyze how gear is actually used."""
        with get_session() as session:
            # Get usage statistics
            stats = PhotoOperations.get_gear_statistics(project_name=self.project_name)
            
            # Check for zoom vs prime usage
            zoom_count = 0
            prime_count = 0
            
            for lens in stats['lenses']:
                if '-' in lens['model']:  # Simple check for zoom lens
                    zoom_count += lens['count']
                else:
                    prime_count += lens['count']
            
            total = zoom_count + prime_count
            if total > 50:  # Enough data
                zoom_ratio = zoom_count / total
                
                if zoom_ratio > 0.8:
                    self._add_recommendation(
                        type='lens',
                        category='usage_pattern',
                        recommendation="Consider adding prime lenses for better image quality",
                        reason=f"You use zoom lenses {zoom_ratio*100:.0f}% of the time",
                        confidence=0.7,
                        priority='medium',
                        supporting_data={
                            'zoom_usage': zoom_ratio,
                            'suggested_primes': self._suggest_prime_lenses(stats)
                        }
                    )
                elif zoom_ratio < 0.2:
                    self._add_recommendation(
                        type='lens',
                        category='usage_pattern',
                        recommendation="Consider a versatile zoom lens for flexibility",
                        reason=f"You rarely use zoom lenses ({zoom_ratio*100:.0f}%)",
                        confidence=0.7,
                        priority='medium',
                        supporting_data={
                            'zoom_usage': zoom_ratio,
                            'suggested_zooms': ['24-70mm f/2.8', '70-200mm f/2.8']
                        }
                    )
    
    def _analyze_shooting_conditions(self):
        """Analyze shooting conditions to recommend appropriate gear."""
        with get_session() as session:
            # Check high ISO usage
            high_iso_query = session.query(
                func.count(Photo.id)
            ).filter(Photo.iso >= 3200)
            
            total_query = session.query(func.count(Photo.id))
            
            # Apply project filter if specified
            if self.project_name:
                high_iso_query = high_iso_query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == self.project_name)
                
                total_query = total_query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == self.project_name)
            
            high_iso_photos = high_iso_query.scalar()
            total_photos = total_query.scalar()
            
            if total_photos > 100:
                high_iso_ratio = high_iso_photos / total_photos
                
                if high_iso_ratio > 0.3:
                    self._add_recommendation(
                        type='camera',
                        category='shooting_conditions',
                        recommendation="Consider a camera with better high-ISO performance",
                        reason=f"{high_iso_ratio*100:.0f}% of photos shot at ISO 3200+",
                        confidence=0.8,
                        priority='high',
                        supporting_data={
                            'high_iso_ratio': high_iso_ratio,
                            'high_iso_count': high_iso_photos
                        }
                    )
                    
                    # Also suggest fast lenses
                    self._add_recommendation(
                        type='lens',
                        category='shooting_conditions',
                        recommendation="Consider faster lenses (f/1.4-f/2.8) for low light",
                        reason="Reduce need for high ISO with wider apertures",
                        confidence=0.7,
                        priority='medium',
                        supporting_data={
                            'current_high_iso_usage': high_iso_ratio
                        }
                    )
            
            # Check for image stabilization needs
            slow_shutter_query = session.query(
                func.count(Photo.id)
            ).filter(
                Photo.shutter_speed_numeric > 0.01  # Slower than 1/100s
            )
            
            # Apply project filter if specified
            if self.project_name:
                slow_shutter_query = slow_shutter_query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == self.project_name)
            
            slow_shutter_photos = slow_shutter_query.scalar()
            
            if slow_shutter_photos > total_photos * 0.2:
                self._add_recommendation(
                    type='lens',
                    category='shooting_conditions',
                    recommendation="Consider lenses with image stabilization",
                    reason=f"Frequent use of slow shutter speeds",
                    confidence=0.6,
                    priority='medium',
                    supporting_data={
                        'slow_shutter_count': slow_shutter_photos
                    }
                )
    
    def _generate_upgrade_recommendations(self):
        """Generate specific upgrade recommendations based on usage."""
        with get_session() as session:
            # Get most-used gear
            camera_query = session.query(
                Photo.camera_model,
                func.count(Photo.id).label('count')
            )
            
            lens_query = session.query(
                Photo.lens_model,
                func.count(Photo.id).label('count')
            )
            
            # Apply project filter if specified
            if self.project_name:
                camera_query = camera_query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == self.project_name)
                
                lens_query = lens_query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == self.project_name)
            
            most_used_camera = camera_query.group_by(Photo.camera_model).order_by(
                func.count(Photo.id).desc()
            ).first()
            
            most_used_lens = lens_query.group_by(Photo.lens_model).order_by(
                func.count(Photo.id).desc()
            ).first()
            
            if most_used_lens and most_used_lens[1] > 100:
                # Heavy use of one lens suggests specialization
                lens_model = most_used_lens[0]
                
                # Check if it's a kit lens
                if any(indicator in lens_model.lower() for indicator in ['kit', '18-55', '24-105']):
                    self._add_recommendation(
                        type='lens',
                        category='quality_upgrade',
                        recommendation="Upgrade your most-used lens to professional version",
                        reason=f"Heavy use of {lens_model} suggests it's limiting you",
                        confidence=0.8,
                        priority='high',
                        supporting_data={
                            'current_lens': lens_model,
                            'usage_count': most_used_lens[1]
                        }
                    )
    
    def _suggest_lens_alternatives(self, current_lens: str, quality_score: float):
        """Suggest specific lens alternatives based on current lens."""
        # Extract focal length from lens model
        import re
        focal_match = re.search(r'(\d+)(?:-(\d+))?\s*mm', current_lens)
        
        if focal_match:
            if focal_match.group(2):  # Zoom lens
                start_fl = int(focal_match.group(1))
                end_fl = int(focal_match.group(2))
                
                # Suggest prime in the middle of the range
                mid_fl = (start_fl + end_fl) // 2
                suggested_fl = self._nearest_common_prime(mid_fl)
                
                self._add_recommendation(
                    type='lens',
                    category='quality_upgrade',
                    recommendation=f"Try a {suggested_fl}mm f/1.4 or f/1.8 prime lens",
                    reason="Prime lenses typically offer better sharpness and low-light performance",
                    confidence=0.7,
                    priority='medium',
                    supporting_data={
                        'suggested_focal_length': suggested_fl,
                        'replaces': current_lens
                    }
                )
    
    def _nearest_common_prime(self, focal_length: int) -> int:
        """Find nearest common prime focal length."""
        common_primes = [24, 28, 35, 50, 85, 105, 135, 200, 300, 400]
        return min(common_primes, key=lambda x: abs(x - focal_length))
    
    def _suggest_prime_lenses(self, stats: Dict) -> List[str]:
        """Suggest prime lenses based on current usage."""
        # Find most-used focal lengths
        focal_usage = defaultdict(int)
        for fl_stat in stats['focal_lengths']:
            fl = fl_stat['focal_length']
            if fl:
                focal_usage[self._nearest_common_prime(fl)] += fl_stat['count']
        
        # Get top 3 focal lengths
        top_focals = sorted(focal_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        
        suggestions = []
        for fl, _ in top_focals:
            if fl <= 50:
                suggestions.append(f"{fl}mm f/1.8 or f/1.4")
            elif fl <= 135:
                suggestions.append(f"{fl}mm f/1.8 or f/2")
            else:
                suggestions.append(f"{fl}mm f/2.8 or f/4")
                
        return suggestions
    
    def _add_recommendation(self, **kwargs):
        """Add a recommendation to the list."""
        recommendation = GearRecommendation(**kwargs)
        self.recommendations.append(recommendation)
        logger.debug(f"Added recommendation: {recommendation.recommendation}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all recommendations."""
        summary = {
            'total_recommendations': len(self.recommendations),
            'by_priority': {
                'high': len([r for r in self.recommendations if r.priority == 'high']),
                'medium': len([r for r in self.recommendations if r.priority == 'medium']),
                'low': len([r for r in self.recommendations if r.priority == 'low'])
            },
            'by_type': {
                'lens': len([r for r in self.recommendations if r.type == 'lens']),
                'camera': len([r for r in self.recommendations if r.type == 'camera']),
                'accessory': len([r for r in self.recommendations if r.type == 'accessory'])
            },
            'top_recommendations': self.recommendations[:5]
        }
        
        return summary