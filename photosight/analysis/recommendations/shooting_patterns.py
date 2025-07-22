"""
Shooting Pattern Analyzer - Identifies photographer behavior patterns.

Analyzes when, where, and how photos are taken to provide insights
about shooting habits and preferences.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy import func, extract, and_

from ...db import get_session
from ...db.models import Photo, AnalysisResult, Project
from ...db.operations import PhotoOperations

logger = logging.getLogger(__name__)


@dataclass
class ShootingPattern:
    """Represents a detected shooting pattern."""
    type: str  # 'time', 'location', 'subject', 'technical'
    pattern: str  # Description of the pattern
    confidence: float  # 0-1 confidence score
    frequency: str  # 'always', 'often', 'sometimes', 'rarely'
    data: Dict[str, Any]  # Supporting data
    insight: str  # Actionable insight


class ShootingPatternAnalyzer:
    """
    Analyzes shooting patterns to identify photographer habits and preferences.
    
    Features:
    - Time-based pattern detection (golden hour, weekends, seasons)
    - Location clustering and favorite spots
    - Subject preference analysis
    - Technical setting patterns
    - Weather correlation (if GPS data available)
    """
    
    # Time of day categories
    TIME_CATEGORIES = {
        'early_morning': (5, 7),     # 5am-7am
        'golden_hour_am': (7, 9),    # 7am-9am
        'morning': (9, 12),          # 9am-12pm
        'midday': (12, 14),          # 12pm-2pm
        'afternoon': (14, 17),       # 2pm-5pm
        'golden_hour_pm': (17, 19),  # 5pm-7pm
        'evening': (19, 21),         # 7pm-9pm
        'night': (21, 5)             # 9pm-5am (wraps around)
    }
    
    # Frequency thresholds
    FREQUENCY_THRESHOLDS = {
        'always': 0.8,    # >80% of photos
        'often': 0.5,     # 50-80%
        'sometimes': 0.2, # 20-50%
        'rarely': 0      # <20%
    }
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        self.patterns = []
        
    def analyze(self, min_photos: int = 50, project_name: Optional[str] = None) -> List[ShootingPattern]:
        """
        Perform comprehensive shooting pattern analysis.
        
        Args:
            min_photos: Minimum photos required for pattern detection
            project_name: Optional project name to filter analysis
            
        Returns:
            List of detected shooting patterns
        """
        self.patterns = []
        self.project_name = project_name
        
        with get_session() as session:
            query = session.query(func.count(Photo.id))
            
            # Apply project filter if specified
            if project_name:
                query = query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == project_name)
            
            total_photos = query.scalar()
            
            if total_photos < min_photos:
                logger.info(f"Not enough photos for pattern analysis ({total_photos} < {min_photos})")
                return []
        
        # Analyze different pattern types
        self._analyze_time_patterns()
        self._analyze_day_patterns()
        self._analyze_seasonal_patterns()
        self._analyze_location_patterns()
        self._analyze_subject_patterns()
        self._analyze_technical_patterns()
        self._analyze_burst_patterns()
        
        # Sort by confidence
        self.patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return self.patterns
    
    def _get_base_query(self, session):
        """Get base query with optional project filter."""
        query = session.query(Photo)
        if self.project_name:
            query = query.join(
                Project, Photo.project_id == Project.id
            ).filter(Project.name == self.project_name)
        return query
    
    def _analyze_time_patterns(self):
        """Analyze when photos are taken during the day."""
        with get_session() as session:
            # Get hour distribution
            query = session.query(
                extract('hour', Photo.date_taken).label('hour'),
                func.count(Photo.id).label('count')
            ).filter(
                Photo.date_taken.isnot(None)
            )
            
            # Apply project filter if specified
            if self.project_name:
                query = query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == self.project_name)
            
            hour_stats = query.group_by('hour').all()
            
            if not hour_stats:
                return
                
            # Categorize by time period
            time_counts = defaultdict(int)
            total = 0
            
            for hour, count in hour_stats:
                total += count
                for category, (start, end) in self.TIME_CATEGORIES.items():
                    if category == 'night':
                        if hour >= start or hour < end:
                            time_counts[category] += count
                    else:
                        if start <= hour < end:
                            time_counts[category] += count
                            break
            
            # Find dominant time periods
            for category, count in time_counts.items():
                if count == 0:
                    continue
                    
                ratio = count / total
                frequency = self._get_frequency(ratio)
                
                if frequency in ['always', 'often']:
                    pattern = ShootingPattern(
                        type='time',
                        pattern=f"{category.replace('_', ' ').title()} photographer",
                        confidence=min(ratio * 1.5, 1.0),  # Boost confidence for strong patterns
                        frequency=frequency,
                        data={
                            'time_category': category,
                            'photo_count': count,
                            'percentage': ratio * 100
                        },
                        insight=self._get_time_insight(category, ratio)
                    )
                    self.patterns.append(pattern)
    
    def _analyze_day_patterns(self):
        """Analyze which days of the week photos are taken."""
        with get_session() as session:
            # Get day of week distribution
            day_stats = session.query(
                extract('dow', Photo.date_taken).label('day'),
                func.count(Photo.id).label('count')
            ).filter(
                Photo.date_taken.isnot(None)
            ).group_by('day').all()
            
            if not day_stats:
                return
            
            # Categorize weekday vs weekend
            weekday_count = sum(count for day, count in day_stats if 1 <= day <= 5)
            weekend_count = sum(count for day, count in day_stats if day in [0, 6])
            total = weekday_count + weekend_count
            
            if total == 0:
                return
                
            weekend_ratio = weekend_count / total
            
            if weekend_ratio > 0.7:
                pattern = ShootingPattern(
                    type='time',
                    pattern="Weekend photographer",
                    confidence=weekend_ratio,
                    frequency=self._get_frequency(weekend_ratio),
                    data={
                        'weekend_photos': weekend_count,
                        'weekday_photos': weekday_count,
                        'weekend_percentage': weekend_ratio * 100
                    },
                    insight="You shoot primarily on weekends. Consider planning photo walks or trips around weekends."
                )
                self.patterns.append(pattern)
            elif weekend_ratio < 0.3:
                pattern = ShootingPattern(
                    type='time',
                    pattern="Weekday photographer",
                    confidence=1 - weekend_ratio,
                    frequency=self._get_frequency(1 - weekend_ratio),
                    data={
                        'weekend_photos': weekend_count,
                        'weekday_photos': weekday_count,
                        'weekday_percentage': (1 - weekend_ratio) * 100
                    },
                    insight="You shoot mainly during weekdays. Your schedule allows for less crowded locations."
                )
                self.patterns.append(pattern)
    
    def _analyze_seasonal_patterns(self):
        """Analyze seasonal shooting preferences."""
        with get_session() as session:
            # Get month distribution
            month_stats = session.query(
                extract('month', Photo.date_taken).label('month'),
                func.count(Photo.id).label('count')
            ).filter(
                Photo.date_taken.isnot(None)
            ).group_by('month').all()
            
            if not month_stats:
                return
            
            # Group by seasons
            seasons = {
                'winter': [12, 1, 2],
                'spring': [3, 4, 5],
                'summer': [6, 7, 8],
                'fall': [9, 10, 11]
            }
            
            season_counts = defaultdict(int)
            total = sum(count for _, count in month_stats)
            
            for month, count in month_stats:
                for season, months in seasons.items():
                    if month in months:
                        season_counts[season] += count
                        break
            
            # Find dominant seasons
            for season, count in season_counts.items():
                ratio = count / total
                if ratio > 0.4:  # More than 40% in one season
                    pattern = ShootingPattern(
                        type='time',
                        pattern=f"{season.title()} photographer",
                        confidence=ratio,
                        frequency=self._get_frequency(ratio),
                        data={
                            'season': season,
                            'photo_count': count,
                            'percentage': ratio * 100
                        },
                        insight=f"You're most active in {season}. Plan projects around this seasonal preference."
                    )
                    self.patterns.append(pattern)
    
    def _analyze_location_patterns(self):
        """Analyze location-based shooting patterns."""
        with get_session() as session:
            # Count photos with GPS
            gps_photos = session.query(func.count(Photo.id)).filter(
                Photo.gps_latitude.isnot(None)
            ).scalar()
            
            total_photos = session.query(func.count(Photo.id)).scalar()
            
            if total_photos == 0:
                return
                
            gps_ratio = gps_photos / total_photos
            
            if gps_ratio > 0.7:
                pattern = ShootingPattern(
                    type='location',
                    pattern="Travel photographer",
                    confidence=gps_ratio,
                    frequency=self._get_frequency(gps_ratio),
                    data={
                        'gps_photos': gps_photos,
                        'percentage': gps_ratio * 100
                    },
                    insight="You frequently shoot in different locations. Consider location scouting apps."
                )
                self.patterns.append(pattern)
            elif gps_ratio < 0.1:
                pattern = ShootingPattern(
                    type='location',
                    pattern="Studio/Home photographer",
                    confidence=1 - gps_ratio,
                    frequency=self._get_frequency(1 - gps_ratio),
                    data={
                        'non_gps_photos': total_photos - gps_photos,
                        'percentage': (1 - gps_ratio) * 100
                    },
                    insight="You rarely geotag photos, suggesting controlled environment shooting."
                )
                self.patterns.append(pattern)
    
    def _analyze_subject_patterns(self):
        """Analyze subject preferences based on AI analysis."""
        with get_session() as session:
            # Check for portrait photography
            portrait_photos = session.query(func.count(Photo.id)).join(
                AnalysisResult, Photo.id == AnalysisResult.photo_id
            ).filter(
                AnalysisResult.person_detected == True,
                AnalysisResult.face_quality_score > 0.5
            ).scalar()
            
            total_analyzed = session.query(func.count(Photo.id)).join(
                AnalysisResult, Photo.id == AnalysisResult.photo_id
            ).scalar()
            
            if total_analyzed > 50:
                portrait_ratio = portrait_photos / total_analyzed
                
                if portrait_ratio > 0.5:
                    pattern = ShootingPattern(
                        type='subject',
                        pattern="Portrait photographer",
                        confidence=portrait_ratio,
                        frequency=self._get_frequency(portrait_ratio),
                        data={
                            'portrait_photos': portrait_photos,
                            'percentage': portrait_ratio * 100
                        },
                        insight="You focus on portraits. Consider portrait-specific lenses like 85mm f/1.4."
                    )
                    self.patterns.append(pattern)
    
    def _analyze_technical_patterns(self):
        """Analyze technical shooting preferences."""
        with get_session() as session:
            # Aperture preferences
            avg_aperture = session.query(
                func.avg(Photo.aperture)
            ).filter(Photo.aperture.isnot(None)).scalar()
            
            if avg_aperture:
                if avg_aperture < 2.8:
                    pattern = ShootingPattern(
                        type='technical',
                        pattern="Wide aperture shooter",
                        confidence=0.8,
                        frequency='often',
                        data={
                            'average_aperture': round(avg_aperture, 1)
                        },
                        insight="You prefer shallow depth of field. Fast prime lenses suit your style."
                    )
                    self.patterns.append(pattern)
                elif avg_aperture > 8:
                    pattern = ShootingPattern(
                        type='technical',
                        pattern="Deep focus photographer",
                        confidence=0.8,
                        frequency='often',
                        data={
                            'average_aperture': round(avg_aperture, 1)
                        },
                        insight="You prefer maximum sharpness. Consider focus stacking for even more depth."
                    )
                    self.patterns.append(pattern)
            
            # ISO preferences
            high_iso_photos = session.query(func.count(Photo.id)).filter(
                Photo.iso >= 3200
            ).scalar()
            
            total_with_iso = session.query(func.count(Photo.id)).filter(
                Photo.iso.isnot(None)
            ).scalar()
            
            if total_with_iso > 0:
                high_iso_ratio = high_iso_photos / total_with_iso
                
                if high_iso_ratio > 0.3:
                    pattern = ShootingPattern(
                        type='technical',
                        pattern="Low light photographer",
                        confidence=high_iso_ratio * 2,  # Boost confidence
                        frequency=self._get_frequency(high_iso_ratio),
                        data={
                            'high_iso_photos': high_iso_photos,
                            'percentage': high_iso_ratio * 100
                        },
                        insight="You often shoot in challenging light. Consider fast lenses or better low-light camera."
                    )
                    self.patterns.append(pattern)
    
    def _analyze_burst_patterns(self):
        """Analyze burst shooting patterns."""
        with get_session() as session:
            # Look for photos taken within seconds of each other
            # This would require more complex time-based analysis
            # For now, analyze based on daily photo counts
            
            daily_stats = session.query(
                func.date(Photo.date_taken).label('date'),
                func.count(Photo.id).label('count')
            ).filter(
                Photo.date_taken.isnot(None)
            ).group_by('date').having(
                func.count(Photo.id) > 50  # High volume days
            ).all()
            
            if len(daily_stats) > 5:  # At least 5 high-volume days
                avg_burst_day_count = sum(count for _, count in daily_stats) / len(daily_stats)
                
                pattern = ShootingPattern(
                    type='technical',
                    pattern="Event/Burst shooter",
                    confidence=0.7,
                    frequency='sometimes',
                    data={
                        'high_volume_days': len(daily_stats),
                        'avg_photos_per_event': int(avg_burst_day_count)
                    },
                    insight="You shoot high volumes at events. Consider faster memory cards and dual card slots."
                )
                self.patterns.append(pattern)
    
    def _get_frequency(self, ratio: float) -> str:
        """Convert ratio to frequency description."""
        for freq, threshold in self.FREQUENCY_THRESHOLDS.items():
            if ratio >= threshold:
                return freq
        return 'rarely'
    
    def _get_time_insight(self, category: str, ratio: float) -> str:
        """Generate insight for time-based patterns."""
        insights = {
            'golden_hour_am': "You love morning golden hour. Best light for landscapes and portraits.",
            'golden_hour_pm': "You prefer evening golden hour. Great for warm, dramatic lighting.",
            'night': "You're a night photographer. Consider fast lenses and good high-ISO performance.",
            'midday': "You shoot in harsh midday light. Use fill flash or seek open shade.",
            'early_morning': "You're an early riser. Great for empty locations and soft light."
        }
        
        return insights.get(category, f"You frequently shoot during {category.replace('_', ' ')}.")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all detected patterns."""
        summary = {
            'total_patterns': len(self.patterns),
            'by_type': defaultdict(list),
            'top_patterns': self.patterns[:5],
            'insights': []
        }
        
        # Group by type
        for pattern in self.patterns:
            summary['by_type'][pattern.type].append(pattern)
        
        # Generate overall insights
        if self.patterns:
            time_patterns = [p for p in self.patterns if p.type == 'time']
            if time_patterns:
                summary['insights'].append(
                    f"Your shooting schedule shows {len(time_patterns)} distinct time preferences"
                )
            
            technical_patterns = [p for p in self.patterns if p.type == 'technical']
            if technical_patterns:
                summary['insights'].append(
                    f"Your technical style is consistent with {technical_patterns[0].pattern}"
                )
        
        return summary