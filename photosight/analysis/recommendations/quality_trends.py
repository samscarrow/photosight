"""
Quality Trend Analyzer - Tracks photography quality improvements over time.

Analyzes technical quality metrics, AI scores, and rejection rates to
identify trends and provide actionable feedback on skill development.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy import func, and_, extract
import numpy as np

from ...db import get_session
from ...db.models import Photo, AnalysisResult, Project
from ...db.operations import AnalysisOperations

logger = logging.getLogger(__name__)


@dataclass
class QualityTrend:
    """Represents a quality trend over time."""
    metric: str  # 'sharpness', 'exposure', 'composition', etc.
    direction: str  # 'improving', 'declining', 'stable'
    change_rate: float  # Percentage change per month
    current_level: float  # Current average score
    confidence: float  # Statistical confidence in trend
    timeframe: str  # Period analyzed
    data_points: List[Tuple[datetime, float]]  # Time series data


@dataclass
class QualityInsight:
    """Actionable insight about quality trends."""
    type: str  # 'improvement', 'regression', 'consistency'
    metric: str  # Which metric this relates to
    message: str  # Human-readable insight
    recommendation: str  # What to do about it
    priority: str  # 'high', 'medium', 'low'
    supporting_data: Dict[str, Any]


class QualityTrendAnalyzer:
    """
    Analyzes photo quality trends over time to track improvement.
    
    Features:
    - Technical quality tracking (sharpness, exposure, contrast)
    - AI score progression
    - Rejection rate trends
    - Skill development insights
    - Equipment impact on quality
    """
    
    # Quality metric definitions
    QUALITY_METRICS = {
        'sharpness': {
            'field': 'sharpness_score',
            'weight': 0.3,
            'threshold': 0.7
        },
        'exposure': {
            'field': 'exposure_quality',
            'weight': 0.2,
            'threshold': 0.6
        },
        'contrast': {
            'field': 'contrast_score',
            'weight': 0.1,
            'threshold': 0.5
        },
        'composition': {
            'field': 'composition_score',
            'weight': 0.2,
            'threshold': 0.6
        },
        'overall_ai': {
            'field': 'overall_ai_score',
            'weight': 0.2,
            'threshold': 0.7
        }
    }
    
    # Trend detection thresholds
    TREND_THRESHOLDS = {
        'significant_improvement': 0.15,  # 15% improvement
        'improvement': 0.05,             # 5% improvement
        'stable': 0.05,                  # ±5% change
        'decline': -0.05,                # 5% decline
        'significant_decline': -0.15     # 15% decline
    }
    
    def __init__(self):
        """Initialize the trend analyzer."""
        self.trends = []
        self.insights = []
        
    def analyze(self, 
                timeframe_days: int = 180,
                min_photos_per_period: int = 10,
                project_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive quality trend analysis.
        
        Args:
            timeframe_days: Days of history to analyze
            min_photos_per_period: Minimum photos per time period for valid analysis
            project_name: Optional project name to filter analysis
            
        Returns:
            Dictionary containing trends, insights, and recommendations
        """
        self.trends = []
        self.insights = []
        self.project_name = project_name
        
        start_date = datetime.now() - timedelta(days=timeframe_days)
        
        # Analyze each quality metric
        for metric_name, metric_config in self.QUALITY_METRICS.items():
            trend = self._analyze_metric_trend(
                metric_name, 
                metric_config,
                start_date,
                min_photos_per_period
            )
            if trend:
                self.trends.append(trend)
        
        # Analyze rejection trends
        rejection_trend = self._analyze_rejection_trend(start_date)
        if rejection_trend:
            self.trends.append(rejection_trend)
        
        # Generate insights from trends
        self._generate_insights()
        
        # Analyze equipment impact
        equipment_impact = self._analyze_equipment_quality_impact()
        
        # Compile results
        return {
            'trends': self.trends,
            'insights': self.insights,
            'equipment_impact': equipment_impact,
            'summary': self._generate_summary()
        }
    
    def _analyze_metric_trend(self,
                             metric_name: str,
                             metric_config: Dict,
                             start_date: datetime,
                             min_photos: int) -> Optional[QualityTrend]:
        """Analyze trend for a specific quality metric."""
        with get_session() as session:
            # Get monthly averages
            field = getattr(AnalysisResult, metric_config['field'])
            
            query = session.query(
                func.date_trunc('month', Photo.date_taken).label('month'),
                func.avg(field).label('avg_score'),
                func.count(Photo.id).label('count')
            ).join(
                Photo, AnalysisResult.photo_id == Photo.id
            ).filter(
                Photo.date_taken >= start_date,
                field.isnot(None)
            )
            
            # Apply project filter if specified
            if self.project_name:
                query = query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == self.project_name)
            
            monthly_stats = query.group_by('month').order_by('month').all()
            
            if len(monthly_stats) < 2:
                return None
            
            # Filter out months with too few photos
            valid_months = [(month, score, count) for month, score, count in monthly_stats 
                           if count >= min_photos]
            
            if len(valid_months) < 2:
                return None
            
            # Extract time series
            dates = [month for month, _, _ in valid_months]
            scores = [float(score) for _, score, _ in valid_months]
            
            # Calculate trend
            trend_direction, change_rate, confidence = self._calculate_trend(dates, scores)
            
            return QualityTrend(
                metric=metric_name,
                direction=trend_direction,
                change_rate=change_rate,
                current_level=scores[-1],
                confidence=confidence,
                timeframe=f"{len(valid_months)} months",
                data_points=list(zip(dates, scores))
            )
    
    def _analyze_rejection_trend(self, start_date: datetime) -> Optional[QualityTrend]:
        """Analyze rejection rate trends."""
        with get_session() as session:
            # Get monthly rejection rates
            monthly_stats = session.query(
                func.date_trunc('month', Photo.date_taken).label('month'),
                func.count(Photo.id).label('total'),
                func.sum(func.cast(Photo.processing_status == 'rejected', func.Integer)).label('rejected')
            ).filter(
                Photo.date_taken >= start_date
            ).group_by('month').order_by('month').all()
            
            if len(monthly_stats) < 2:
                return None
            
            # Calculate rejection rates
            dates = []
            rates = []
            
            for month, total, rejected in monthly_stats:
                if total > 10:  # Minimum photos for valid rate
                    dates.append(month)
                    rates.append((rejected or 0) / total)
            
            if len(dates) < 2:
                return None
            
            # Calculate trend (note: for rejection, declining is good)
            trend_direction, change_rate, confidence = self._calculate_trend(dates, rates)
            
            # Invert direction for rejection (declining rejection = improving)
            if trend_direction == 'improving':
                trend_direction = 'declining'
            elif trend_direction == 'declining':
                trend_direction = 'improving'
            
            return QualityTrend(
                metric='rejection_rate',
                direction=trend_direction,
                change_rate=-change_rate,  # Invert rate too
                current_level=rates[-1],
                confidence=confidence,
                timeframe=f"{len(dates)} months",
                data_points=list(zip(dates, rates))
            )
    
    def _calculate_trend(self, 
                        dates: List[datetime], 
                        values: List[float]) -> Tuple[str, float, float]:
        """Calculate trend direction, rate, and confidence."""
        if len(dates) < 2:
            return 'stable', 0.0, 0.0
        
        # Convert dates to numeric (days since first date)
        first_date = dates[0]
        x = [(date - first_date).days for date in dates]
        y = values
        
        # Simple linear regression
        x_array = np.array(x)
        y_array = np.array(y)
        
        # Calculate slope
        n = len(x)
        if n < 2:
            return 'stable', 0.0, 0.0
            
        x_mean = np.mean(x_array)
        y_mean = np.mean(y_array)
        
        numerator = np.sum((x_array - x_mean) * (y_array - y_mean))
        denominator = np.sum((x_array - x_mean) ** 2)
        
        if denominator == 0:
            return 'stable', 0.0, 0.0
            
        slope = numerator / denominator
        
        # Calculate R-squared for confidence
        y_pred = slope * (x_array - x_mean) + y_mean
        ss_res = np.sum((y_array - y_pred) ** 2)
        ss_tot = np.sum((y_array - y_mean) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = abs(r_squared)
        
        # Convert slope to monthly percentage change
        days_in_period = x[-1] - x[0]
        months = days_in_period / 30.0
        
        if months > 0 and y_mean > 0:
            total_change = slope * days_in_period
            change_rate = (total_change / y_mean) / months
        else:
            change_rate = 0.0
        
        # Determine direction
        if change_rate > self.TREND_THRESHOLDS['improvement']:
            direction = 'improving'
        elif change_rate < self.TREND_THRESHOLDS['decline']:
            direction = 'declining'
        else:
            direction = 'stable'
        
        return direction, change_rate, confidence
    
    def _analyze_equipment_quality_impact(self) -> Dict[str, Any]:
        """Analyze how different equipment affects quality."""
        with get_session() as session:
            # Get quality by camera
            camera_quality = session.query(
                Photo.camera_model,
                func.avg(AnalysisResult.overall_ai_score).label('avg_quality'),
                func.count(Photo.id).label('count')
            ).join(
                AnalysisResult, Photo.id == AnalysisResult.photo_id
            ).filter(
                Photo.camera_model.isnot(None),
                AnalysisResult.overall_ai_score.isnot(None)
            ).group_by(Photo.camera_model).having(
                func.count(Photo.id) > 20  # Minimum sample size
            ).all()
            
            # Get quality by lens
            lens_quality = session.query(
                Photo.lens_model,
                func.avg(AnalysisResult.sharpness_score).label('avg_sharpness'),
                func.count(Photo.id).label('count')
            ).join(
                AnalysisResult, Photo.id == AnalysisResult.photo_id
            ).filter(
                Photo.lens_model.isnot(None),
                AnalysisResult.sharpness_score.isnot(None)
            ).group_by(Photo.lens_model).having(
                func.count(Photo.id) > 20
            ).all()
            
        impact = {
            'cameras': [
                {
                    'model': cam,
                    'avg_quality': float(quality),
                    'sample_size': count
                }
                for cam, quality, count in camera_quality
            ],
            'lenses': [
                {
                    'model': lens,
                    'avg_sharpness': float(sharp),
                    'sample_size': count
                }
                for lens, sharp, count in lens_quality
            ]
        }
        
        # Sort by quality
        impact['cameras'].sort(key=lambda x: x['avg_quality'], reverse=True)
        impact['lenses'].sort(key=lambda x: x['avg_sharpness'], reverse=True)
        
        return impact
    
    def _generate_insights(self):
        """Generate actionable insights from trends."""
        for trend in self.trends:
            if trend.direction == 'improving' and trend.confidence > 0.7:
                if trend.metric == 'sharpness':
                    self.insights.append(QualityInsight(
                        type='improvement',
                        metric=trend.metric,
                        message=f"Your focus accuracy has improved {abs(trend.change_rate)*100:.1f}% per month",
                        recommendation="Keep practicing your current focusing technique",
                        priority='medium',
                        supporting_data={'trend': trend}
                    ))
                elif trend.metric == 'rejection_rate':
                    self.insights.append(QualityInsight(
                        type='improvement',
                        metric=trend.metric,
                        message=f"Your keeper rate is improving - {abs(trend.change_rate)*100:.1f}% fewer rejections monthly",
                        recommendation="Maintain your current shooting discipline",
                        priority='high',
                        supporting_data={'trend': trend}
                    ))
            
            elif trend.direction == 'declining' and trend.confidence > 0.7:
                if trend.metric == 'sharpness':
                    self.insights.append(QualityInsight(
                        type='regression',
                        metric=trend.metric,
                        message=f"Sharpness declining {abs(trend.change_rate)*100:.1f}% per month",
                        recommendation="Check lens calibration and use faster shutter speeds",
                        priority='high',
                        supporting_data={'trend': trend}
                    ))
                elif trend.metric == 'exposure':
                    self.insights.append(QualityInsight(
                        type='regression',
                        metric=trend.metric,
                        message=f"Exposure accuracy declining {abs(trend.change_rate)*100:.1f}% per month",
                        recommendation="Review metering modes and use histogram more actively",
                        priority='high',
                        supporting_data={'trend': trend}
                    ))
            
            # Consistency insights
            if trend.direction == 'stable' and trend.current_level > 0.8:
                self.insights.append(QualityInsight(
                    type='consistency',
                    metric=trend.metric,
                    message=f"Consistently high {trend.metric} scores (avg: {trend.current_level:.2f})",
                    recommendation=f"Your {trend.metric} is excellent - focus on other areas",
                    priority='low',
                    supporting_data={'trend': trend}
                ))
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of quality trends."""
        summary = {
            'overall_trend': self._calculate_overall_trend(),
            'improving_areas': [],
            'declining_areas': [],
            'stable_areas': [],
            'top_recommendations': []
        }
        
        # Categorize trends
        for trend in self.trends:
            area = {
                'metric': trend.metric,
                'change_rate': f"{abs(trend.change_rate)*100:.1f}%",
                'confidence': trend.confidence
            }
            
            if trend.direction == 'improving':
                summary['improving_areas'].append(area)
            elif trend.direction == 'declining':
                summary['declining_areas'].append(area)
            else:
                summary['stable_areas'].append(area)
        
        # Get top recommendations
        high_priority_insights = [i for i in self.insights if i.priority == 'high']
        summary['top_recommendations'] = [
            i.recommendation for i in high_priority_insights[:3]
        ]
        
        return summary
    
    def _calculate_overall_trend(self) -> str:
        """Calculate overall quality trend."""
        if not self.trends:
            return "insufficient_data"
        
        # Weight trends by metric importance and confidence
        weighted_sum = 0
        total_weight = 0
        
        for trend in self.trends:
            if trend.metric in self.QUALITY_METRICS:
                weight = self.QUALITY_METRICS[trend.metric]['weight'] * trend.confidence
                weighted_sum += trend.change_rate * weight
                total_weight += weight
        
        if total_weight == 0:
            return "stable"
        
        overall_rate = weighted_sum / total_weight
        
        if overall_rate > 0.05:
            return "improving"
        elif overall_rate < -0.05:
            return "declining"
        else:
            return "stable"
    
    def get_report(self) -> str:
        """Generate a human-readable quality report."""
        lines = ["Photography Quality Trend Report", "=" * 40, ""]
        
        # Overall summary
        overall = self._calculate_overall_trend()
        lines.append(f"Overall Trend: {overall.upper()}")
        lines.append("")
        
        # Improving areas
        improving = [t for t in self.trends if t.direction == 'improving']
        if improving:
            lines.append("Improving Areas:")
            for trend in improving:
                lines.append(f"  • {trend.metric}: +{trend.change_rate*100:.1f}% per month")
            lines.append("")
        
        # Areas needing attention
        declining = [t for t in self.trends if t.direction == 'declining']
        if declining:
            lines.append("Areas Needing Attention:")
            for trend in declining:
                lines.append(f"  • {trend.metric}: {trend.change_rate*100:.1f}% per month")
            lines.append("")
        
        # Top insights
        if self.insights:
            lines.append("Key Insights:")
            for insight in self.insights[:5]:
                lines.append(f"  • {insight.message}")
            lines.append("")
        
        return "\n".join(lines)