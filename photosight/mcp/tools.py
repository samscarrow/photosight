"""
MCP Tools for PhotoSight natural language queries and analytics.

Implements tools that AI assistants can use to query the photo database,
generate statistics, and provide insights.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import text, func

from ..db import get_session
from ..db.operations import PhotoOperations, AnalysisOperations
from ..db.models import Photo, AnalysisResult
from .security import SecurityManager, SecurityError
from .query_builder import NaturalLanguageQueryBuilder

logger = logging.getLogger(__name__)


class QueryTool:
    """
    Tool for natural language photo queries.
    
    Allows AI assistants to search photos using natural language
    and returns structured results.
    """
    
    def __init__(self, security: SecurityManager):
        self.security = security
        self.query_builder = NaturalLanguageQueryBuilder()
        
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition."""
        return {
            "name": "query_photos",
            "description": "Search photos using natural language queries",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query (e.g., 'sharp portraits with 85mm lens')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 50, max: 100)"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project name to filter results"
                    }
                },
                "required": ["query"]
            }
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a natural language photo query.
        
        Args:
            arguments: Tool arguments with 'query' and optional 'limit'
            
        Returns:
            Query results with photos and metadata
        """
        query = arguments.get('query', '')
        limit = min(arguments.get('limit', 50), 100)
        project = arguments.get('project')
        
        try:
            # Validate natural language query
            if not self.security.validate_natural_language_query(query):
                return {
                    "error": "Query contains forbidden operations",
                    "results": []
                }
            
            # Convert natural language to search parameters
            search_params = self.query_builder.parse(query)
            
            # Add project filter if specified
            if project:
                search_params['project_name'] = project
            
            # Execute search
            photos = PhotoOperations.search_photos(
                **search_params,
                limit=limit
            )
            
            # Format results
            results = []
            for photo in photos:
                results.append({
                    "id": photo.id,
                    "filename": photo.filename,
                    "date_taken": photo.date_taken.isoformat() if photo.date_taken else None,
                    "camera": f"{photo.camera_make} {photo.camera_model}",
                    "lens": photo.lens_model,
                    "settings": {
                        "iso": photo.iso,
                        "aperture": f"f/{photo.aperture}" if photo.aperture else None,
                        "shutter_speed": photo.shutter_speed_display,
                        "focal_length": f"{photo.focal_length}mm" if photo.focal_length else None
                    },
                    "location": {
                        "latitude": photo.gps_latitude,
                        "longitude": photo.gps_longitude
                    } if photo.gps_latitude else None,
                    "quality_status": photo.processing_status
                })
            
            return {
                "query": query,
                "parsed_parameters": search_params,
                "count": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                "error": str(e),
                "results": []
            }


class StatisticsTool:
    """
    Tool for generating photography statistics and analytics.
    
    Provides insights into gear usage, shooting patterns, and quality metrics.
    """
    
    def __init__(self, security: SecurityManager):
        self.security = security
        
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition."""
        return {
            "name": "get_statistics",
            "description": "Get photography statistics and analytics",
            "input_schema": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["gear", "shooting", "quality", "temporal", "location"],
                        "description": "Type of statistics to retrieve"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period (e.g., '30days', '6months', 'all')"
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Grouping for statistics (e.g., 'camera', 'month')"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project name to filter statistics"
                    }
                },
                "required": ["type"]
            }
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate photography statistics.
        
        Args:
            arguments: Tool arguments with statistics type and options
            
        Returns:
            Statistical analysis results
        """
        stat_type = arguments.get('type')
        period = arguments.get('period', 'all')
        group_by = arguments.get('group_by')
        project = arguments.get('project')
        
        try:
            # Parse time period
            date_filter = self._parse_period(period)
            
            if stat_type == 'gear':
                return await self._get_gear_statistics(date_filter, project)
            elif stat_type == 'shooting':
                return await self._get_shooting_statistics(date_filter, project)
            elif stat_type == 'quality':
                return await self._get_quality_statistics(date_filter, project)
            elif stat_type == 'temporal':
                return await self._get_temporal_statistics(date_filter, group_by)
            elif stat_type == 'location':
                return await self._get_location_statistics(date_filter)
            else:
                return {"error": f"Unknown statistics type: {stat_type}"}
                
        except Exception as e:
            logger.error(f"Statistics generation failed: {e}")
            return {"error": str(e)}
    
    def _parse_period(self, period: str) -> Optional[datetime]:
        """Parse period string to date filter."""
        if period == 'all':
            return None
            
        # Parse periods like '30days', '6months', '1year'
        import re
        match = re.match(r'(\d+)(days?|months?|years?)', period.lower())
        if match:
            amount = int(match.group(1))
            unit = match.group(2).rstrip('s')
            
            if unit == 'day':
                return datetime.now() - timedelta(days=amount)
            elif unit == 'month':
                return datetime.now() - timedelta(days=amount * 30)
            elif unit == 'year':
                return datetime.now() - timedelta(days=amount * 365)
                
        return None
    
    async def _get_gear_statistics(self, since_date: Optional[datetime], project_name: Optional[str] = None) -> Dict[str, Any]:
        """Get camera gear usage statistics."""
        stats = PhotoOperations.get_gear_statistics(project_name=project_name)
        
        # Add usage percentages
        total_photos = sum(cam['count'] for cam in stats['cameras'])
        if total_photos > 0:
            for cam in stats['cameras']:
                cam['percentage'] = round(cam['count'] / total_photos * 100, 1)
            
        return {
            "type": "gear",
            "statistics": stats,
            "insights": self._generate_gear_insights(stats),
            "project": project_name
        }
    
    async def _get_shooting_statistics(self, since_date: Optional[datetime], project_name: Optional[str] = None) -> Dict[str, Any]:
        """Get shooting pattern statistics."""
        stats = PhotoOperations.get_shooting_statistics(project_name=project_name)
        
        return {
            "type": "shooting",
            "statistics": stats,
            "insights": self._generate_shooting_insights(stats),
            "project": project_name
        }
    
    async def _get_quality_statistics(self, since_date: Optional[datetime], project_name: Optional[str] = None) -> Dict[str, Any]:
        """Get photo quality statistics."""
        with get_session() as session:
            # Base query
            query = session.query(Photo)
            
            # Filter by project if specified
            if project_name:
                from ..db.models import Project
                query = query.join(Project).filter(Project.name == project_name)
            
            # Get acceptance rate
            total = query.count()
            accepted = query.filter(
                Photo.processing_status == 'processed'
            ).count()
            rejected = query.filter(
                Photo.processing_status == 'rejected'
            ).count()
            
            # Get rejection reasons
            rejection_query = query.filter(
                Photo.rejection_reason.isnot(None)
            )
            rejection_stats = rejection_query.with_entities(
                Photo.rejection_reason,
                func.count(Photo.id).label('count')
            ).group_by(Photo.rejection_reason).all()
            
            return {
                "type": "quality",
                "statistics": {
                    "total_photos": total,
                    "accepted": accepted,
                    "rejected": rejected,
                    "acceptance_rate": round(accepted / total * 100, 1) if total else 0,
                    "rejection_reasons": [
                        {"reason": r[0], "count": r[1]} 
                        for r in rejection_stats
                    ]
                },
                "insights": self._generate_quality_insights(rejection_stats),
                "project": project_name
            }
    
    def _generate_gear_insights(self, stats: Dict) -> List[str]:
        """Generate insights from gear statistics."""
        insights = []
        
        # Most used camera
        if stats['cameras']:
            top_camera = stats['cameras'][0]
            insights.append(
                f"Your primary camera is {top_camera['model']} "
                f"({top_camera['percentage']}% of photos)"
            )
        
        # Focal length preferences
        if stats['focal_lengths']:
            common_fl = [fl for fl in stats['focal_lengths'] if fl['count'] > 10]
            if common_fl:
                fl_ranges = self._categorize_focal_lengths(common_fl)
                insights.append(f"You prefer {fl_ranges} focal lengths")
        
        return insights
    
    def _categorize_focal_lengths(self, focal_lengths: List[Dict]) -> str:
        """Categorize focal length usage."""
        wide = sum(fl['count'] for fl in focal_lengths if fl['focal_length'] < 35)
        normal = sum(fl['count'] for fl in focal_lengths if 35 <= fl['focal_length'] <= 85)
        tele = sum(fl['count'] for fl in focal_lengths if fl['focal_length'] > 85)
        
        total = wide + normal + tele
        if total == 0:
            return "varied"
            
        if wide / total > 0.5:
            return "wide-angle"
        elif tele / total > 0.5:
            return "telephoto"
        else:
            return "normal range"
    
    def _generate_shooting_insights(self, stats: Dict) -> List[str]:
        """Generate insights from shooting statistics."""
        insights = []
        
        # ISO usage
        if stats.get('average_iso'):
            avg_iso = int(stats['average_iso'])
            if avg_iso < 400:
                insights.append(f"You prefer low ISO settings (avg: {avg_iso})")
            elif avg_iso > 1600:
                insights.append(f"You often shoot in low light (avg ISO: {avg_iso})")
        
        # GPS usage
        if stats.get('gps_percentage', 0) > 50:
            insights.append("You frequently geotag your photos")
        
        return insights
    
    def _generate_quality_insights(self, rejection_stats: List) -> List[str]:
        """Generate insights from quality statistics."""
        insights = []
        
        if rejection_stats:
            top_reason = max(rejection_stats, key=lambda x: x[1])
            insights.append(
                f"Most common rejection reason: {top_reason[0]} "
                f"({top_reason[1]} photos)"
            )
        
        return insights


class InsightsTool:
    """
    Tool for generating photography insights and recommendations.
    
    Provides personalized recommendations based on shooting patterns.
    """
    
    def __init__(self, security: SecurityManager):
        self.security = security
        
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition."""
        return {
            "name": "get_insights",
            "description": "Get personalized photography insights and recommendations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["gear_recommendations", "quality_improvement", "shooting_patterns"],
                        "description": "Type of insights to generate"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context for insights"
                    }
                },
                "required": ["type"]
            }
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate photography insights.
        
        Args:
            arguments: Tool arguments with insight type
            
        Returns:
            Personalized insights and recommendations
        """
        insight_type = arguments.get('type')
        context = arguments.get('context', {})
        
        try:
            if insight_type == 'gear_recommendations':
                return await self._get_gear_recommendations()
            elif insight_type == 'quality_improvement':
                return await self._get_quality_improvements()
            elif insight_type == 'shooting_patterns':
                return await self._get_shooting_pattern_insights()
            else:
                return {"error": f"Unknown insight type: {insight_type}"}
                
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return {"error": str(e)}
    
    async def _get_gear_recommendations(self) -> Dict[str, Any]:
        """Generate gear recommendations based on usage patterns."""
        # Get current gear statistics
        stats = PhotoOperations.get_gear_statistics()
        
        recommendations = []
        
        # Analyze focal length gaps
        focal_lengths = [fl['focal_length'] for fl in stats['focal_lengths']]
        gaps = self._find_focal_length_gaps(focal_lengths)
        
        if gaps:
            recommendations.append({
                "type": "focal_length_gap",
                "recommendation": f"Consider lenses in the {gaps[0]}mm range",
                "reason": "You have no photos in this focal length range"
            })
        
        # Analyze lens vs zoom usage
        lens_variety = len(set(lens['model'] for lens in stats['lenses']))
        if lens_variety < 3:
            recommendations.append({
                "type": "lens_variety",
                "recommendation": "Consider expanding your lens collection",
                "reason": "Limited variety in current lens usage"
            })
        
        return {
            "type": "gear_recommendations",
            "recommendations": recommendations,
            "current_gear": {
                "cameras": stats['cameras'][:3],
                "lenses": stats['lenses'][:5]
            }
        }
    
    def _find_focal_length_gaps(self, focal_lengths: List[int]) -> List[str]:
        """Find gaps in focal length coverage."""
        if not focal_lengths:
            return []
            
        gaps = []
        ranges = [(14, 24), (24, 35), (35, 85), (85, 135), (135, 200), (200, 400)]
        
        for start, end in ranges:
            if not any(start <= fl <= end for fl in focal_lengths):
                gaps.append(f"{start}-{end}")
                
        return gaps
    
    async def _get_quality_improvements(self) -> Dict[str, Any]:
        """Generate quality improvement suggestions."""
        # Analyze rejection patterns
        with get_session() as session:
            rejection_stats = session.query(
                Photo.rejection_reason,
                func.count(Photo.id).label('count')
            ).filter(
                Photo.rejection_reason.isnot(None)
            ).group_by(Photo.rejection_reason).all()
            
        improvements = []
        
        for reason, count in rejection_stats:
            if reason == 'blurry' and count > 10:
                improvements.append({
                    "issue": "Frequent blur issues",
                    "suggestions": [
                        "Increase minimum shutter speed",
                        "Use image stabilization",
                        "Check focus calibration"
                    ]
                })
            elif reason == 'underexposed' and count > 10:
                improvements.append({
                    "issue": "Underexposure problems",
                    "suggestions": [
                        "Use exposure compensation +0.3 to +0.7",
                        "Enable highlight priority mode",
                        "Consider fill flash for backlit scenes"
                    ]
                })
        
        return {
            "type": "quality_improvement",
            "improvements": improvements,
            "overall_tip": "Focus on your most common issue first for maximum impact"
        }
    
    async def _get_shooting_pattern_insights(self) -> Dict[str, Any]:
        """Analyze and provide insights on shooting patterns."""
        stats = PhotoOperations.get_shooting_statistics()
        
        patterns = []
        
        # Time-based patterns would go here
        # For now, provide basic insights
        if stats.get('average_aperture', 0) < 4:
            patterns.append({
                "pattern": "Shallow depth of field preference",
                "insight": "You prefer wide apertures for subject isolation"
            })
        
        if stats.get('flash_percentage', 0) < 5:
            patterns.append({
                "pattern": "Natural light photographer",
                "insight": "You rarely use flash, preferring available light"
            })
        
        return {
            "type": "shooting_patterns",
            "patterns": patterns,
            "summary": "Your shooting style analysis based on technical data"
        }