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

from ..db.connection import get_session
from ..db.operations import PhotoOperations
from ..db.models import Photo, Project, Task, ProjectPhoto, ProcessingRecipe
from sqlalchemy import and_, or_, func, distinct
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
            
            # Execute search using database
            with get_session() as session:
                # Build base query
                db_query = session.query(Photo)
                
                # Apply filters from parsed parameters
                if 'camera_model' in search_params:
                    db_query = db_query.filter(Photo.camera_model.ilike(f"%{search_params['camera_model']}%"))
                
                if 'lens_model' in search_params:
                    db_query = db_query.filter(Photo.lens_model.ilike(f"%{search_params['lens_model']}%"))
                
                if 'focal_length' in search_params:
                    fl = search_params['focal_length']
                    if isinstance(fl, dict):
                        if 'min' in fl:
                            db_query = db_query.filter(Photo.focal_length >= fl['min'])
                        if 'max' in fl:
                            db_query = db_query.filter(Photo.focal_length <= fl['max'])
                    else:
                        db_query = db_query.filter(Photo.focal_length == fl)
                
                if 'aperture' in search_params:
                    db_query = db_query.filter(Photo.aperture == search_params['aperture'])
                
                if 'iso' in search_params:
                    iso = search_params['iso']
                    if isinstance(iso, dict):
                        if 'min' in iso:
                            db_query = db_query.filter(Photo.iso >= iso['min'])
                        if 'max' in iso:
                            db_query = db_query.filter(Photo.iso <= iso['max'])
                    else:
                        db_query = db_query.filter(Photo.iso == iso)
                
                if 'rating' in search_params:
                    db_query = db_query.filter(Photo.rating >= search_params['rating'])
                
                if 'tags' in search_params:
                    # PostgreSQL array contains
                    for tag in search_params['tags']:
                        db_query = db_query.filter(Photo.tags.contains([tag]))
                
                if 'date_range' in search_params:
                    dr = search_params['date_range']
                    if 'start' in dr:
                        db_query = db_query.filter(Photo.capture_date >= dr['start'])
                    if 'end' in dr:
                        db_query = db_query.filter(Photo.capture_date <= dr['end'])
                
                # Filter by project if specified
                if project:
                    db_query = db_query.join(Photo.projects).filter(Project.name == project)
                
                # Apply limit and execute
                photos = db_query.limit(limit).all()
            
            # Format results
            results = []
            for photo in photos:
                results.append({
                    "id": photo.id,
                    "filename": photo.file_name,
                    "date_taken": photo.capture_date.isoformat() if photo.capture_date else None,
                    "camera": f"{photo.camera_make} {photo.camera_model}",
                    "lens": photo.lens_model,
                    "settings": {
                        "iso": photo.iso,
                        "aperture": f"f/{photo.aperture}" if photo.aperture else None,
                        "shutter_speed": photo.shutter_speed,
                        "focal_length": f"{photo.focal_length}mm" if photo.focal_length else None
                    },
                    "location": {
                        "latitude": photo.gps_latitude,
                        "longitude": photo.gps_longitude
                    } if photo.gps_latitude else None,
                    "quality_status": photo.meta_data.get('processing_status', 'pending') if photo.meta_data else 'pending',
                    "projects": [p.name for p in photo.projects]
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
        with get_session() as session:
            # Base query
            query = session.query(Photo)
            
            # Filter by date if specified
            if since_date:
                query = query.filter(Photo.capture_date >= since_date)
            
            # Filter by project if specified
            if project_name:
                query = query.join(Photo.projects).filter(Project.name == project_name)
            
            # Get camera statistics
            camera_stats = query.with_entities(
                Photo.camera_model,
                func.count(Photo.id).label('count')
            ).filter(
                Photo.camera_model.isnot(None)
            ).group_by(Photo.camera_model).order_by(func.count(Photo.id).desc()).all()
            
            # Get lens statistics
            lens_stats = query.with_entities(
                Photo.lens_model,
                func.count(Photo.id).label('count')
            ).filter(
                Photo.lens_model.isnot(None)
            ).group_by(Photo.lens_model).order_by(func.count(Photo.id).desc()).all()
            
            # Get focal length distribution
            fl_stats = query.with_entities(
                Photo.focal_length,
                func.count(Photo.id).label('count')
            ).filter(
                Photo.focal_length.isnot(None)
            ).group_by(Photo.focal_length).order_by(Photo.focal_length).all()
            
            stats = {
                'cameras': [
                    {'model': cam[0], 'count': cam[1]}
                    for cam in camera_stats
                ],
                'lenses': [
                    {'model': lens[0], 'count': lens[1]}
                    for lens in lens_stats
                ],
                'focal_lengths': [
                    {'focal_length': int(fl[0]), 'count': fl[1]}
                    for fl in fl_stats if fl[0]
                ]
            }
        
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
        with get_session() as session:
            # Base query
            query = session.query(Photo)
            
            # Filter by date if specified
            if since_date:
                query = query.filter(Photo.capture_date >= since_date)
            
            # Filter by project if specified
            if project_name:
                query = query.join(Photo.projects).filter(Project.name == project_name)
            
            # Calculate statistics
            total_photos = query.count()
            
            # Average settings
            avg_stats = query.with_entities(
                func.avg(Photo.iso).label('avg_iso'),
                func.avg(Photo.aperture).label('avg_aperture'),
                func.avg(Photo.focal_length).label('avg_focal_length')
            ).first()
            
            # GPS usage
            gps_count = query.filter(
                Photo.gps_latitude.isnot(None),
                Photo.gps_longitude.isnot(None)
            ).count()
            
            # Most common ISO values
            iso_distribution = query.with_entities(
                Photo.iso,
                func.count(Photo.id).label('count')
            ).filter(
                Photo.iso.isnot(None)
            ).group_by(Photo.iso).order_by(func.count(Photo.id).desc()).limit(10).all()
            
            # Aperture distribution
            aperture_distribution = query.with_entities(
                Photo.aperture,
                func.count(Photo.id).label('count')
            ).filter(
                Photo.aperture.isnot(None)
            ).group_by(Photo.aperture).order_by(Photo.aperture).all()
            
            stats = {
                'total_photos': total_photos,
                'average_iso': int(avg_stats.avg_iso) if avg_stats.avg_iso else None,
                'average_aperture': round(float(avg_stats.avg_aperture), 1) if avg_stats.avg_aperture else None,
                'average_focal_length': int(avg_stats.avg_focal_length) if avg_stats.avg_focal_length else None,
                'gps_percentage': round(gps_count / total_photos * 100, 1) if total_photos else 0,
                'iso_distribution': [
                    {'iso': iso[0], 'count': iso[1]}
                    for iso in iso_distribution
                ],
                'aperture_distribution': [
                    {'aperture': float(ap[0]), 'count': ap[1]}
                    for ap in aperture_distribution if ap[0]
                ]
            }
        
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
            
            # Filter by date if specified
            if since_date:
                query = query.filter(Photo.capture_date >= since_date)
            
            # Filter by project if specified
            if project_name:
                query = query.join(Photo.projects).filter(Project.name == project_name)
            
            # Get quality metrics from metadata
            total = query.count()
            
            # Count by rating
            rating_stats = query.with_entities(
                Photo.rating,
                func.count(Photo.id).label('count')
            ).filter(
                Photo.rating.isnot(None)
            ).group_by(Photo.rating).order_by(Photo.rating.desc()).all()
            
            # Count accepted (rating >= 3)
            accepted = query.filter(Photo.rating >= 3).count()
            rejected = query.filter(Photo.rating < 3).count()
            
            # Get photos with processing recipes (processed)
            processed = session.query(func.count(distinct(ProcessingRecipe.photo_id))).scalar() or 0
            
            # Extract rejection reasons from metadata
            rejection_reasons = {}
            photos_with_issues = query.filter(
                or_(
                    Photo.rating < 3,
                    Photo.meta_data['quality_issues'].isnot(None)
                )
            ).all()
            
            for photo in photos_with_issues:
                if photo.meta_data and 'quality_issues' in photo.meta_data:
                    for issue in photo.meta_data.get('quality_issues', []):
                        rejection_reasons[issue] = rejection_reasons.get(issue, 0) + 1
            
            rejection_stats = [
                (reason, count)
                for reason, count in rejection_reasons.items()
            ]
            
            return {
                "type": "quality",
                "statistics": {
                    "total_photos": total,
                    "accepted": accepted,
                    "rejected": rejected,
                    "processed": processed,
                    "acceptance_rate": round(accepted / total * 100, 1) if total else 0,
                    "rating_distribution": [
                        {"rating": r[0], "count": r[1]}
                        for r in rating_stats
                    ],
                    "quality_issues": [
                        {"issue": r[0], "count": r[1]} 
                        for r in sorted(rejection_stats, key=lambda x: x[1], reverse=True)
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
                f"Most common quality issue: {top_reason[0]} "
                f"({top_reason[1]} photos)"
            )
        
        return insights
    
    async def _get_temporal_statistics(self, since_date: Optional[datetime], group_by: Optional[str]) -> Dict[str, Any]:
        """Get temporal statistics."""
        with get_session() as session:
            # Base query
            query = session.query(Photo)
            
            # Filter by date if specified
            if since_date:
                query = query.filter(Photo.capture_date >= since_date)
            
            # Group by time period
            if group_by == 'month':
                temporal_stats = query.with_entities(
                    func.date_trunc('month', Photo.capture_date).label('period'),
                    func.count(Photo.id).label('count')
                ).filter(
                    Photo.capture_date.isnot(None)
                ).group_by('period').order_by('period').all()
                
                return {
                    "type": "temporal",
                    "group_by": "month",
                    "statistics": [
                        {
                            "period": period.strftime('%Y-%m') if period else None,
                            "count": count
                        }
                        for period, count in temporal_stats
                    ]
                }
            elif group_by == 'year':
                temporal_stats = query.with_entities(
                    func.date_part('year', Photo.capture_date).label('year'),
                    func.count(Photo.id).label('count')
                ).filter(
                    Photo.capture_date.isnot(None)
                ).group_by('year').order_by('year').all()
                
                return {
                    "type": "temporal",
                    "group_by": "year",
                    "statistics": [
                        {
                            "year": int(year) if year else None,
                            "count": count
                        }
                        for year, count in temporal_stats
                    ]
                }
            else:
                # Default to daily for recent data
                temporal_stats = query.with_entities(
                    func.date(Photo.capture_date).label('date'),
                    func.count(Photo.id).label('count')
                ).filter(
                    Photo.capture_date.isnot(None)
                ).group_by('date').order_by('date').limit(90).all()
                
                return {
                    "type": "temporal",
                    "group_by": "day",
                    "statistics": [
                        {
                            "date": date.isoformat() if date else None,
                            "count": count
                        }
                        for date, count in temporal_stats
                    ]
                }
    
    async def _get_location_statistics(self, since_date: Optional[datetime]) -> Dict[str, Any]:
        """Get location statistics."""
        with get_session() as session:
            # Base query
            query = session.query(Photo)
            
            # Filter by date if specified
            if since_date:
                query = query.filter(Photo.capture_date >= since_date)
            
            # Get photos with GPS data
            gps_photos = query.filter(
                Photo.gps_latitude.isnot(None),
                Photo.gps_longitude.isnot(None)
            ).all()
            
            # Calculate location clusters (simple grid-based)
            location_clusters = {}
            grid_size = 0.1  # ~10km grid
            
            for photo in gps_photos:
                # Round to grid
                lat_grid = round(photo.gps_latitude / grid_size) * grid_size
                lon_grid = round(photo.gps_longitude / grid_size) * grid_size
                key = f"{lat_grid:.1f},{lon_grid:.1f}"
                
                if key not in location_clusters:
                    location_clusters[key] = {
                        'center_lat': lat_grid,
                        'center_lon': lon_grid,
                        'count': 0,
                        'photos': []
                    }
                
                location_clusters[key]['count'] += 1
                if len(location_clusters[key]['photos']) < 5:
                    location_clusters[key]['photos'].append(photo.file_name)
            
            # Sort by photo count
            top_locations = sorted(
                location_clusters.values(),
                key=lambda x: x['count'],
                reverse=True
            )[:10]
            
            return {
                "type": "location",
                "total_geotagged": len(gps_photos),
                "location_clusters": top_locations,
                "coverage_area": self._calculate_coverage_area(gps_photos) if gps_photos else None
            }
    
    def _calculate_coverage_area(self, photos: List[Photo]) -> Dict[str, float]:
        """Calculate the geographic coverage area."""
        if not photos:
            return None
            
        lats = [p.gps_latitude for p in photos if p.gps_latitude]
        lons = [p.gps_longitude for p in photos if p.gps_longitude]
        
        if not lats or not lons:
            return None
            
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
            'span_lat': max(lats) - min(lats),
            'span_lon': max(lons) - min(lons)
        }


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
        with get_session() as session:
            # Get camera statistics
            camera_stats = session.query(
                Photo.camera_model,
                func.count(Photo.id).label('count')
            ).filter(
                Photo.camera_model.isnot(None)
            ).group_by(Photo.camera_model).order_by(func.count(Photo.id).desc()).all()
            
            # Get lens statistics
            lens_stats = session.query(
                Photo.lens_model,
                func.count(Photo.id).label('count')
            ).filter(
                Photo.lens_model.isnot(None)
            ).group_by(Photo.lens_model).order_by(func.count(Photo.id).desc()).all()
            
            # Get focal length distribution
            fl_stats = session.query(
                Photo.focal_length,
                func.count(Photo.id).label('count')
            ).filter(
                Photo.focal_length.isnot(None)
            ).group_by(Photo.focal_length).order_by(Photo.focal_length).all()
            
            stats = {
                'cameras': [
                    {'model': cam[0], 'count': cam[1]}
                    for cam in camera_stats
                ],
                'lenses': [
                    {'model': lens[0], 'count': lens[1]}
                    for lens in lens_stats
                ],
                'focal_lengths': [
                    {'focal_length': int(fl[0]), 'count': fl[1]}
                    for fl in fl_stats if fl[0]
                ]
            }
        
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
        # Analyze quality patterns
        with get_session() as session:
            # Get low-rated photos
            low_rated = session.query(Photo).filter(
                Photo.rating < 3
            ).limit(100).all()
            
            # Analyze common issues
            issue_counts = {}
            for photo in low_rated:
                if photo.meta_data and 'quality_issues' in photo.meta_data:
                    for issue in photo.meta_data.get('quality_issues', []):
                        issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            rejection_stats = [
                (issue, count)
                for issue, count in issue_counts.items()
            ]
            
        improvements = []
        
        for reason, count in rejection_stats:
            if reason == 'blur' and count > 10:
                improvements.append({
                    "issue": "Frequent blur issues",
                    "suggestions": [
                        "Increase minimum shutter speed",
                        "Use image stabilization",
                        "Check focus calibration"
                    ]
                })
            elif reason == 'underexposure' and count > 10:
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
        with get_session() as session:
            # Calculate statistics
            avg_stats = session.query(
                func.avg(Photo.aperture).label('avg_aperture'),
                func.avg(Photo.iso).label('avg_iso')
            ).first()
            
            # Check for flash usage in metadata
            total_photos = session.query(Photo).count()
            flash_photos = session.query(Photo).filter(
                Photo.meta_data['flash_used'] == True
            ).count()
            
            stats = {
                'average_aperture': float(avg_stats.avg_aperture) if avg_stats.avg_aperture else None,
                'average_iso': int(avg_stats.avg_iso) if avg_stats.avg_iso else None,
                'flash_percentage': round(flash_photos / total_photos * 100, 1) if total_photos else 0
            }
        
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