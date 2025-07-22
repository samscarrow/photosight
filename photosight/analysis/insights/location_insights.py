"""
Location-based Insights - Analyze photography patterns by location.

Provides insights about shooting locations, travel patterns, and
location-specific quality metrics.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
from math import radians, cos, sin, asin, sqrt
from sqlalchemy import func, and_, or_

from ...db import get_session
from ...db.models import Photo, AnalysisResult, Project
from ...db.operations import PhotoOperations

logger = logging.getLogger(__name__)


@dataclass
class LocationCluster:
    """Represents a cluster of photos taken in proximity."""
    center_lat: float
    center_lon: float
    radius_km: float
    photo_count: int
    date_range: Tuple[datetime, datetime]
    avg_quality: float
    location_name: Optional[str] = None
    photos: List[Photo] = None


@dataclass
class LocationInsight:
    """Location-based photography insight."""
    type: str  # 'hotspot', 'travel', 'quality', 'time_pattern'
    title: str
    description: str
    location: Optional[Tuple[float, float]]  # (lat, lon)
    data: Dict[str, Any]
    recommendation: str


class LocationInsights:
    """
    Analyzes location-based photography patterns and provides insights.
    
    Features:
    - Hotspot detection (frequently photographed locations)
    - Travel pattern analysis
    - Location-specific quality metrics
    - Time-of-day preferences by location
    - Seasonal location preferences
    - Distance and coverage analysis
    """
    
    # Clustering parameters
    CLUSTER_RADIUS_KM = 1.0  # 1km radius for location clustering
    MIN_PHOTOS_FOR_HOTSPOT = 10
    
    def __init__(self):
        """Initialize location insights analyzer."""
        self.clusters = []
        self.insights = []
        
    def analyze(self, include_unnamed: bool = True, project_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive location-based analysis.
        
        Args:
            include_unnamed: Include locations without reverse geocoding
            project_name: Optional project name to filter analysis
            
        Returns:
            Dictionary containing location insights and statistics
        """
        self.clusters = []
        self.insights = []
        self.project_name = project_name
        
        # Check if we have location data
        with get_session() as session:
            query = session.query(func.count(Photo.id)).filter(
                Photo.gps_latitude.isnot(None)
            )
            
            # Apply project filter if specified
            if project_name:
                query = query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == project_name)
                
            gps_photos = query.scalar()
            
            if gps_photos < 10:
                logger.info(f"Insufficient GPS data for location analysis ({gps_photos} photos)")
                return {
                    'error': 'Insufficient location data',
                    'gps_photos': gps_photos
                }
        
        # Find location clusters (hotspots)
        self._find_location_clusters()
        
        # Analyze travel patterns
        self._analyze_travel_patterns()
        
        # Analyze location quality
        self._analyze_location_quality()
        
        # Analyze time patterns by location
        self._analyze_location_time_patterns()
        
        # Generate coverage map
        coverage_stats = self._analyze_coverage()
        
        # Compile results
        return {
            'hotspots': [self._cluster_to_dict(c) for c in self.clusters[:10]],
            'insights': self.insights,
            'coverage': coverage_stats,
            'summary': self._generate_summary()
        }
    
    def _find_location_clusters(self):
        """Find clusters of photos taken in proximity."""
        with get_session() as session:
            # Get all photos with GPS
            query = session.query(Photo).filter(
                Photo.gps_latitude.isnot(None),
                Photo.gps_longitude.isnot(None)
            )
            
            # Apply project filter if specified
            if self.project_name:
                query = query.join(
                    Project, Photo.project_id == Project.id
                ).filter(Project.name == self.project_name)
            
            gps_photos = query.all()
            
            if not gps_photos:
                return
            
            # Simple clustering algorithm
            clusters = []
            processed = set()
            
            for i, photo in enumerate(gps_photos):
                if i in processed:
                    continue
                    
                # Start new cluster
                cluster_photos = [photo]
                processed.add(i)
                
                # Find nearby photos
                for j, other in enumerate(gps_photos[i+1:], i+1):
                    if j in processed:
                        continue
                        
                    distance = self._haversine_distance(
                        photo.gps_latitude, photo.gps_longitude,
                        other.gps_latitude, other.gps_longitude
                    )
                    
                    if distance <= self.CLUSTER_RADIUS_KM:
                        cluster_photos.append(other)
                        processed.add(j)
                
                # Create cluster if it has enough photos
                if len(cluster_photos) >= self.MIN_PHOTOS_FOR_HOTSPOT:
                    cluster = self._create_cluster(cluster_photos)
                    clusters.append(cluster)
            
            # Sort by photo count
            clusters.sort(key=lambda c: c.photo_count, reverse=True)
            self.clusters = clusters[:20]  # Keep top 20 hotspots
            
            # Generate hotspot insights
            for cluster in self.clusters[:5]:
                self.insights.append(LocationInsight(
                    type='hotspot',
                    title=f"Photography Hotspot: {cluster.location_name or 'Unknown Location'}",
                    description=f"{cluster.photo_count} photos taken within {cluster.radius_km:.1f}km",
                    location=(cluster.center_lat, cluster.center_lon),
                    data={
                        'photo_count': cluster.photo_count,
                        'avg_quality': cluster.avg_quality,
                        'date_range': (cluster.date_range[0].isoformat(), 
                                     cluster.date_range[1].isoformat())
                    },
                    recommendation="This is one of your favorite shooting locations"
                ))
    
    def _create_cluster(self, photos: List[Photo]) -> LocationCluster:
        """Create a cluster from a list of photos."""
        # Calculate center
        avg_lat = sum(p.gps_latitude for p in photos) / len(photos)
        avg_lon = sum(p.gps_longitude for p in photos) / len(photos)
        
        # Calculate radius
        max_distance = 0
        for photo in photos:
            distance = self._haversine_distance(
                avg_lat, avg_lon,
                photo.gps_latitude, photo.gps_longitude
            )
            max_distance = max(max_distance, distance)
        
        # Get date range
        dates = [p.date_taken for p in photos if p.date_taken]
        date_range = (min(dates), max(dates)) if dates else (None, None)
        
        # Calculate average quality
        with get_session() as session:
            photo_ids = [p.id for p in photos]
            avg_quality = session.query(
                func.avg(AnalysisResult.overall_ai_score)
            ).filter(
                AnalysisResult.photo_id.in_(photo_ids)
            ).scalar() or 0.0
        
        return LocationCluster(
            center_lat=avg_lat,
            center_lon=avg_lon,
            radius_km=max_distance,
            photo_count=len(photos),
            date_range=date_range,
            avg_quality=float(avg_quality),
            photos=photos
        )
    
    def _analyze_travel_patterns(self):
        """Analyze travel and movement patterns."""
        with get_session() as session:
            # Get photos ordered by date
            gps_photos = session.query(
                Photo.id,
                Photo.gps_latitude,
                Photo.gps_longitude,
                Photo.date_taken
            ).filter(
                Photo.gps_latitude.isnot(None),
                Photo.gps_longitude.isnot(None),
                Photo.date_taken.isnot(None)
            ).order_by(Photo.date_taken).all()
            
            if len(gps_photos) < 2:
                return
            
            # Calculate travel statistics
            total_distance = 0
            max_single_distance = 0
            travel_days = set()
            
            for i in range(1, len(gps_photos)):
                prev = gps_photos[i-1]
                curr = gps_photos[i]
                
                # Calculate distance
                distance = self._haversine_distance(
                    prev.gps_latitude, prev.gps_longitude,
                    curr.gps_latitude, curr.gps_longitude
                )
                
                # Only count if photos are on different days (actual travel)
                if prev.date_taken.date() != curr.date_taken.date():
                    total_distance += distance
                    max_single_distance = max(max_single_distance, distance)
                    travel_days.add(curr.date_taken.date())
            
            # Generate travel insights
            if total_distance > 100:  # More than 100km total
                self.insights.append(LocationInsight(
                    type='travel',
                    title="Active Travel Photographer",
                    description=f"Total travel distance: {total_distance:.0f}km across {len(travel_days)} days",
                    location=None,
                    data={
                        'total_distance_km': total_distance,
                        'max_trip_km': max_single_distance,
                        'travel_days': len(travel_days)
                    },
                    recommendation="Your travel photography shows diverse locations"
                ))
            
            # Check for international travel (very large distances)
            if max_single_distance > 1000:
                self.insights.append(LocationInsight(
                    type='travel',
                    title="International Photographer",
                    description=f"Longest single journey: {max_single_distance:.0f}km",
                    location=None,
                    data={'max_distance_km': max_single_distance},
                    recommendation="Consider organizing photos by trip/country"
                ))
    
    def _analyze_location_quality(self):
        """Analyze photo quality by location."""
        if not self.clusters:
            return
            
        # Find locations with notably high or low quality
        high_quality_locations = []
        low_quality_locations = []
        
        for cluster in self.clusters:
            if cluster.avg_quality > 0.8 and cluster.photo_count >= 20:
                high_quality_locations.append(cluster)
            elif cluster.avg_quality < 0.5 and cluster.photo_count >= 20:
                low_quality_locations.append(cluster)
        
        # Generate quality insights
        if high_quality_locations:
            best = high_quality_locations[0]
            self.insights.append(LocationInsight(
                type='quality',
                title="Best Photography Location",
                description=f"Average quality score: {best.avg_quality:.2f}",
                location=(best.center_lat, best.center_lon),
                data={
                    'avg_quality': best.avg_quality,
                    'photo_count': best.photo_count
                },
                recommendation="This location consistently produces your best work"
            ))
        
        if low_quality_locations:
            worst = low_quality_locations[0]
            self.insights.append(LocationInsight(
                type='quality',
                title="Challenging Photography Location",
                description=f"Average quality score: {worst.avg_quality:.2f}",
                location=(worst.center_lat, worst.center_lon),
                data={
                    'avg_quality': worst.avg_quality,
                    'photo_count': worst.photo_count
                },
                recommendation="Consider different techniques or times for this location"
            ))
    
    def _analyze_location_time_patterns(self):
        """Analyze when photos are taken at different locations."""
        if not self.clusters:
            return
            
        # Analyze time patterns for top locations
        for cluster in self.clusters[:3]:
            if not cluster.photos:
                continue
                
            # Get hour distribution for this location
            hours = defaultdict(int)
            for photo in cluster.photos:
                if photo.date_taken:
                    hours[photo.date_taken.hour] += 1
            
            if not hours:
                continue
                
            # Find peak hours
            peak_hour = max(hours.items(), key=lambda x: x[1])[0]
            
            # Determine time preference
            if 5 <= peak_hour <= 9:
                time_pref = "sunrise/golden hour"
            elif 17 <= peak_hour <= 19:
                time_pref = "sunset/golden hour"
            elif 10 <= peak_hour <= 16:
                time_pref = "daylight"
            else:
                time_pref = "night/blue hour"
            
            self.insights.append(LocationInsight(
                type='time_pattern',
                title=f"Preferred Time at {cluster.location_name or 'Hotspot'}",
                description=f"Most photos taken during {time_pref}",
                location=(cluster.center_lat, cluster.center_lon),
                data={
                    'peak_hour': peak_hour,
                    'time_preference': time_pref,
                    'hour_distribution': dict(hours)
                },
                recommendation=f"Plan visits to this location during {time_pref}"
            ))
    
    def _analyze_coverage(self) -> Dict[str, Any]:
        """Analyze geographic coverage of photography."""
        with get_session() as session:
            # Get bounding box of all photos
            bounds = session.query(
                func.min(Photo.gps_latitude).label('min_lat'),
                func.max(Photo.gps_latitude).label('max_lat'),
                func.min(Photo.gps_longitude).label('min_lon'),
                func.max(Photo.gps_longitude).label('max_lon')
            ).filter(
                Photo.gps_latitude.isnot(None),
                Photo.gps_longitude.isnot(None)
            ).first()
            
            if not bounds or not bounds.min_lat:
                return {}
            
            # Calculate coverage area
            lat_range = bounds.max_lat - bounds.min_lat
            lon_range = bounds.max_lon - bounds.min_lon
            
            # Approximate area (simple rectangle)
            avg_lat = (bounds.min_lat + bounds.max_lat) / 2
            lat_km = lat_range * 111  # 1 degree latitude ≈ 111km
            lon_km = lon_range * 111 * cos(radians(avg_lat))
            area_km2 = lat_km * lon_km
            
            # Count unique "grid cells" visited (1km x 1km grid)
            photos = session.query(
                Photo.gps_latitude,
                Photo.gps_longitude
            ).filter(
                Photo.gps_latitude.isnot(None),
                Photo.gps_longitude.isnot(None)
            ).all()
            
            grid_cells = set()
            for lat, lon in photos:
                # Convert to 1km grid cells
                grid_lat = int(lat * 111)  # ~1km precision
                grid_lon = int(lon * 111 * cos(radians(lat)))
                grid_cells.add((grid_lat, grid_lon))
            
            coverage_stats = {
                'bounding_box': {
                    'min_lat': bounds.min_lat,
                    'max_lat': bounds.max_lat,
                    'min_lon': bounds.min_lon,
                    'max_lon': bounds.max_lon
                },
                'approximate_area_km2': area_km2,
                'unique_locations': len(grid_cells),
                'coverage_density': len(photos) / len(grid_cells) if grid_cells else 0
            }
            
            # Generate coverage insight
            if area_km2 > 10000:  # More than 10,000 km²
                self.insights.append(LocationInsight(
                    type='travel',
                    title="Wide Geographic Coverage",
                    description=f"Photos span approximately {area_km2:,.0f} km²",
                    location=None,
                    data=coverage_stats,
                    recommendation="Your photography covers diverse geographic areas"
                ))
            
            return coverage_stats
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points on Earth (in km)."""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
    
    def _cluster_to_dict(self, cluster: LocationCluster) -> Dict[str, Any]:
        """Convert cluster to dictionary for JSON serialization."""
        return {
            'center': {
                'latitude': cluster.center_lat,
                'longitude': cluster.center_lon
            },
            'radius_km': cluster.radius_km,
            'photo_count': cluster.photo_count,
            'date_range': {
                'start': cluster.date_range[0].isoformat() if cluster.date_range[0] else None,
                'end': cluster.date_range[1].isoformat() if cluster.date_range[1] else None
            },
            'avg_quality': cluster.avg_quality,
            'name': cluster.location_name
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of location insights."""
        summary = {
            'total_hotspots': len(self.clusters),
            'total_insights': len(self.insights),
            'insight_types': defaultdict(int),
            'recommendations': []
        }
        
        # Count insight types
        for insight in self.insights:
            summary['insight_types'][insight.type] += 1
        
        # Top recommendations
        summary['recommendations'] = [
            insight.recommendation 
            for insight in self.insights[:3]
        ]
        
        return summary
    
    def get_map_data(self) -> Dict[str, Any]:
        """Get data formatted for map visualization."""
        map_data = {
            'hotspots': [],
            'travel_routes': [],
            'quality_heatmap': []
        }
        
        # Format hotspots for mapping
        for cluster in self.clusters:
            map_data['hotspots'].append({
                'lat': cluster.center_lat,
                'lon': cluster.center_lon,
                'radius': cluster.radius_km * 1000,  # Convert to meters
                'intensity': min(cluster.photo_count / 100, 1.0),  # Normalize
                'label': f"{cluster.photo_count} photos",
                'quality': cluster.avg_quality
            })
        
        # Could add travel routes if needed
        # This would require sequential GPS data processing
        
        return map_data