"""
Natural language query builder for PhotoSight MCP.

Converts natural language queries into structured search parameters
for the PhotoOperations.search_photos method.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class NaturalLanguageQueryBuilder:
    """
    Parses natural language queries and converts them to search parameters.
    
    Examples:
    - "sharp portraits with 85mm lens" -> lens_model="85mm", (implied portrait detection)
    - "high ISO photos from last month" -> iso_range=(1600, None), date_range=(last_month, now)
    - "wide angle landscapes with GPS" -> focal_length_range=(14, 35), has_gps=True
    """
    
    # Camera model patterns
    CAMERA_PATTERNS = [
        (r'(?:sony\s*)?a7\s*(?:r)?(?:iii|iv|v)?', 'A7'),
        (r'(?:canon\s*)?(?:eos\s*)?r5', 'R5'),
        (r'(?:nikon\s*)?z9', 'Z9'),
    ]
    
    # Lens patterns
    LENS_PATTERNS = [
        (r'(\d+)(?:-(\d+))?\s*mm', 'focal_length'),
        (r'f/?(\d+(?:\.\d+)?)', 'aperture'),
        (r'(?:sony|sigma|tamron|canon|nikon)\s+\S+', 'lens_model'),
    ]
    
    # ISO patterns
    ISO_RANGES = {
        'low iso': (100, 800),
        'high iso': (1600, None),
        'very high iso': (6400, None),
    }
    
    # Focal length categories
    FOCAL_LENGTH_CATEGORIES = {
        'wide angle': (14, 35),
        'wide': (14, 35),
        'normal': (35, 85),
        'standard': (35, 85),
        'portrait': (85, 135),
        'telephoto': (135, 400),
        'tele': (135, 400),
        'super telephoto': (400, None),
    }
    
    # Time period patterns
    TIME_PERIODS = {
        'today': timedelta(days=1),
        'yesterday': (timedelta(days=2), timedelta(days=1)),
        'this week': timedelta(weeks=1),
        'last week': (timedelta(weeks=2), timedelta(weeks=1)),
        'this month': timedelta(days=30),
        'last month': (timedelta(days=60), timedelta(days=30)),
        'this year': timedelta(days=365),
        'last year': (timedelta(days=730), timedelta(days=365)),
    }
    
    # Quality indicators
    QUALITY_KEYWORDS = {
        'sharp': {'sharpness_min': 0.7},
        'blurry': {'sharpness_max': 0.3},
        'bright': {'brightness_min': 0.6},
        'dark': {'brightness_max': 0.4},
        'accepted': {'status': 'processed'},
        'rejected': {'status': 'rejected'},
    }
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query into search parameters.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary of search parameters for PhotoOperations.search_photos
        """
        query_lower = query.lower()
        params = {}
        
        # Parse camera model
        camera = self._parse_camera(query_lower)
        if camera:
            params['camera_model'] = camera
        
        # Parse lens information
        lens_params = self._parse_lens(query_lower)
        params.update(lens_params)
        
        # Parse ISO
        iso_params = self._parse_iso(query_lower)
        params.update(iso_params)
        
        # Parse time periods
        time_params = self._parse_time(query_lower)
        params.update(time_params)
        
        # Parse quality indicators
        quality_params = self._parse_quality(query_lower)
        params.update(quality_params)
        
        # Parse GPS/location
        if any(word in query_lower for word in ['gps', 'location', 'geotagged', 'map']):
            params['has_gps'] = True
        elif 'no gps' in query_lower or 'without location' in query_lower:
            params['has_gps'] = False
        
        # Parse flash
        if 'flash' in query_lower:
            if 'no flash' in query_lower or 'without flash' in query_lower:
                params['flash_fired'] = False
            else:
                params['flash_fired'] = True
        
        # Parse ordering
        order_params = self._parse_ordering(query_lower)
        params.update(order_params)
        
        # Parse project name
        project_params = self._parse_project(query_lower, query)
        params.update(project_params)
        
        logger.debug(f"Parsed query '{query}' to parameters: {params}")
        return params
    
    def _parse_camera(self, query: str) -> Optional[str]:
        """Parse camera model from query."""
        for pattern, model_prefix in self.CAMERA_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return model_prefix
        return None
    
    def _parse_lens(self, query: str) -> Dict[str, Any]:
        """Parse lens-related parameters."""
        params = {}
        
        # Check for focal length categories first
        for category, (min_fl, max_fl) in self.FOCAL_LENGTH_CATEGORIES.items():
            if category in query:
                params['focal_length_range'] = (min_fl, max_fl)
                break
        
        # Parse specific focal lengths (overrides categories)
        fl_match = re.search(r'(\d+)(?:-(\d+))?\s*mm', query)
        if fl_match:
            if fl_match.group(2):  # Range like "24-70mm"
                params['focal_length_range'] = (
                    int(fl_match.group(1)),
                    int(fl_match.group(2))
                )
            else:  # Single focal length like "85mm"
                focal = int(fl_match.group(1))
                # Allow some tolerance for single focal length
                params['focal_length_range'] = (focal - 5, focal + 5)
        
        # Parse aperture
        aperture_match = re.search(r'f/?(\d+(?:\.\d+)?)', query)
        if aperture_match:
            aperture = float(aperture_match.group(1))
            # Allow some tolerance
            params['aperture_range'] = (aperture - 0.2, aperture + 0.2)
        
        # Parse lens model/brand
        lens_brands = ['sony', 'sigma', 'tamron', 'canon', 'nikon', 'zeiss']
        for brand in lens_brands:
            if brand in query:
                # Find the lens model after the brand
                brand_match = re.search(rf'{brand}\s+(\S+(?:\s+\S+)*?)(?:\s|$)', query)
                if brand_match:
                    params['lens_model'] = f"{brand} {brand_match.group(1)}"
                    break
        
        return params
    
    def _parse_iso(self, query: str) -> Dict[str, Any]:
        """Parse ISO-related parameters."""
        params = {}
        
        # Check for ISO categories
        for category, iso_range in self.ISO_RANGES.items():
            if category in query:
                params['iso_range'] = iso_range
                return params
        
        # Parse specific ISO values
        iso_match = re.search(r'iso\s*(\d+)(?:\s*-\s*(\d+))?', query)
        if iso_match:
            if iso_match.group(2):  # Range like "ISO 800-3200"
                params['iso_range'] = (
                    int(iso_match.group(1)),
                    int(iso_match.group(2))
                )
            else:  # Single ISO like "ISO 1600"
                iso = int(iso_match.group(1))
                # Treat as minimum
                params['iso_range'] = (iso, None)
        
        return params
    
    def _parse_time(self, query: str) -> Dict[str, Any]:
        """Parse time-related parameters."""
        params = {}
        now = datetime.now()
        
        # Check for predefined periods
        for period, delta in self.TIME_PERIODS.items():
            if period in query:
                if isinstance(delta, tuple):  # Range like "last week"
                    params['date_range'] = (
                        now - delta[0],
                        now - delta[1]
                    )
                else:  # Single period like "this week"
                    params['date_range'] = (now - delta, now)
                return params
        
        # Parse specific dates (simplified)
        year_match = re.search(r'(\d{4})', query)
        if year_match:
            year = int(year_match.group(1))
            if 2000 <= year <= now.year:
                # Check for month
                months = ['january', 'february', 'march', 'april', 'may', 'june',
                         'july', 'august', 'september', 'october', 'november', 'december']
                for i, month in enumerate(months):
                    if month in query:
                        start_date = datetime(year, i + 1, 1)
                        if i == 11:  # December
                            end_date = datetime(year + 1, 1, 1)
                        else:
                            end_date = datetime(year, i + 2, 1)
                        params['date_range'] = (start_date, end_date)
                        return params
                
                # Just year
                params['date_range'] = (
                    datetime(year, 1, 1),
                    datetime(year + 1, 1, 1)
                )
        
        return params
    
    def _parse_quality(self, query: str) -> Dict[str, Any]:
        """Parse quality-related parameters."""
        params = {}
        
        for keyword, keyword_params in self.QUALITY_KEYWORDS.items():
            if keyword in query:
                params.update(keyword_params)
        
        # Parse shutter speed
        shutter_match = re.search(r'1/(\d+)\s*(?:s|sec)?', query)
        if shutter_match:
            shutter_speed = 1.0 / int(shutter_match.group(1))
            params['shutter_speed_min'] = shutter_speed
        
        return params
    
    def _parse_ordering(self, query: str) -> Dict[str, Any]:
        """Parse ordering preferences."""
        params = {}
        
        if 'recent' in query or 'latest' in query or 'newest' in query:
            params['order_by'] = 'date_taken'
        elif 'high iso' in query:
            params['order_by'] = 'iso'
        elif 'wide' in query or 'telephoto' in query:
            params['order_by'] = 'focal_length'
        
        return params
    
    def _parse_project(self, query_lower: str, original_query: str) -> Dict[str, Any]:
        """Parse project-related parameters."""
        params = {}
        
        # Look for patterns like "from [project name] project" or "in [project name]"
        # Pattern 1: "from the Smith Wedding project"
        project_match = re.search(r'(?:from|in|for)\s+(?:the\s+)?([^,]+?)\s*(?:project|shoot|assignment)', 
                                 original_query, re.IGNORECASE)
        if project_match:
            params['project_name'] = project_match.group(1).strip()
            return params
        
        # Pattern 2: "[Project Name] photos" at start of query
        if original_query.lower().endswith(' photos') or original_query.lower().endswith(' images'):
            # Extract everything before "photos" or "images"
            words = original_query.split()
            if len(words) >= 2:
                # Check if this might be a project name (capitalized words)
                potential_project = ' '.join(words[:-1])
                if any(word[0].isupper() for word in potential_project.split() if word):
                    params['project_name'] = potential_project.strip()
        
        # Pattern 3: Look for quoted project names
        quote_match = re.search(r'["\']([^"\']+)["\']', original_query)
        if quote_match and 'project' in query_lower:
            params['project_name'] = quote_match.group(1).strip()
        
        return params