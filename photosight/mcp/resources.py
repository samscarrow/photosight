"""
MCP Resources for PhotoSight schema and metadata exposure.

Provides structured information about the database schema and available
metadata fields to help AI assistants understand the data model.
"""

import json
import logging
from typing import Dict, Any, List
from sqlalchemy import inspect

from ..db.models import Base, Photo, AnalysisResult, ProcessingRecipe
from ..db import get_engine

logger = logging.getLogger(__name__)


class SchemaResource:
    """
    Exposes database schema information as MCP resources.
    
    Helps AI assistants understand the structure of the photo database
    and available fields for queries.
    """
    
    def get_resources(self) -> List[Dict[str, Any]]:
        """Get MCP resource definitions for schema."""
        resources = []
        
        # Photo table schema
        resources.append({
            "uri": "photosight://schema/photos",
            "name": "Photos Table Schema",
            "description": "Structure of the main photos table with EXIF metadata",
            "mimeType": "application/json",
            "content": self._get_photo_schema()
        })
        
        # Analysis results schema
        resources.append({
            "uri": "photosight://schema/analysis_results",
            "name": "Analysis Results Schema",
            "description": "Structure of photo analysis results table",
            "mimeType": "application/json",
            "content": self._get_analysis_schema()
        })
        
        # Query examples
        resources.append({
            "uri": "photosight://schema/query_examples",
            "name": "Query Examples",
            "description": "Example natural language queries and their interpretations",
            "mimeType": "application/json",
            "content": self._get_query_examples()
        })
        
        return resources
    
    def _get_photo_schema(self) -> str:
        """Get detailed photo table schema."""
        schema = {
            "table": "photos",
            "description": "Main table storing photo metadata and EXIF information",
            "columns": {
                # Core fields
                "id": {"type": "integer", "description": "Unique photo ID"},
                "file_path": {"type": "text", "description": "Full path to photo file"},
                "filename": {"type": "string", "description": "Photo filename"},
                "file_size": {"type": "bigint", "description": "File size in bytes"},
                
                # EXIF fields
                "camera_make": {"type": "string", "description": "Camera manufacturer"},
                "camera_model": {"type": "string", "description": "Camera model name"},
                "lens_model": {"type": "string", "description": "Lens model name"},
                "iso": {"type": "integer", "description": "ISO sensitivity value"},
                "aperture": {"type": "float", "description": "Aperture f-number (e.g., 2.8)"},
                "shutter_speed_numeric": {"type": "float", "description": "Shutter speed in seconds"},
                "focal_length": {"type": "float", "description": "Focal length in mm"},
                "focal_length_35mm": {"type": "integer", "description": "35mm equivalent focal length"},
                
                # Additional metadata
                "date_taken": {"type": "datetime", "description": "When photo was captured"},
                "gps_latitude": {"type": "float", "description": "GPS latitude coordinate"},
                "gps_longitude": {"type": "float", "description": "GPS longitude coordinate"},
                "flash_fired": {"type": "boolean", "description": "Whether flash was used"},
                "processing_status": {"type": "string", "description": "Photo status: pending, processed, rejected"},
                "rejection_reason": {"type": "text", "description": "Why photo was rejected"}
            },
            "indexes": [
                "camera_model", "lens_model", "iso", "date_taken", 
                "focal_length", "processing_status"
            ],
            "searchable_fields": [
                "camera_model", "lens_model", "iso_range", "aperture_range",
                "focal_length_range", "date_range", "has_gps", "flash_fired"
            ]
        }
        
        return json.dumps(schema, indent=2)
    
    def _get_analysis_schema(self) -> str:
        """Get analysis results table schema."""
        schema = {
            "table": "analysis_results",
            "description": "Stores technical and AI analysis results for photos",
            "columns": {
                "id": {"type": "integer", "description": "Unique analysis ID"},
                "photo_id": {"type": "integer", "description": "Reference to photos.id"},
                "analysis_type": {"type": "string", "description": "Type: technical, ai_curation, similarity"},
                
                # Technical scores
                "sharpness_score": {"type": "float", "description": "Image sharpness (0-1)"},
                "exposure_quality": {"type": "float", "description": "Exposure quality (0-1)"},
                "contrast_score": {"type": "float", "description": "Contrast score (0-1)"},
                
                # AI scores
                "overall_ai_score": {"type": "float", "description": "Overall AI quality score (0-1)"},
                "person_detected": {"type": "boolean", "description": "Whether people were detected"},
                "face_quality_score": {"type": "float", "description": "Face quality if detected (0-1)"},
                "composition_score": {"type": "float", "description": "Composition quality (0-1)"}
            },
            "relationships": {
                "photo": "Many-to-one relationship with photos table"
            }
        }
        
        return json.dumps(schema, indent=2)
    
    def _get_query_examples(self) -> str:
        """Get example queries for AI reference."""
        examples = [
            {
                "natural_language": "Show me sharp portraits taken with my 85mm lens",
                "interpreted_as": {
                    "lens_model": "85mm",
                    "sharpness_min": 0.7,
                    "person_detected": True
                }
            },
            {
                "natural_language": "Find high ISO photos from last month",
                "interpreted_as": {
                    "iso_range": [1600, None],
                    "date_range": ["30 days ago", "now"]
                }
            },
            {
                "natural_language": "Wide angle landscapes with GPS coordinates",
                "interpreted_as": {
                    "focal_length_range": [14, 35],
                    "has_gps": True,
                    "composition_type": "landscape"
                }
            },
            {
                "natural_language": "Photos taken with Sony A7III at f/1.4",
                "interpreted_as": {
                    "camera_model": "A7III",
                    "aperture_range": [1.3, 1.5]
                }
            },
            {
                "natural_language": "Rejected photos due to blur",
                "interpreted_as": {
                    "processing_status": "rejected",
                    "rejection_reason": "blurry"
                }
            }
        ]
        
        return json.dumps({"examples": examples}, indent=2)


class MetadataResource:
    """
    Exposes metadata field information as MCP resources.
    
    Provides detailed information about EXIF fields, their meanings,
    and typical values.
    """
    
    def get_resources(self) -> List[Dict[str, Any]]:
        """Get MCP resource definitions for metadata."""
        resources = []
        
        # EXIF field glossary
        resources.append({
            "uri": "photosight://metadata/exif_glossary",
            "name": "EXIF Field Glossary",
            "description": "Detailed explanation of EXIF metadata fields",
            "mimeType": "application/json",
            "content": self._get_exif_glossary()
        })
        
        # Common values
        resources.append({
            "uri": "photosight://metadata/common_values",
            "name": "Common Metadata Values",
            "description": "Typical values for various metadata fields",
            "mimeType": "application/json",
            "content": self._get_common_values()
        })
        
        # Field relationships
        resources.append({
            "uri": "photosight://metadata/field_relationships",
            "name": "Metadata Field Relationships",
            "description": "How different metadata fields relate to each other",
            "mimeType": "application/json",
            "content": self._get_field_relationships()
        })
        
        return resources
    
    def _get_exif_glossary(self) -> str:
        """Get EXIF field explanations."""
        glossary = {
            "camera_fields": {
                "camera_make": "Camera manufacturer (e.g., Sony, Canon, Nikon)",
                "camera_model": "Specific camera model (e.g., A7III, R5, Z9)",
                "lens_model": "Lens model including focal range and aperture"
            },
            "exposure_fields": {
                "iso": "Sensor sensitivity - higher values for low light",
                "aperture": "Lens opening size (f-number) - smaller = wider opening",
                "shutter_speed_numeric": "Exposure time in seconds (e.g., 0.005 = 1/200s)",
                "exposure_compensation": "Manual exposure adjustment in stops"
            },
            "focal_fields": {
                "focal_length": "Actual lens focal length in mm",
                "focal_length_35mm": "Equivalent focal length on full-frame sensor"
            },
            "technical_fields": {
                "flash_fired": "Whether flash was triggered during capture",
                "white_balance": "Color temperature setting (Auto, Daylight, etc.)",
                "metering_mode": "How camera measured light (Spot, Matrix, etc.)",
                "focus_mode": "AF-S (single), AF-C (continuous), Manual"
            },
            "location_fields": {
                "gps_latitude": "Decimal degrees latitude (-90 to 90)",
                "gps_longitude": "Decimal degrees longitude (-180 to 180)",
                "gps_altitude": "Elevation in meters above sea level"
            }
        }
        
        return json.dumps(glossary, indent=2)
    
    def _get_common_values(self) -> str:
        """Get common metadata values for reference."""
        common_values = {
            "iso_values": {
                "low": [100, 200, 400],
                "medium": [800, 1600],
                "high": [3200, 6400],
                "very_high": [12800, 25600, 51200]
            },
            "aperture_values": {
                "wide": [1.2, 1.4, 1.8, 2.0, 2.8],
                "medium": [4.0, 5.6],
                "narrow": [8.0, 11.0, 16.0]
            },
            "focal_lengths": {
                "ultra_wide": [14, 16, 20, 24],
                "wide": [28, 35],
                "normal": [50],
                "portrait": [85, 105, 135],
                "telephoto": [200, 300, 400, 600]
            },
            "shutter_speeds": {
                "fast": ["1/2000", "1/1000", "1/500"],
                "medium": ["1/250", "1/125", "1/60"],
                "slow": ["1/30", "1/15", "1/8", "1/4"],
                "long": ["1/2", "1s", "2s", "4s"]
            }
        }
        
        return json.dumps(common_values, indent=2)
    
    def _get_field_relationships(self) -> str:
        """Get metadata field relationships."""
        relationships = {
            "exposure_triangle": {
                "description": "ISO, aperture, and shutter speed work together",
                "relationships": [
                    "Higher ISO allows faster shutter speeds",
                    "Wider aperture (lower f-number) allows faster shutter speeds",
                    "Longer shutter speeds allow lower ISO or narrower aperture"
                ]
            },
            "focal_length_effects": {
                "description": "How focal length affects other parameters",
                "relationships": [
                    "Longer focal length requires faster shutter speed (1/focal_length rule)",
                    "Wider focal lengths typically have wider maximum aperture",
                    "Telephoto lenses compress perspective, wide angles expand it"
                ]
            },
            "quality_indicators": {
                "description": "Metadata that affects image quality",
                "relationships": [
                    "Higher ISO typically means more noise",
                    "Very wide apertures may reduce edge sharpness",
                    "Slow shutter speeds risk motion blur"
                ]
            }
        }
        
        return json.dumps(relationships, indent=2)