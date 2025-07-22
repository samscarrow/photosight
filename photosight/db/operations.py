"""
Database operations for PhotoSight.

Provides high-level CRUD operations and business logic
for managing photos, analysis results, and processing recipes.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError

from .models import (
    Photo, AnalysisResult, ProcessingRecipe, BatchSession,
    FaceDetection, SimilarityGroup, PhotoSimilarity, CompositionAnalysis,
    Project, Task, ProjectStatus, TaskStatus, TaskPriority
)
from .connection import get_session

logger = logging.getLogger(__name__)


class PhotoOperations:
    """Operations for managing photos in the database."""
    
    @staticmethod
    def create_photo(
        file_path: str,
        filename: str,
        metadata: Dict[str, Any],
        exif_data: Optional[Dict] = None
    ) -> Optional[Photo]:
        """
        Create a new photo record with promoted EXIF fields.
        
        Args:
            file_path: Full path to the photo file
            filename: Just the filename
            metadata: Photo metadata from analysis pipeline
            exif_data: Complete EXIF data dictionary
            
        Returns:
            Photo: Created photo object or None if failed
        """
        try:
            with get_session() as session:
                # Check if photo already exists
                existing = session.query(Photo).filter(Photo.file_path == file_path).first()
                if existing:
                    logger.warning(f"Photo already exists: {file_path}")
                    return existing
                
                # Create photo with promoted EXIF fields
                photo = Photo(
                    file_path=file_path,
                    filename=filename,
                    file_size=metadata.get('file_size'),
                    checksum=metadata.get('checksum'),
                    
                    # Promoted EXIF fields for efficient querying
                    image_width=PhotoOperations._safe_int(exif_data, 'EXIF:ExifImageWidth'),
                    image_height=PhotoOperations._safe_int(exif_data, 'EXIF:ExifImageHeight'),
                    date_taken=PhotoOperations._parse_exif_date(exif_data, 'EXIF:DateTimeOriginal'),
                    camera_make=PhotoOperations._safe_str(exif_data, 'EXIF:Make'),
                    camera_model=PhotoOperations._safe_str(exif_data, 'EXIF:Model'),
                    lens_model=PhotoOperations._safe_str(exif_data, 'EXIF:LensModel'),
                    iso=PhotoOperations._safe_int(exif_data, 'EXIF:ISO'),
                    aperture=PhotoOperations._parse_aperture(exif_data),
                    shutter_speed_numeric=PhotoOperations._parse_shutter_speed_numeric(exif_data),
                    shutter_speed_display=PhotoOperations._safe_str(exif_data, 'EXIF:ShutterSpeedValue'),
                    focal_length=PhotoOperations._safe_float(exif_data, 'EXIF:FocalLength'),
                    focal_length_35mm=PhotoOperations._safe_int(exif_data, 'EXIF:FocalLengthIn35mmFormat'),
                    
                    # Additional technical metadata
                    flash_fired=PhotoOperations._parse_flash_fired(exif_data),
                    white_balance=PhotoOperations._safe_str(exif_data, 'EXIF:WhiteBalance'),
                    metering_mode=PhotoOperations._safe_str(exif_data, 'EXIF:MeteringMode'),
                    exposure_mode=PhotoOperations._safe_str(exif_data, 'EXIF:ExposureMode'),
                    exposure_compensation=PhotoOperations._safe_float(exif_data, 'EXIF:ExposureCompensation'),
                    
                    # Image quality and format
                    bit_depth=PhotoOperations._safe_int(exif_data, 'EXIF:BitsPerSample'),
                    compression=PhotoOperations._safe_str(exif_data, 'EXIF:Compression'),
                    color_space=PhotoOperations._safe_str(exif_data, 'EXIF:ColorSpace'),
                    
                    # GPS coordinates (if available)
                    gps_latitude=PhotoOperations._parse_gps_coordinate(exif_data, 'GPS:GPSLatitude', 'GPS:GPSLatitudeRef'),
                    gps_longitude=PhotoOperations._parse_gps_coordinate(exif_data, 'GPS:GPSLongitude', 'GPS:GPSLongitudeRef'),
                    gps_altitude=PhotoOperations._safe_float(exif_data, 'GPS:GPSAltitude'),
                    
                    # Store complete data
                    exif_data=exif_data or {},
                    raw_metadata=metadata
                )
                
                session.add(photo)
                session.flush()  # Get ID before commit
                logger.info(f"Created photo record: {photo.id} - {filename}")
                return photo
                
        except Exception as e:
            logger.error(f"Failed to create photo record: {e}")
            return None
    
    @staticmethod
    def _safe_str(data: Optional[Dict], key: str) -> Optional[str]:
        """Safely extract string value from EXIF data."""
        if not data or key not in data:
            return None
        value = data[key]
        return str(value).strip() if value else None
    
    @staticmethod
    def _safe_int(data: Optional[Dict], key: str) -> Optional[int]:
        """Safely extract integer value from EXIF data."""
        if not data or key not in data:
            return None
        try:
            return int(float(str(data[key])))
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _safe_float(data: Optional[Dict], key: str) -> Optional[float]:
        """Safely extract float value from EXIF data."""
        if not data or key not in data:
            return None
        try:
            return float(str(data[key]))
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _parse_exif_date(data: Optional[Dict], key: str) -> Optional[datetime]:
        """Parse EXIF date string to datetime object."""
        if not data or key not in data:
            return None
        try:
            date_str = str(data[key])
            # EXIF format is typically 'YYYY:MM:DD HH:MM:SS'
            return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _parse_aperture(data: Optional[Dict]) -> Optional[float]:
        """Parse aperture value from EXIF data."""
        # Try FNumber first, then ApertureValue
        for key in ['EXIF:FNumber', 'EXIF:ApertureValue']:
            if data and key in data:
                try:
                    value = str(data[key])
                    # Handle fraction format like "28/10" -> 2.8
                    if '/' in value:
                        num, den = value.split('/')
                        return float(num) / float(den)
                    return float(value)
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
        return None
    
    @staticmethod
    def _parse_shutter_speed_numeric(data: Optional[Dict]) -> Optional[float]:
        """Parse shutter speed to numeric value in seconds."""
        if not data:
            return None
        
        # Try ExposureTime first (most direct)
        for key in ['EXIF:ExposureTime', 'EXIF:ShutterSpeedValue']:
            if key in data:
                try:
                    value = str(data[key])
                    # Handle fraction format like "1/200" -> 0.005
                    if '/' in value:
                        num, den = value.split('/')
                        return float(num) / float(den)
                    return float(value)
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
        return None
    
    @staticmethod
    def _parse_flash_fired(data: Optional[Dict]) -> Optional[bool]:
        """Parse flash fired status from EXIF data."""
        if not data or 'EXIF:Flash' not in data:
            return None
        try:
            flash_value = data['EXIF:Flash']
            # Flash value is a bitmask, bit 0 indicates if flash fired
            return bool(int(flash_value) & 1)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _parse_gps_coordinate(data: Optional[Dict], coord_key: str, ref_key: str) -> Optional[float]:
        """Parse GPS coordinate from EXIF data."""
        if not data or coord_key not in data or ref_key not in data:
            return None
        try:
            coord_str = str(data[coord_key])
            ref = str(data[ref_key])
            
            # Parse coordinate string like "40 deg 45' 30.00\" N"
            # or decimal format
            if 'deg' in coord_str:
                # DMS format: degrees minutes seconds
                parts = coord_str.replace('deg', '').replace("'", '').replace('"', '').split()
                degrees = float(parts[0])
                minutes = float(parts[1]) if len(parts) > 1 else 0
                seconds = float(parts[2]) if len(parts) > 2 else 0
                coordinate = degrees + minutes/60 + seconds/3600
            else:
                coordinate = float(coord_str)
            
            # Apply hemisphere reference
            if ref.upper() in ['S', 'W']:
                coordinate = -coordinate
                
            return coordinate
        except (ValueError, TypeError, IndexError):
            return None
    
    @staticmethod
    def get_photo_by_path(file_path: str) -> Optional[Photo]:
        """Get photo by file path."""
        try:
            with get_session() as session:
                return session.query(Photo).filter(Photo.file_path == file_path).first()
        except Exception as e:
            logger.error(f"Failed to get photo by path: {e}")
            return None
    
    @staticmethod
    def get_photo_by_id(photo_id: int) -> Optional[Photo]:
        """Get photo by ID."""
        try:
            with get_session() as session:
                return session.query(Photo).filter(Photo.id == photo_id).first()
        except Exception as e:
            logger.error(f"Failed to get photo by ID: {e}")
            return None
    
    @staticmethod
    def update_processing_status(photo_id: int, status: str, reason: Optional[str] = None) -> bool:
        """Update photo processing status."""
        try:
            with get_session() as session:
                photo = session.query(Photo).filter(Photo.id == photo_id).first()
                if photo:
                    photo.processing_status = status
                    photo.last_analyzed = datetime.utcnow()
                    if reason:
                        photo.rejection_reason = reason
                    logger.debug(f"Updated photo {photo_id} status to {status}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to update photo status: {e}")
            return False
    
    @staticmethod
    def get_photos_by_status(status: str) -> List[Photo]:
        """Get all photos with a specific processing status."""
        try:
            with get_session() as session:
                return session.query(Photo).filter(Photo.processing_status == status).all()
        except Exception as e:
            logger.error(f"Failed to get photos by status: {e}")
            return []
    
    @staticmethod
    def search_photos(
        camera_model: Optional[str] = None,
        lens_model: Optional[str] = None,
        iso_range: Optional[Tuple[int, int]] = None,
        aperture_range: Optional[Tuple[float, float]] = None,
        focal_length_range: Optional[Tuple[float, float]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        status: Optional[str] = None,
        has_gps: Optional[bool] = None,
        image_size_min: Optional[Tuple[int, int]] = None,  # (width, height)
        shutter_speed_min: Optional[float] = None,  # minimum speed in seconds
        flash_fired: Optional[bool] = None,
        project_name: Optional[str] = None,  # Project filter
        limit: int = 100,
        order_by: str = 'date_taken'
    ) -> List[Photo]:
        """Enhanced photo search with comprehensive EXIF-based filtering."""
        try:
            with get_session() as session:
                query = session.query(Photo)
                
                # Text-based filters
                if camera_model:
                    query = query.filter(Photo.camera_model.ilike(f"%{camera_model}%"))
                
                if lens_model:
                    query = query.filter(Photo.lens_model.ilike(f"%{lens_model}%"))
                
                # Numeric range filters
                if iso_range:
                    query = query.filter(
                        and_(Photo.iso >= iso_range[0], Photo.iso <= iso_range[1])
                    )
                
                if aperture_range:
                    query = query.filter(
                        and_(Photo.aperture >= aperture_range[0], Photo.aperture <= aperture_range[1])
                    )
                
                if focal_length_range:
                    query = query.filter(
                        and_(Photo.focal_length >= focal_length_range[0], 
                             Photo.focal_length <= focal_length_range[1])
                    )
                
                if shutter_speed_min:
                    query = query.filter(Photo.shutter_speed_numeric >= shutter_speed_min)
                
                # Date filters
                if date_range:
                    query = query.filter(
                        and_(Photo.date_taken >= date_range[0], Photo.date_taken <= date_range[1])
                    )
                
                # Boolean filters
                if status:
                    query = query.filter(Photo.processing_status == status)
                
                if has_gps is not None:
                    if has_gps:
                        query = query.filter(
                            and_(Photo.gps_latitude.isnot(None), Photo.gps_longitude.isnot(None))
                        )
                    else:
                        query = query.filter(
                            or_(Photo.gps_latitude.is_(None), Photo.gps_longitude.is_(None))
                        )
                
                if flash_fired is not None:
                    query = query.filter(Photo.flash_fired == flash_fired)
                
                # Image size filters
                if image_size_min:
                    query = query.filter(
                        and_(Photo.image_width >= image_size_min[0], 
                             Photo.image_height >= image_size_min[1])
                    )
                
                # Project filter
                if project_name:
                    query = query.join(
                        Project, Photo.project_id == Project.id
                    ).filter(Project.name == project_name)
                
                # Ordering
                if order_by == 'date_taken':
                    query = query.order_by(desc(Photo.date_taken))
                elif order_by == 'iso':
                    query = query.order_by(desc(Photo.iso))
                elif order_by == 'focal_length':
                    query = query.order_by(desc(Photo.focal_length))
                elif order_by == 'aperture':
                    query = query.order_by(Photo.aperture)
                else:
                    query = query.order_by(desc(Photo.imported_at))
                
                return query.limit(limit).all()
                
        except Exception as e:
            logger.error(f"Failed to search photos: {e}")
            return []
    
    @staticmethod
    def get_photos_by_camera(camera_model: str, limit: int = 50) -> List[Photo]:
        """Get all photos taken with a specific camera model."""
        try:
            with get_session() as session:
                return session.query(Photo).filter(
                    Photo.camera_model.ilike(f"%{camera_model}%")
                ).order_by(desc(Photo.date_taken)).limit(limit).all()
        except Exception as e:
            logger.error(f"Failed to get photos by camera: {e}")
            return []
    
    @staticmethod
    def get_photos_by_lens(lens_model: str, limit: int = 50) -> List[Photo]:
        """Get all photos taken with a specific lens."""
        try:
            with get_session() as session:
                return session.query(Photo).filter(
                    Photo.lens_model.ilike(f"%{lens_model}%")
                ).order_by(desc(Photo.date_taken)).limit(limit).all()
        except Exception as e:
            logger.error(f"Failed to get photos by lens: {e}")
            return []
    
    @staticmethod
    def get_high_iso_photos(iso_threshold: int = 1600, limit: int = 50) -> List[Photo]:
        """Get photos taken at high ISO settings."""
        try:
            with get_session() as session:
                return session.query(Photo).filter(
                    Photo.iso >= iso_threshold
                ).order_by(desc(Photo.iso)).limit(limit).all()
        except Exception as e:
            logger.error(f"Failed to get high ISO photos: {e}")
            return []
    
    @staticmethod
    def get_photos_with_gps(limit: int = 50) -> List[Photo]:
        """Get photos that have GPS coordinates."""
        try:
            with get_session() as session:
                return session.query(Photo).filter(
                    and_(Photo.gps_latitude.isnot(None), Photo.gps_longitude.isnot(None))
                ).order_by(desc(Photo.date_taken)).limit(limit).all()
        except Exception as e:
            logger.error(f"Failed to get photos with GPS: {e}")
            return []
    
    @staticmethod
    def get_gear_statistics(project_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about camera gear usage."""
        try:
            with get_session() as session:
                # Base queries
                camera_query = session.query(
                    Photo.camera_model, func.count(Photo.id).label('count')
                ).filter(Photo.camera_model.isnot(None))
                
                lens_query = session.query(
                    Photo.lens_model, func.count(Photo.id).label('count')
                ).filter(Photo.lens_model.isnot(None))
                
                focal_query = session.query(
                    Photo.focal_length_35mm, func.count(Photo.id).label('count')
                ).filter(Photo.focal_length_35mm.isnot(None))
                
                iso_query = session.query(
                    Photo.iso, func.count(Photo.id).label('count')
                ).filter(Photo.iso.isnot(None))
                
                # Apply project filter if specified
                if project_name:
                    camera_query = camera_query.join(
                        Project, Photo.project_id == Project.id
                    ).filter(Project.name == project_name)
                    
                    lens_query = lens_query.join(
                        Project, Photo.project_id == Project.id
                    ).filter(Project.name == project_name)
                    
                    focal_query = focal_query.join(
                        Project, Photo.project_id == Project.id
                    ).filter(Project.name == project_name)
                    
                    iso_query = iso_query.join(
                        Project, Photo.project_id == Project.id
                    ).filter(Project.name == project_name)
                
                # Execute queries
                camera_stats = camera_query.group_by(
                    Photo.camera_model
                ).order_by(desc('count')).all()
                
                lens_stats = lens_query.group_by(
                    Photo.lens_model
                ).order_by(desc('count')).all()
                
                focal_length_stats = focal_query.group_by(
                    Photo.focal_length_35mm
                ).order_by(Photo.focal_length_35mm).all()
                
                iso_stats = iso_query.group_by(
                    Photo.iso
                ).order_by(Photo.iso).all()
                
                return {
                    'cameras': [{'model': r[0], 'count': r[1]} for r in camera_stats],
                    'lenses': [{'model': r[0], 'count': r[1]} for r in lens_stats],
                    'focal_lengths': [{'focal_length': r[0], 'count': r[1]} for r in focal_length_stats],
                    'iso_usage': [{'iso': r[0], 'count': r[1]} for r in iso_stats]
                }
                
        except Exception as e:
            logger.error(f"Failed to get gear statistics: {e}")
            return {}
    
    @staticmethod
    def get_shooting_statistics() -> Dict[str, Any]:
        """Get detailed shooting pattern statistics."""
        try:
            with get_session() as session:
                # Total photo count
                total_photos = session.query(func.count(Photo.id)).scalar()
                
                # Photos with GPS
                gps_photos = session.query(func.count(Photo.id)).filter(
                    and_(Photo.gps_latitude.isnot(None), Photo.gps_longitude.isnot(None))
                ).scalar()
                
                # Flash usage
                flash_photos = session.query(func.count(Photo.id)).filter(
                    Photo.flash_fired == True
                ).scalar()
                
                # Average settings
                avg_stats = session.query(
                    func.avg(Photo.iso).label('avg_iso'),
                    func.avg(Photo.aperture).label('avg_aperture'),
                    func.avg(Photo.focal_length).label('avg_focal_length'),
                    func.avg(Photo.shutter_speed_numeric).label('avg_shutter_speed')
                ).first()
                
                # Date range
                date_range = session.query(
                    func.min(Photo.date_taken).label('earliest'),
                    func.max(Photo.date_taken).label('latest')
                ).first()
                
                return {
                    'total_photos': total_photos,
                    'photos_with_gps': gps_photos,
                    'gps_percentage': (gps_photos / total_photos * 100) if total_photos else 0,
                    'flash_photos': flash_photos,
                    'flash_percentage': (flash_photos / total_photos * 100) if total_photos else 0,
                    'average_iso': float(avg_stats.avg_iso) if avg_stats.avg_iso else None,
                    'average_aperture': float(avg_stats.avg_aperture) if avg_stats.avg_aperture else None,
                    'average_focal_length': float(avg_stats.avg_focal_length) if avg_stats.avg_focal_length else None,
                    'average_shutter_speed': float(avg_stats.avg_shutter_speed) if avg_stats.avg_shutter_speed else None,
                    'date_range': {
                        'earliest': date_range.earliest.isoformat() if date_range.earliest else None,
                        'latest': date_range.latest.isoformat() if date_range.latest else None
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get shooting statistics: {e}")
            return {}


class AnalysisOperations:
    """Operations for managing analysis results."""
    
    @staticmethod
    def save_technical_analysis(
        photo_id: int,
        results: Dict[str, Any],
        analysis_version: str = "1.0"
    ) -> Optional[AnalysisResult]:
        """Save technical analysis results."""
        try:
            with get_session() as session:
                analysis = AnalysisResult(
                    photo_id=photo_id,
                    analysis_type="technical",
                    analysis_version=analysis_version,
                    sharpness_score=results.get('sharpness_score'),
                    blur_score=results.get('blur_score'),
                    exposure_quality=results.get('exposure_quality'),
                    contrast_score=results.get('contrast_score'),
                    noise_level=results.get('noise_level'),
                    histogram_data=results.get('histogram_data'),
                    black_clip_percentage=results.get('black_clip_percentage'),
                    white_clip_percentage=results.get('white_clip_percentage'),
                    mean_brightness=results.get('mean_brightness'),
                    shadow_percentage=results.get('shadow_percentage'),
                    highlight_percentage=results.get('highlight_percentage'),
                    processing_time_ms=results.get('processing_time_ms'),
                    analysis_data=results.get('additional_data', {})
                )
                
                session.add(analysis)
                session.flush()
                logger.debug(f"Saved technical analysis for photo {photo_id}")
                return analysis
                
        except Exception as e:
            logger.error(f"Failed to save technical analysis: {e}")
            return None
    
    @staticmethod
    def save_ai_analysis(
        photo_id: int,
        results: Dict[str, Any],
        analysis_version: str = "1.0"
    ) -> Optional[AnalysisResult]:
        """Save AI curation analysis results."""
        try:
            with get_session() as session:
                analysis = AnalysisResult(
                    photo_id=photo_id,
                    analysis_type="ai_curation",
                    analysis_version=analysis_version,
                    overall_ai_score=results.get('overall_score'),
                    person_detected=results.get('person_detected', False),
                    person_count=results.get('person_count', 0),
                    face_quality_score=results.get('face_quality_score'),
                    composition_score=results.get('composition_score'),
                    aesthetic_score=results.get('aesthetic_score'),
                    processing_time_ms=results.get('processing_time_ms'),
                    analysis_data=results.get('additional_data', {})
                )
                
                session.add(analysis)
                session.flush()
                logger.debug(f"Saved AI analysis for photo {photo_id}")
                return analysis
                
        except Exception as e:
            logger.error(f"Failed to save AI analysis: {e}")
            return None
    
    @staticmethod
    def get_latest_analysis(photo_id: int, analysis_type: str) -> Optional[AnalysisResult]:
        """Get the latest analysis result for a photo and type."""
        try:
            with get_session() as session:
                return session.query(AnalysisResult).filter(
                    and_(
                        AnalysisResult.photo_id == photo_id,
                        AnalysisResult.analysis_type == analysis_type
                    )
                ).order_by(desc(AnalysisResult.created_at)).first()
        except Exception as e:
            logger.error(f"Failed to get latest analysis: {e}")
            return None
    
    @staticmethod
    def get_photos_by_score_range(
        analysis_type: str,
        score_field: str,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        limit: int = 100
    ) -> List[Tuple[Photo, AnalysisResult]]:
        """Get photos within a specific score range."""
        try:
            with get_session() as session:
                query = session.query(Photo, AnalysisResult).join(
                    AnalysisResult, Photo.id == AnalysisResult.photo_id
                ).filter(AnalysisResult.analysis_type == analysis_type)
                
                score_column = getattr(AnalysisResult, score_field, None)
                if not score_column:
                    logger.error(f"Invalid score field: {score_field}")
                    return []
                
                if min_score is not None:
                    query = query.filter(score_column >= min_score)
                if max_score is not None:
                    query = query.filter(score_column <= max_score)
                
                return query.order_by(desc(score_column)).limit(limit).all()
                
        except Exception as e:
            logger.error(f"Failed to get photos by score range: {e}")
            return []


class RecipeOperations:
    """Operations for managing processing recipes."""
    
    @staticmethod
    def save_recipe(
        photo_id: int,
        recipe_name: str,
        recipe_data: Dict[str, Any],
        output_file_path: Optional[str] = None,
        processing_time_ms: Optional[int] = None
    ) -> Optional[ProcessingRecipe]:
        """Save a processing recipe."""
        try:
            with get_session() as session:
                # Get next iteration number
                latest = session.query(ProcessingRecipe).filter(
                    ProcessingRecipe.photo_id == photo_id
                ).order_by(desc(ProcessingRecipe.iteration_number)).first()
                
                next_iteration = (latest.iteration_number + 1) if latest else 1
                
                # Generate recipe hash for deduplication
                import hashlib
                import json
                recipe_json = json.dumps(recipe_data, sort_keys=True)
                recipe_hash = hashlib.sha256(recipe_json.encode()).hexdigest()
                
                recipe = ProcessingRecipe(
                    photo_id=photo_id,
                    recipe_name=recipe_name,
                    iteration_number=next_iteration,
                    recipe_data=recipe_data,
                    recipe_hash=recipe_hash,
                    output_file_path=output_file_path,
                    processing_time_ms=processing_time_ms,
                    created_by="photosight-v1.0"
                )
                
                session.add(recipe)
                session.flush()
                logger.debug(f"Saved recipe {recipe_name} iteration {next_iteration} for photo {photo_id}")
                return recipe
                
        except Exception as e:
            logger.error(f"Failed to save recipe: {e}")
            return None
    
    @staticmethod
    def get_latest_recipe(photo_id: int) -> Optional[ProcessingRecipe]:
        """Get the latest recipe for a photo."""
        try:
            with get_session() as session:
                return session.query(ProcessingRecipe).filter(
                    ProcessingRecipe.photo_id == photo_id
                ).order_by(desc(ProcessingRecipe.iteration_number)).first()
        except Exception as e:
            logger.error(f"Failed to get latest recipe: {e}")
            return None
    
    @staticmethod
    def get_recipe_history(photo_id: int) -> List[ProcessingRecipe]:
        """Get all recipe iterations for a photo."""
        try:
            with get_session() as session:
                return session.query(ProcessingRecipe).filter(
                    ProcessingRecipe.photo_id == photo_id
                ).order_by(asc(ProcessingRecipe.iteration_number)).all()
        except Exception as e:
            logger.error(f"Failed to get recipe history: {e}")
            return []


class BatchOperations:
    """Operations for managing batch processing sessions."""
    
    @staticmethod
    def create_batch_session(
        session_name: str,
        config_snapshot: Dict[str, Any],
        processing_stages: List[str]
    ) -> Optional[BatchSession]:
        """Create a new batch processing session."""
        try:
            with get_session() as session:
                batch = BatchSession(
                    session_name=session_name,
                    config_snapshot=config_snapshot,
                    processing_stages=processing_stages,
                    status="running"
                )
                
                session.add(batch)
                session.flush()
                logger.info(f"Created batch session: {batch.id} - {session_name}")
                return batch
                
        except Exception as e:
            logger.error(f"Failed to create batch session: {e}")
            return None
    
    @staticmethod
    def update_batch_stats(
        batch_id: int,
        total_photos: Optional[int] = None,
        processed_photos: Optional[int] = None,
        accepted_photos: Optional[int] = None,
        rejected_photos: Optional[int] = None
    ) -> bool:
        """Update batch processing statistics."""
        try:
            with get_session() as session:
                batch = session.query(BatchSession).filter(BatchSession.id == batch_id).first()
                if not batch:
                    return False
                
                if total_photos is not None:
                    batch.total_photos = total_photos
                if processed_photos is not None:
                    batch.processed_photos = processed_photos
                if accepted_photos is not None:
                    batch.accepted_photos = accepted_photos
                if rejected_photos is not None:
                    batch.rejected_photos = rejected_photos
                
                logger.debug(f"Updated batch {batch_id} stats")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update batch stats: {e}")
            return False
    
    @staticmethod
    def complete_batch_session(
        batch_id: int,
        summary_report: Dict[str, Any],
        status: str = "completed"
    ) -> bool:
        """Complete a batch processing session."""
        try:
            with get_session() as session:
                batch = session.query(BatchSession).filter(BatchSession.id == batch_id).first()
                if not batch:
                    return False
                
                batch.end_time = datetime.utcnow()
                batch.status = status
                batch.summary_report = summary_report
                
                # Calculate total processing time
                if batch.start_time and batch.end_time:
                    total_time = (batch.end_time - batch.start_time).total_seconds() * 1000
                    batch.total_processing_time_ms = int(total_time)
                
                # Calculate average processing time
                if batch.processed_photos and batch.processed_photos > 0:
                    batch.average_processing_time_ms = batch.total_processing_time_ms / batch.processed_photos
                
                logger.info(f"Completed batch session: {batch_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to complete batch session: {e}")
            return False


class ProjectOperations:
    """Operations for managing photography projects."""
    
    @staticmethod
    def create_project(
        name: str,
        description: Optional[str] = None,
        client_name: Optional[str] = None,
        project_type: Optional[str] = None,
        shoot_date: Optional[datetime] = None,
        due_date: Optional[datetime] = None,
        budget: Optional[float] = None,
        location: Optional[str] = None,
        expected_photos: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Project]:
        """
        Create a new photography project.
        
        Args:
            name: Unique project name
            description: Project description
            client_name: Client name for the project
            project_type: Type of project (wedding, portrait, etc.)
            shoot_date: Date of the shoot
            due_date: Project due date
            budget: Project budget
            location: Shoot location
            expected_photos: Expected number of deliverable photos
            metadata: Additional custom metadata
            
        Returns:
            Created Project object or None if failed
        """
        try:
            with get_session() as session:
                # Check if project already exists
                existing = session.query(Project).filter(Project.name == name).first()
                if existing:
                    logger.warning(f"Project already exists: {name}")
                    return None
                
                project = Project(
                    name=name,
                    description=description,
                    client_name=client_name,
                    project_type=project_type,
                    shoot_date=shoot_date,
                    due_date=due_date,
                    budget=budget,
                    location=location,
                    expected_photos=expected_photos,
                    status=ProjectStatus.PLANNING,
                    metadata=metadata or {}
                )
                
                session.add(project)
                session.flush()
                logger.info(f"Created project: {project.id} - {name}")
                return project
                
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return None
    
    @staticmethod
    def assign_photos_to_project(
        project_name: str,
        photo_ids: Optional[List[int]] = None,
        photo_paths: Optional[List[str]] = None
    ) -> int:
        """
        Assign photos to a project.
        
        Args:
            project_name: Name of the project
            photo_ids: List of photo IDs to assign
            photo_paths: List of photo file paths to assign
            
        Returns:
            Number of photos assigned
        """
        try:
            with get_session() as session:
                project = session.query(Project).filter(Project.name == project_name).first()
                if not project:
                    raise ValueError(f"Project '{project_name}' not found")
                
                count = 0
                
                # Assign by IDs
                if photo_ids:
                    result = session.query(Photo).filter(
                        Photo.id.in_(photo_ids),
                        Photo.project_id.is_(None)  # Only assign unassigned photos
                    ).update({"project_id": project.id}, synchronize_session=False)
                    count += result
                
                # Assign by paths
                if photo_paths:
                    result = session.query(Photo).filter(
                        Photo.file_path.in_(photo_paths),
                        Photo.project_id.is_(None)
                    ).update({"project_id": project.id}, synchronize_session=False)
                    count += result
                
                session.commit()
                logger.info(f"Assigned {count} photos to project '{project_name}'")
                return count
                
        except Exception as e:
            logger.error(f"Failed to assign photos to project: {e}")
            return 0
    
    @staticmethod
    def update_project_status(
        project_name: str,
        status: ProjectStatus,
        notes: Optional[str] = None
    ) -> bool:
        """Update project status."""
        try:
            with get_session() as session:
                project = session.query(Project).filter(Project.name == project_name).first()
                if not project:
                    return False
                
                project.status = status
                if notes:
                    project.notes = notes
                
                # Set completion date if completed
                if status == ProjectStatus.COMPLETED:
                    project.completed_at = datetime.utcnow()
                
                logger.info(f"Updated project '{project_name}' status to {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update project status: {e}")
            return False
    
    @staticmethod
    def get_project_by_name(name: str) -> Optional[Project]:
        """Get project by name."""
        with get_session() as session:
            return session.query(Project).filter(Project.name == name).first()
    
    @staticmethod
    def list_projects(
        status: Optional[ProjectStatus] = None,
        client_name: Optional[str] = None,
        project_type: Optional[str] = None,
        active_only: bool = False
    ) -> List[Project]:
        """
        List projects with optional filters.
        
        Args:
            status: Filter by project status
            client_name: Filter by client name
            project_type: Filter by project type
            active_only: Only show active projects
            
        Returns:
            List of projects matching criteria
        """
        with get_session() as session:
            query = session.query(Project)
            
            if status:
                query = query.filter(Project.status == status)
            elif active_only:
                query = query.filter(Project.status.in_([
                    ProjectStatus.PLANNING,
                    ProjectStatus.ACTIVE,
                    ProjectStatus.ON_HOLD
                ]))
            
            if client_name:
                query = query.filter(Project.client_name.ilike(f"%{client_name}%"))
            
            if project_type:
                query = query.filter(Project.project_type == project_type)
            
            return query.order_by(Project.created_at.desc()).all()
    
    @staticmethod
    def get_project_statistics(project_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a project."""
        try:
            with get_session() as session:
                project = session.query(Project).filter(Project.name == project_name).first()
                if not project:
                    return {}
                
                # Photo statistics
                total_photos = session.query(func.count(Photo.id)).filter(
                    Photo.project_id == project.id
                ).scalar() or 0
                
                accepted_photos = session.query(func.count(Photo.id)).filter(
                    Photo.project_id == project.id,
                    Photo.processing_status == ProcessingStatus.PROCESSED
                ).scalar() or 0
                
                rejected_photos = session.query(func.count(Photo.id)).filter(
                    Photo.project_id == project.id,
                    Photo.processing_status == ProcessingStatus.REJECTED
                ).scalar() or 0
                
                # Task statistics
                total_tasks = session.query(func.count(Task.id)).filter(
                    Task.project_id == project.id
                ).scalar() or 0
                
                completed_tasks = session.query(func.count(Task.id)).filter(
                    Task.project_id == project.id,
                    Task.status == TaskStatus.COMPLETED
                ).scalar() or 0
                
                # Progress calculation
                progress = 0
                if project.expected_photos and project.expected_photos > 0:
                    progress = min(100, (accepted_photos / project.expected_photos) * 100)
                
                return {
                    'project_id': project.id,
                    'project_name': project.name,
                    'status': project.status.value,
                    'total_photos': total_photos,
                    'accepted_photos': accepted_photos,
                    'rejected_photos': rejected_photos,
                    'expected_photos': project.expected_photos,
                    'delivered_photos': project.delivered_photos,
                    'progress_percentage': round(progress, 1),
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'task_completion_rate': round((completed_tasks / total_tasks * 100) if total_tasks > 0 else 0, 1),
                    'days_until_due': (project.due_date - datetime.utcnow()).days if project.due_date else None
                }
                
        except Exception as e:
            logger.error(f"Failed to get project statistics: {e}")
            return {}


class TaskOperations:
    """Operations for managing project tasks."""
    
    @staticmethod
    def create_task(
        project_name: str,
        task_name: str,
        description: Optional[str] = None,
        task_type: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        assigned_to: Optional[str] = None,
        due_date: Optional[datetime] = None,
        estimated_hours: Optional[float] = None,
        parent_task_id: Optional[int] = None,
        depends_on_task_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Task]:
        """
        Create a new task for a project.
        
        Args:
            project_name: Name of the project
            task_name: Task name
            description: Task description
            task_type: Type of task (cull, edit, review, etc.)
            priority: Task priority
            assigned_to: Person assigned to the task
            due_date: Task due date
            estimated_hours: Estimated hours to complete
            parent_task_id: Parent task ID for subtasks
            depends_on_task_id: Task this depends on
            metadata: Additional metadata
            
        Returns:
            Created Task object or None if failed
        """
        try:
            with get_session() as session:
                project = session.query(Project).filter(Project.name == project_name).first()
                if not project:
                    raise ValueError(f"Project '{project_name}' not found")
                
                task = Task(
                    project_id=project.id,
                    name=task_name,
                    description=description,
                    task_type=task_type,
                    priority=priority,
                    assigned_to=assigned_to,
                    due_date=due_date,
                    estimated_hours=estimated_hours,
                    parent_task_id=parent_task_id,
                    depends_on_task_id=depends_on_task_id,
                    status=TaskStatus.TODO,
                    metadata=metadata or {}
                )
                
                session.add(task)
                session.flush()
                logger.info(f"Created task: {task.id} - {task_name} for project '{project_name}'")
                return task
                
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return None
    
    @staticmethod
    def update_task_status(
        task_id: int,
        status: TaskStatus,
        actual_hours: Optional[float] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Update task status."""
        try:
            with get_session() as session:
                task = session.query(Task).filter(Task.id == task_id).first()
                if not task:
                    return False
                
                task.status = status
                
                # Set timestamps based on status
                if status == TaskStatus.IN_PROGRESS and not task.started_at:
                    task.started_at = datetime.utcnow()
                elif status == TaskStatus.COMPLETED:
                    task.completed_at = datetime.utcnow()
                    if actual_hours is not None:
                        task.actual_hours = actual_hours
                
                if notes:
                    task.notes = notes
                
                logger.info(f"Updated task {task_id} status to {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
            return False
    
    @staticmethod
    def assign_photos_to_task(
        task_id: int,
        photo_ids: List[int]
    ) -> int:
        """
        Assign photos to a task.
        
        Args:
            task_id: Task ID
            photo_ids: List of photo IDs to assign
            
        Returns:
            Number of photos assigned
        """
        try:
            with get_session() as session:
                task = session.query(Task).filter(Task.id == task_id).first()
                if not task:
                    raise ValueError(f"Task {task_id} not found")
                
                count = 0
                for photo_id in photo_ids:
                    photo = session.query(Photo).filter(Photo.id == photo_id).first()
                    if photo and photo not in task.photos:
                        task.photos.append(photo)
                        count += 1
                
                session.commit()
                logger.info(f"Assigned {count} photos to task {task_id}")
                return count
                
        except Exception as e:
            logger.error(f"Failed to assign photos to task: {e}")
            return 0
    
    @staticmethod
    def list_tasks(
        project_name: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        assigned_to: Optional[str] = None,
        priority: Optional[TaskPriority] = None,
        include_completed: bool = False
    ) -> List[Task]:
        """
        List tasks with filters.
        
        Args:
            project_name: Filter by project name
            status: Filter by task status
            assigned_to: Filter by assignee
            priority: Filter by priority
            include_completed: Include completed tasks
            
        Returns:
            List of tasks matching criteria
        """
        with get_session() as session:
            query = session.query(Task)
            
            if project_name:
                project = session.query(Project).filter(Project.name == project_name).first()
                if project:
                    query = query.filter(Task.project_id == project.id)
            
            if status:
                query = query.filter(Task.status == status)
            elif not include_completed:
                query = query.filter(Task.status != TaskStatus.COMPLETED)
            
            if assigned_to:
                query = query.filter(Task.assigned_to == assigned_to)
            
            if priority:
                query = query.filter(Task.priority == priority)
            
            return query.order_by(Task.priority.desc(), Task.due_date).all()
    
    @staticmethod
    def get_task_dependencies(task_id: int) -> Dict[str, Any]:
        """Get task dependencies and blocking relationships."""
        with get_session() as session:
            task = session.query(Task).filter(Task.id == task_id).first()
            if not task:
                return {}
            
            # Tasks this depends on
            depends_on = []
            if task.depends_on:
                depends_on.append({
                    'id': task.depends_on.id,
                    'name': task.depends_on.name,
                    'status': task.depends_on.status.value
                })
            
            # Tasks that depend on this
            dependent_tasks = session.query(Task).filter(
                Task.depends_on_task_id == task_id
            ).all()
            
            blocks = [
                {
                    'id': t.id,
                    'name': t.name,
                    'status': t.status.value
                }
                for t in dependent_tasks
            ]
            
            return {
                'task_id': task_id,
                'depends_on': depends_on,
                'blocks': blocks,
                'can_start': len(depends_on) == 0 or all(
                    d['status'] == TaskStatus.COMPLETED.value for d in depends_on
                )
            }