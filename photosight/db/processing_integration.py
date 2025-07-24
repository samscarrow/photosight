#!/usr/bin/env python3
"""
Database integration utilities for the processing pipeline.

Provides functions to store and retrieve processing results, camera profiles,
and EXIF data in the PostgreSQL database.
"""

from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func
import json
import logging
from dataclasses import asdict

from photosight.db.models import (
    Photo, AnalysisResult, ProcessingRecipe, CameraProfile, 
    ProjectPhoto, Task, Project
)
from photosight.db.connection import get_db_session
from photosight.processing.raw_processor import EXIFData, ToneMapperSettings

logger = logging.getLogger(__name__)


class ProcessingDatabaseManager:
    """Manages database operations for the processing pipeline"""
    
    def __init__(self):
        self.session = None
    
    def store_photo_with_exif(self, file_path: str, exif_data: EXIFData, 
                             file_checksum: str) -> Optional[int]:
        """
        Store photo with EXIF data in database
        
        Args:
            file_path: Path to the photo file
            exif_data: Extracted EXIF data
            file_checksum: File checksum for deduplication
            
        Returns:
            Photo ID if successful, None otherwise
        """
        try:
            with get_db_session() as session:
                # Check if photo already exists
                existing_photo = session.query(Photo).filter(
                    Photo.checksum == file_checksum
                ).first()
                
                if existing_photo:
                    logger.info(f"Photo already exists: {existing_photo.id}")
                    return existing_photo.id
                
                # Create new photo record
                photo = Photo(
                    file_path=file_path,
                    filename=file_path.split('/')[-1],
                    checksum=file_checksum,
                    
                    # Image dimensions
                    image_width=exif_data.image_width,
                    image_height=exif_data.image_height,
                    
                    # Timestamps
                    date_taken=exif_data.datetime_original_parsed,
                    
                    # Camera info (promoted fields)
                    camera_make=exif_data.camera_make,
                    camera_model=exif_data.camera_model,
                    lens_model=exif_data.lens_model,
                    camera_serial=exif_data.camera_serial,
                    lens_serial=exif_data.lens_serial,
                    
                    # Exposure settings (promoted fields)
                    iso=exif_data.iso,
                    aperture=exif_data.aperture,
                    shutter_speed_display=exif_data.shutter_speed,
                    focal_length=exif_data.focal_length,
                    focal_length_35mm=exif_data.focal_length_35mm,
                    
                    # Additional metadata
                    exposure_compensation=exif_data.exposure_compensation,
                    flash_fired=exif_data.flash_fired,
                    white_balance=exif_data.white_balance,
                    color_temp_kelvin=exif_data.color_temp_kelvin,
                    metering_mode=exif_data.metering_mode,
                    
                    # GPS
                    gps_latitude=exif_data.gps_latitude,
                    gps_longitude=exif_data.gps_longitude,
                    gps_altitude=exif_data.gps_altitude,
                    
                    # Full EXIF data as JSONB
                    exif_data=asdict(exif_data)
                )
                
                session.add(photo)
                session.flush()  # Get the ID
                
                # Associate with camera profile if possible
                profile = self.find_camera_profile(session, exif_data)
                if profile:
                    photo.camera_profile_id = profile.id
                    logger.info(f"Associated photo with camera profile: {profile.camera_make} {profile.camera_model}")
                
                session.commit()
                logger.info(f"Stored photo in database: {photo.id}")
                return photo.id
                
        except Exception as e:
            logger.error(f"Failed to store photo with EXIF: {e}")
            return None
    
    def find_camera_profile(self, session: Session, exif_data: EXIFData) -> Optional[CameraProfile]:
        """
        Find the best matching camera profile for given EXIF data
        
        Args:
            session: Database session
            exif_data: EXIF data containing camera info
            
        Returns:
            Best matching CameraProfile or None
        """
        if not exif_data.camera_make or not exif_data.camera_model:
            return None
        
        # Try exact model match first
        profile = session.query(CameraProfile).filter(
            CameraProfile.camera_make == exif_data.camera_make,
            CameraProfile.camera_model == exif_data.camera_model,
            CameraProfile.is_active == True
        ).order_by(CameraProfile.confidence_score.desc()).first()
        
        if profile:
            return profile
        
        # Try make-only match with fallback model names
        fallback_models = ['Default', 'Generic']
        for model_name in fallback_models:
            profile = session.query(CameraProfile).filter(
                CameraProfile.camera_make == exif_data.camera_make,
                CameraProfile.camera_model == model_name,
                CameraProfile.is_active == True
            ).first()
            if profile:
                return profile
        
        # Fallback to Sony A7 III (primary camera) if no match found
        fallback = session.query(CameraProfile).filter(
            CameraProfile.camera_make == 'Sony',
            CameraProfile.camera_model == 'ILCE-7M3'
        ).first()
        
        if fallback:
            return fallback
        
        # Final fallback to global default
        return session.query(CameraProfile).filter(
            CameraProfile.camera_make == 'Unknown',
            CameraProfile.camera_model == 'Default'
        ).first()
    
    def store_analysis_results(self, photo_id: int, analysis_data: Dict[str, Any],
                             camera_profile_used: Optional[CameraProfile] = None) -> Optional[int]:
        """
        Store comprehensive analysis results
        
        Args:
            photo_id: ID of the photo
            analysis_data: Complete analysis results dictionary
            camera_profile_used: Camera profile that was used
            
        Returns:
            AnalysisResult ID if successful
        """
        try:
            with get_db_session() as session:
                # Extract white balance analysis if available
                wb_analysis = analysis_data.get('wb_analysis', {})
                
                # Extract exposure analysis if available
                exposure_analysis = analysis_data.get('exposure_analysis', {})
                clipping_analysis = exposure_analysis.get('clipping_analysis', {})
                
                result = AnalysisResult(
                    photo_id=photo_id,
                    analysis_type='comprehensive',
                    
                    # Technical scores from existing analysis
                    sharpness_score=analysis_data.get('sharpness_score'),
                    exposure_quality=analysis_data.get('exposure_quality'),
                    contrast_score=analysis_data.get('contrast_score'),
                    noise_level=analysis_data.get('noise_level'),
                    
                    # White balance analysis results
                    wb_estimated_temp=wb_analysis.get('estimated_temp_kelvin'),
                    wb_confidence=wb_analysis.get('confidence'),
                    wb_method_used=wb_analysis.get('method_used'),
                    color_cast_strength=wb_analysis.get('color_cast', {}).get('strength'),
                    dominant_cast=wb_analysis.get('color_cast', {}).get('dominant'),
                    
                    # Tone mapping analysis
                    dynamic_range_score=exposure_analysis.get('dynamic_range_score'),
                    zone_distribution=exposure_analysis.get('zone_distribution'),
                    highlight_clipping=clipping_analysis.get('highlight_clipping'),
                    shadow_clipping=clipping_analysis.get('shadow_clipping'),
                    
                    # Camera profile info
                    camera_profile_used_id=camera_profile_used.id if camera_profile_used else None,
                    profile_confidence=analysis_data.get('profile_confidence', 0.8),
                    
                    # Performance metrics
                    processing_time_ms=analysis_data.get('processing_time_ms'),
                    algorithm_versions=analysis_data.get('algorithm_versions', {}),
                    
                    # Complete analysis data
                    analysis_data=analysis_data
                )
                
                session.add(result)
                session.commit()
                logger.info(f"Stored analysis results for photo {photo_id}")
                return result.id
                
        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")
            return None
    
    def store_processing_recipe(self, recipe_data: Dict[str, Any], 
                              tone_settings: Optional[ToneMapperSettings] = None) -> Optional[int]:
        """
        Store a processing recipe with tone mapping settings
        
        Args:
            recipe_data: Recipe parameters
            tone_settings: Tone mapping settings
            
        Returns:
            ProcessingRecipe ID if successful
        """
        try:
            with get_db_session() as session:
                recipe = ProcessingRecipe(
                    name=recipe_data.get('name', f"Recipe_{int(func.now().extract('epoch'))}"),
                    description=recipe_data.get('description', ''),
                    category=recipe_data.get('category', 'auto_generated'),
                    parameters=recipe_data,
                    tone_curve_type=tone_settings.tone_curve_type if tone_settings else 'parametric',
                    tone_settings=asdict(tone_settings) if tone_settings else None,
                )
                
                session.add(recipe)
                session.commit()
                logger.info(f"Stored processing recipe: {recipe.id}")
                return recipe.id
                
        except Exception as e:
            logger.error(f"Failed to store processing recipe: {e}")
            return None
    
    def get_photos_for_batch_processing(self, project_id: Optional[int] = None,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get photos ready for batch processing
        
        Args:
            project_id: Optional project filter
            limit: Maximum number of photos to return
            
        Returns:
            List of photo data with EXIF and camera profile info
        """
        try:
            with get_db_session() as session:
                query = session.query(Photo).filter(
                    Photo.processing_status.in_(['pending', 'error'])
                )
                
                if project_id:
                    query = query.join(ProjectPhoto).filter(
                        ProjectPhoto.project_id == project_id
                    )
                
                photos = query.limit(limit).all()
                
                results = []
                for photo in photos:
                    results.append({
                        'id': photo.id,
                        'file_path': photo.file_path,
                        'exif_data': photo.exif_data,
                        'camera_profile': {
                            'id': photo.camera_profile.id,
                            'make': photo.camera_profile.camera_make,
                            'model': photo.camera_profile.camera_model,
                            'settings': {
                                'high_iso_threshold': photo.camera_profile.high_iso_threshold,
                                'highlight_headroom': photo.camera_profile.highlight_headroom,
                                'shadow_lift_bias': photo.camera_profile.shadow_lift_bias,
                                'wb_confidence_boost': photo.camera_profile.wb_confidence_boost,
                                'color_grading_bias': photo.camera_profile.color_grading_bias
                            }
                        } if photo.camera_profile else None
                    })
                
                logger.info(f"Retrieved {len(results)} photos for batch processing")
                return results
                
        except Exception as e:
            logger.error(f"Failed to get photos for batch processing: {e}")
            return []
    
    def update_photo_processing_status(self, photo_id: int, status: str, 
                                     reason: Optional[str] = None):
        """Update photo processing status"""
        try:
            with get_db_session() as session:
                photo = session.query(Photo).filter(Photo.id == photo_id).first()
                if photo:
                    photo.processing_status = status
                    if reason:
                        photo.rejection_reason = reason
                    session.commit()
                    logger.info(f"Updated photo {photo_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update photo status: {e}")
    
    def get_camera_profile_stats(self) -> Dict[str, Any]:
        """Get statistics about camera profile usage"""
        try:
            with get_db_session() as session:
                # Profile usage stats
                profile_stats = session.query(
                    CameraProfile.camera_make,
                    CameraProfile.camera_model,
                    func.count(Photo.id).label('photo_count')
                ).join(
                    Photo, CameraProfile.id == Photo.camera_profile_id
                ).group_by(
                    CameraProfile.camera_make, 
                    CameraProfile.camera_model
                ).all()
                
                return {
                    'profile_usage': [
                        {
                            'make': stat.camera_make,
                            'model': stat.camera_model,
                            'photo_count': stat.photo_count
                        }
                        for stat in profile_stats
                    ],
                    'total_profiles': session.query(CameraProfile).count(),
                    'active_profiles': session.query(CameraProfile).filter(
                        CameraProfile.is_active == True
                    ).count()
                }
        except Exception as e:
            logger.error(f"Failed to get camera profile stats: {e}")
            return {}