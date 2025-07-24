#!/usr/bin/env python3
"""
Camera Profile Migration Script

Populates the camera_profiles table with manufacturer-specific processing profiles
based on the profiles defined in the raw_processor.py.
"""

from sqlalchemy.orm import Session
from photosight.db.models import CameraProfile
from photosight.db.connection import get_db_session
import logging

logger = logging.getLogger(__name__)

def create_default_camera_profiles():
    """Create default camera profiles based on raw_processor.py profiles"""
    
    profiles_data = [
        # Sony Alpha 7 Mark III - Primary default camera
        {
            'camera_make': 'Sony',
            'camera_model': 'ILCE-7M3',  # A7 III
            'high_iso_threshold': 2500,
            'highlight_headroom': 0.25,
            'shadow_lift_bias': 0.1,
            'wb_confidence_boost': 1.2,
            'color_grading_bias': 'cinematic',
            'base_iso': 100,
            'noise_floor': 0.001,
            'confidence_score': 1.0,  # Highest confidence - primary camera
        },
        
        # Other Sony cameras
        {
            'camera_make': 'Sony',
            'camera_model': 'A7R V',
            'high_iso_threshold': 3200,
            'highlight_headroom': 0.3,
            'shadow_lift_bias': 0.1,
            'wb_confidence_boost': 1.2,
            'color_grading_bias': 'cinematic',
            'base_iso': 100,
            'noise_floor': 0.0008,
            'confidence_score': 0.95,
        },
        {
            'camera_make': 'Sony',
            'camera_model': 'A7 IV',
            'high_iso_threshold': 2500,
            'highlight_headroom': 0.2,
            'shadow_lift_bias': 0.08,
            'wb_confidence_boost': 1.15,
            'color_grading_bias': 'cinematic',
            'base_iso': 100,
            'noise_floor': 0.001,
            'confidence_score': 0.90,
        },
        {
            'camera_make': 'Sony',
            'camera_model': 'A6700',
            'high_iso_threshold': 2000,
            'highlight_headroom': 0.15,
            'shadow_lift_bias': 0.05,
            'wb_confidence_boost': 1.1,
            'color_grading_bias': 'bright_airy',
            'base_iso': 100,
            'noise_floor': 0.0012,
            'confidence_score': 0.85,
        },
        
        # Canon cameras
        {
            'camera_make': 'Canon',
            'camera_model': 'EOS R5',
            'high_iso_threshold': 3200,
            'highlight_headroom': 0.25,
            'shadow_lift_bias': 0.12,
            'wb_confidence_boost': 1.25,
            'color_grading_bias': 'film_emulation',
            'base_iso': 100,
            'noise_floor': 0.0009,
            'confidence_score': 0.95,
        },
        {
            'camera_make': 'Canon',
            'camera_model': 'EOS R6 Mark II',
            'high_iso_threshold': 2800,
            'highlight_headroom': 0.2,
            'shadow_lift_bias': 0.1,
            'wb_confidence_boost': 1.2,
            'color_grading_bias': 'film_emulation',
            'base_iso': 100,
            'noise_floor': 0.001,
            'confidence_score': 0.90,
        },
        {
            'camera_make': 'Canon',
            'camera_model': 'EOS 90D',
            'high_iso_threshold': 1600,
            'highlight_headroom': 0.1,
            'shadow_lift_bias': 0.05,
            'wb_confidence_boost': 1.1,
            'color_grading_bias': 'bright_airy',
            'base_iso': 100,
            'noise_floor': 0.0015,
            'confidence_score': 0.80,
        },
        
        # Nikon cameras  
        {
            'camera_make': 'Nikon',
            'camera_model': 'Z9',
            'high_iso_threshold': 3200,
            'highlight_headroom': 0.35,
            'shadow_lift_bias': 0.15,
            'wb_confidence_boost': 1.3,
            'color_grading_bias': 'vintage',
            'base_iso': 64,
            'noise_floor': 0.0007,
            'confidence_score': 0.95,
        },
        {
            'camera_make': 'Nikon',
            'camera_model': 'Z7 II',
            'high_iso_threshold': 2500,
            'highlight_headroom': 0.25,
            'shadow_lift_bias': 0.12,
            'wb_confidence_boost': 1.25,
            'color_grading_bias': 'vintage',
            'base_iso': 64,
            'noise_floor': 0.0009,
            'confidence_score': 0.90,
        },
        {
            'camera_make': 'Nikon',
            'camera_model': 'D850',
            'high_iso_threshold': 2000,
            'highlight_headroom': 0.3,
            'shadow_lift_bias': 0.1,
            'wb_confidence_boost': 1.2,
            'color_grading_bias': 'vintage',
            'base_iso': 64,
            'noise_floor': 0.001,
            'confidence_score': 0.85,
        },
        
        # Fujifilm cameras
        {
            'camera_make': 'Fujifilm',
            'camera_model': 'X-T5',
            'high_iso_threshold': 2500,
            'highlight_headroom': 0.2,
            'shadow_lift_bias': 0.08,
            'wb_confidence_boost': 1.15,
            'color_grading_bias': 'film_emulation',
            'base_iso': 125,
            'noise_floor': 0.001,
            'confidence_score': 0.90,
        },
        
        # Default/fallback profile
        {
            'camera_make': 'Unknown',
            'camera_model': 'Default',
            'high_iso_threshold': 1600,
            'highlight_headroom': 0.0,
            'shadow_lift_bias': 0.0,
            'wb_confidence_boost': 1.0,
            'color_grading_bias': 'none',
            'base_iso': 100,
            'noise_floor': 0.001,
            'confidence_score': 0.5,
        }
    ]
    
    with get_db_session() as session:
        existing_profiles = session.query(CameraProfile).all()
        existing_keys = {(p.camera_make, p.camera_model) for p in existing_profiles}
        
        created_count = 0
        for profile_data in profiles_data:
            key = (profile_data['camera_make'], profile_data['camera_model'])
            
            if key not in existing_keys:
                profile = CameraProfile(**profile_data)
                session.add(profile)
                created_count += 1
                logger.info(f"Created profile for {profile_data['camera_make']} {profile_data['camera_model']}")
        
        if created_count > 0:
            session.commit()
            logger.info(f"Successfully created {created_count} camera profiles")
        else:
            logger.info("No new camera profiles needed - all already exist")

def migrate_existing_photos_to_profiles():
    """Associate existing photos with their camera profiles"""
    
    with get_db_session() as session:
        # Get photos without camera profiles
        from photosight.db.models import Photo
        
        photos_without_profiles = session.query(Photo).filter(
            Photo.camera_profile_id.is_(None),
            Photo.camera_make.isnot(None),
            Photo.camera_model.isnot(None)
        ).all()
        
        updated_count = 0
        for photo in photos_without_profiles:
            # Find matching camera profile
            profile = session.query(CameraProfile).filter(
                CameraProfile.camera_make == photo.camera_make,
                CameraProfile.camera_model == photo.camera_model,
                CameraProfile.is_active == True
            ).first()
            
            if profile:
                photo.camera_profile_id = profile.id
                updated_count += 1
            else:
                # Try fallback to make-only match
                profile = session.query(CameraProfile).filter(
                    CameraProfile.camera_make == photo.camera_make,
                    CameraProfile.camera_model == 'Default',
                    CameraProfile.is_active == True
                ).first()
                
                if not profile:
                    # Use global default
                    profile = session.query(CameraProfile).filter(
                        CameraProfile.camera_make == 'Unknown',
                        CameraProfile.camera_model == 'Default'
                    ).first()
                
                if profile:
                    photo.camera_profile_id = profile.id
                    updated_count += 1
        
        if updated_count > 0:
            session.commit()
            logger.info(f"Successfully associated {updated_count} photos with camera profiles")
        else:
            logger.info("No photos needed camera profile association")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Creating default camera profiles...")
    create_default_camera_profiles()
    
    print("Associating existing photos with profiles...")
    migrate_existing_photos_to_profiles()
    
    print("Camera profile migration complete!")