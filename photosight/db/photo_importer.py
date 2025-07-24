"""
Photo metadata importer for PhotoSight database

Imports photo metadata, generates thumbnails, and stores in PostgreSQL
without storing the actual RAW files.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

import exifread
import rawpy
import cv2
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from ..db.models import Photo, Project, Task, ProjectPhoto, ProcessingRecipe
from ..db.database import get_db_session
from ..config import load_config

logger = logging.getLogger(__name__)


class PhotoMetadataImporter:
    """Import photo metadata and thumbnails into database"""
    
    def __init__(self, 
                 thumbnail_size: Tuple[int, int] = (400, 400),
                 preview_size: Tuple[int, int] = (1200, 1200),
                 skip_existing: bool = True):
        """
        Initialize photo metadata importer
        
        Args:
            thumbnail_size: Size for thumbnail generation
            preview_size: Size for preview generation
            skip_existing: Skip photos already in database
        """
        self.thumbnail_size = thumbnail_size
        self.preview_size = preview_size
        self.skip_existing = skip_existing
        self.config = load_config()
        
        # Supported file extensions
        self.raw_extensions = {'.arw', '.cr2', '.cr3', '.nef', '.orf', '.dng', '.raf'}
        self.jpeg_extensions = {'.jpg', '.jpeg', '.png', '.tiff'}
        self.supported_extensions = self.raw_extensions | self.jpeg_extensions
        
    def import_directory(self, directory: Path, 
                        project_name: Optional[str] = None,
                        recursive: bool = True,
                        batch_size: int = 50) -> Dict[str, Any]:
        """
        Import all photos from a directory
        
        Args:
            directory: Directory to import from
            project_name: Optional project to associate photos with
            recursive: Recurse into subdirectories
            batch_size: Number of photos to process before committing
            
        Returns:
            Import statistics
        """
        logger.info(f"Starting import from {directory}")
        
        # Find all image files
        image_files = self._find_image_files(directory, recursive)
        logger.info(f"Found {len(image_files)} image files")
        
        # Get or create project
        project = None
        if project_name:
            with get_db_session() as session:
                project = self._get_or_create_project(session, project_name)
                project_id = project.id
        else:
            project_id = None
        
        # Import statistics
        stats = {
            'total_files': len(image_files),
            'imported': 0,
            'skipped': 0,
            'errors': 0,
            'thumbnails_generated': 0,
            'processing_time': 0
        }
        
        # Process in batches
        start_time = datetime.now()
        
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            batch_stats = self._process_batch(batch, project_id)
            
            # Update statistics
            stats['imported'] += batch_stats['imported']
            stats['skipped'] += batch_stats['skipped']
            stats['errors'] += batch_stats['errors']
            stats['thumbnails_generated'] += batch_stats['thumbnails_generated']
            
            logger.info(f"Progress: {i + len(batch)}/{len(image_files)} files processed")
        
        stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Import complete: {stats['imported']} imported, "
                   f"{stats['skipped']} skipped, {stats['errors']} errors")
        
        return stats
    
    def import_single_photo(self, file_path: Path, 
                          project_id: Optional[int] = None) -> Optional[Photo]:
        """
        Import a single photo
        
        Args:
            file_path: Path to photo file
            project_id: Optional project ID
            
        Returns:
            Created Photo object or None if failed
        """
        try:
            with get_db_session() as session:
                # Check if already exists
                if self.skip_existing:
                    existing = session.query(Photo).filter_by(
                        file_path=str(file_path)
                    ).first()
                    if existing:
                        logger.debug(f"Skipping existing photo: {file_path}")
                        return existing
                
                # Extract metadata
                metadata = self._extract_metadata(file_path)
                if not metadata:
                    logger.error(f"Failed to extract metadata from {file_path}")
                    return None
                
                # Generate thumbnail
                thumbnail_path = self._generate_thumbnail(file_path)
                
                # Create photo record
                photo = Photo(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    file_hash=metadata['file_hash'],
                    file_size=metadata['file_size'],
                    file_type=file_path.suffix.lower(),
                    
                    # Camera info
                    camera_make=metadata.get('camera_make'),
                    camera_model=metadata.get('camera_model'),
                    lens_model=metadata.get('lens_model'),
                    
                    # Shooting settings
                    iso=metadata.get('iso'),
                    aperture=metadata.get('aperture'),
                    shutter_speed=metadata.get('shutter_speed'),
                    focal_length=metadata.get('focal_length'),
                    
                    # Timestamps
                    capture_date=metadata.get('capture_date'),
                    import_date=datetime.now(),
                    
                    # Dimensions
                    width=metadata.get('width'),
                    height=metadata.get('height'),
                    
                    # GPS
                    gps_latitude=metadata.get('gps_latitude'),
                    gps_longitude=metadata.get('gps_longitude'),
                    gps_altitude=metadata.get('gps_altitude'),
                    
                    # Paths
                    thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
                    
                    # Metadata
                    meta_data=metadata.get('extended_metadata', {})
                )
                
                session.add(photo)
                
                # Associate with project if specified
                if project_id:
                    project_photo = ProjectPhoto(
                        project_id=project_id,
                        photo_id=photo.id
                    )
                    session.add(project_photo)
                
                session.commit()
                logger.info(f"Imported photo: {file_path.name}")
                
                return photo
                
        except Exception as e:
            logger.error(f"Error importing {file_path}: {e}")
            return None
    
    def _find_image_files(self, directory: Path, recursive: bool) -> List[Path]:
        """Find all supported image files in directory"""
        image_files = []
        
        if recursive:
            for ext in self.supported_extensions:
                image_files.extend(directory.rglob(f"*{ext}"))
                image_files.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            for ext in self.supported_extensions:
                image_files.extend(directory.glob(f"*{ext}"))
                image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        return image_files
    
    def _process_batch(self, files: List[Path], 
                      project_id: Optional[int]) -> Dict[str, int]:
        """Process a batch of files"""
        stats = {
            'imported': 0,
            'skipped': 0,
            'errors': 0,
            'thumbnails_generated': 0
        }
        
        with get_db_session() as session:
            for file_path in files:
                try:
                    # Check if already exists
                    if self.skip_existing:
                        existing = session.query(Photo).filter_by(
                            file_path=str(file_path)
                        ).first()
                        if existing:
                            stats['skipped'] += 1
                            continue
                    
                    # Extract metadata
                    metadata = self._extract_metadata(file_path)
                    if not metadata:
                        stats['errors'] += 1
                        continue
                    
                    # Generate thumbnail
                    thumbnail_path = self._generate_thumbnail(file_path)
                    if thumbnail_path:
                        stats['thumbnails_generated'] += 1
                    
                    # Create photo record
                    photo = Photo(
                        file_path=str(file_path),
                        file_name=file_path.name,
                        file_hash=metadata['file_hash'],
                        file_size=metadata['file_size'],
                        file_type=file_path.suffix.lower(),
                        
                        camera_make=metadata.get('camera_make'),
                        camera_model=metadata.get('camera_model'),
                        lens_model=metadata.get('lens_model'),
                        
                        iso=metadata.get('iso'),
                        aperture=metadata.get('aperture'),
                        shutter_speed=metadata.get('shutter_speed'),
                        focal_length=metadata.get('focal_length'),
                        
                        capture_date=metadata.get('capture_date'),
                        import_date=datetime.now(),
                        
                        width=metadata.get('width'),
                        height=metadata.get('height'),
                        
                        gps_latitude=metadata.get('gps_latitude'),
                        gps_longitude=metadata.get('gps_longitude'),
                        gps_altitude=metadata.get('gps_altitude'),
                        
                        thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
                        
                        meta_data=metadata.get('extended_metadata', {})
                    )
                    
                    session.add(photo)
                    
                    # Associate with project
                    if project_id:
                        project_photo = ProjectPhoto(
                            project_id=project_id,
                            photo_id=photo.id
                        )
                        session.add(project_photo)
                    
                    stats['imported'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    stats['errors'] += 1
            
            # Commit batch
            try:
                session.commit()
            except Exception as e:
                logger.error(f"Error committing batch: {e}")
                session.rollback()
                stats['errors'] += len(files)
                stats['imported'] = 0
        
        return stats
    
    def _extract_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from image file"""
        try:
            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_hash': self._calculate_file_hash(file_path),
                'extended_metadata': {}
            }
            
            # Read EXIF data
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            
            # Extract camera info
            metadata['camera_make'] = self._get_exif_value(tags, 'Image Make')
            metadata['camera_model'] = self._get_exif_value(tags, 'Image Model')
            metadata['lens_model'] = self._get_exif_value(tags, 'EXIF LensModel')
            
            # Extract shooting settings
            metadata['iso'] = self._get_exif_number(tags, 'EXIF ISOSpeedRatings')
            metadata['aperture'] = self._get_exif_ratio(tags, 'EXIF FNumber')
            metadata['shutter_speed'] = self._get_exif_value(tags, 'EXIF ExposureTime')
            metadata['focal_length'] = self._get_exif_ratio(tags, 'EXIF FocalLength')
            
            # Extract timestamps
            date_str = self._get_exif_value(tags, 'EXIF DateTimeOriginal')
            if date_str:
                try:
                    metadata['capture_date'] = datetime.strptime(
                        date_str, '%Y:%m:%d %H:%M:%S'
                    )
                except:
                    metadata['capture_date'] = None
            
            # Extract GPS if available
            metadata['gps_latitude'] = self._get_gps_coordinate(
                tags, 'GPS GPSLatitude', 'GPS GPSLatitudeRef'
            )
            metadata['gps_longitude'] = self._get_gps_coordinate(
                tags, 'GPS GPSLongitude', 'GPS GPSLongitudeRef'
            )
            metadata['gps_altitude'] = self._get_exif_ratio(tags, 'GPS GPSAltitude')
            
            # Get image dimensions
            if file_path.suffix.lower() in self.raw_extensions:
                # For RAW files, use rawpy
                with rawpy.imread(str(file_path)) as raw:
                    metadata['width'] = raw.sizes.width
                    metadata['height'] = raw.sizes.height
            else:
                # For regular images
                img = Image.open(file_path)
                metadata['width'] = img.width
                metadata['height'] = img.height
                img.close()
            
            # Store additional EXIF data
            for key, value in tags.items():
                if key not in ['JPEGThumbnail', 'TIFFThumbnail']:
                    metadata['extended_metadata'][key] = str(value)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return None
    
    def _generate_thumbnail(self, file_path: Path) -> Optional[Path]:
        """Generate thumbnail for image"""
        try:
            # Create thumbnails directory
            thumbnails_dir = Path(self.config.get('thumbnails_dir', 'thumbnails'))
            thumbnails_dir.mkdir(exist_ok=True)
            
            # Generate thumbnail filename
            file_hash = self._calculate_file_hash(file_path)[:16]
            thumbnail_name = f"{file_hash}_thumb.jpg"
            thumbnail_path = thumbnails_dir / thumbnail_name
            
            # Skip if already exists
            if thumbnail_path.exists():
                return thumbnail_path
            
            # Load and resize image
            if file_path.suffix.lower() in self.raw_extensions:
                # For RAW files
                with rawpy.imread(str(file_path)) as raw:
                    # Use thumbnail if available
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        with open(thumbnail_path, 'wb') as f:
                            f.write(thumb.data)
                    else:
                        # Generate from RAW
                        rgb = raw.postprocess(
                            use_camera_wb=True,
                            half_size=True,
                            no_auto_bright=False
                        )
                        img = Image.fromarray(rgb)
                        img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                        img.save(thumbnail_path, 'JPEG', quality=85)
            else:
                # For regular images
                img = Image.open(file_path)
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85)
                img.close()
            
            logger.debug(f"Generated thumbnail: {thumbnail_path}")
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Error generating thumbnail for {file_path}: {e}")
            return None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_or_create_project(self, session: Session, project_name: str) -> Project:
        """Get existing project or create new one"""
        project = session.query(Project).filter_by(name=project_name).first()
        
        if not project:
            project = Project(
                name=project_name,
                description=f"Imported photos for {project_name}",
                status="Active",
                created_at=datetime.now()
            )
            session.add(project)
            session.commit()
            logger.info(f"Created new project: {project_name}")
        
        return project
    
    def _get_exif_value(self, tags: Dict, key: str) -> Optional[str]:
        """Get EXIF value as string"""
        if key in tags:
            return str(tags[key]).strip()
        return None
    
    def _get_exif_number(self, tags: Dict, key: str) -> Optional[int]:
        """Get EXIF value as number"""
        value = self._get_exif_value(tags, key)
        if value:
            try:
                # Handle arrays like [400]
                if value.startswith('[') and value.endswith(']'):
                    value = value[1:-1]
                return int(value)
            except:
                pass
        return None
    
    def _get_exif_ratio(self, tags: Dict, key: str) -> Optional[float]:
        """Get EXIF value as ratio"""
        if key in tags:
            value = tags[key]
            try:
                if hasattr(value, 'num') and hasattr(value, 'den'):
                    return float(value.num) / float(value.den)
                else:
                    return float(str(value))
            except:
                pass
        return None
    
    def _get_gps_coordinate(self, tags: Dict, coord_key: str, ref_key: str) -> Optional[float]:
        """Extract GPS coordinate from EXIF"""
        if coord_key in tags and ref_key in tags:
            coord = tags[coord_key].values
            ref = str(tags[ref_key])
            
            # Convert to decimal degrees
            degrees = float(coord[0].num) / float(coord[0].den)
            minutes = float(coord[1].num) / float(coord[1].den)
            seconds = float(coord[2].num) / float(coord[2].den)
            
            decimal = degrees + minutes / 60.0 + seconds / 3600.0
            
            # Apply reference (N/S for latitude, E/W for longitude)
            if ref in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        
        return None