"""
Google Drive Photo Server for PhotoSight Web Viewer

Serves photos from Google Drive for the web interface.
"""

import os
import logging
import tempfile
import subprocess
from typing import Optional
from pathlib import Path
import rawpy
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


class GoogleDrivePhotoServer:
    """Serves photos from Google Drive for the web viewer."""
    
    def __init__(self):
        self.cache_dir = Path.home() / ".photosight" / "gdrive_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_photo_from_gdrive_id(self, file_id: str) -> Optional[bytes]:
        """Download photo from Google Drive by file ID."""
        try:
            # Check cache first
            cache_file = self.cache_dir / f"{file_id}.cache"
            if cache_file.exists():
                return cache_file.read_bytes()
            
            # Download from Google Drive
            with tempfile.NamedTemporaryFile() as temp_file:
                result = subprocess.run([
                    "gdrive", "files", "download", file_id,
                    "--path", temp_file.name,
                    "--force"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    photo_bytes = temp_file.read()
                    
                    # Cache for future use
                    cache_file.write_bytes(photo_bytes)
                    
                    return photo_bytes
                
                logger.error(f"Failed to download {file_id}: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading photo {file_id}: {e}")
            return None
    
    def generate_thumbnail_from_gdrive(self, file_id: str, size: tuple = (300, 300)) -> Optional[bytes]:
        """Generate thumbnail from Google Drive ARW file."""
        try:
            # Download the ARW file
            arw_bytes = self.get_photo_from_gdrive_id(file_id)
            if not arw_bytes:
                return None
            
            # Save to temporary file for rawpy processing
            with tempfile.NamedTemporaryFile(suffix='.ARW', delete=False) as temp_file:
                temp_file.write(arw_bytes)
                temp_file.flush()
                
                try:
                    # Process ARW with rawpy
                    with rawpy.imread(temp_file.name) as raw:
                        # Try to extract embedded thumbnail first (fastest)
                        try:
                            thumb = raw.extract_thumb()
                            if thumb.format == rawpy.ThumbFormat.JPEG:
                                img = Image.open(io.BytesIO(thumb.data))
                            else:
                                # Fallback: process RAW to RGB
                                rgb = raw.postprocess(
                                    use_camera_wb=True,
                                    half_size=True,  # Faster processing
                                    no_auto_bright=True
                                )
                                img = Image.fromarray(rgb)
                        except Exception:
                            # Final fallback: full RAW processing
                            rgb = raw.postprocess(
                                use_camera_wb=True,
                                half_size=True,
                                no_auto_bright=True
                            )
                            img = Image.fromarray(rgb)
                    
                    # Convert to RGB if necessary
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    
                    # Generate thumbnail
                    img.thumbnail(size, Image.Resampling.LANCZOS)
                    
                    # Save to bytes
                    output = io.BytesIO()
                    img.save(output, format='JPEG', quality=85, optimize=True)
                    return output.getvalue()
                    
                finally:
                    # Clean up temp file
                    os.unlink(temp_file.name)
                    
        except Exception as e:
            logger.error(f"Error generating thumbnail from {file_id}: {e}")
            return None
    
    def get_file_id_from_database_path(self, db_path: str) -> Optional[str]:
        """Extract Google Drive file ID from database path."""
        # Database paths will be like: "gdrive://1BxG7K8..."
        if db_path.startswith('gdrive://'):
            return db_path[9:]  # Remove "gdrive://" prefix
        return None
    
    def clear_cache(self):
        """Clear the local cache of downloaded photos."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


# Global instance for the web viewer
gdrive_server = GoogleDrivePhotoServer()