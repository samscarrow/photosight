"""
Modified RAW processor that can handle JPEG files for testing
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import exiftool
from datetime import datetime
from PIL import Image
import rawpy

logger = logging.getLogger(__name__)


class TestableRawProcessor:
    """RAW processor that also handles JPEG for testing"""
    
    def __init__(self, enable_protection: bool = True):
        """
        Initialize processor
        
        Args:
            enable_protection: Enable source file protection (ignored in test processor)
        """
        self.exiftool = None
        self.enable_protection = enable_protection
        
    def __enter__(self):
        """Context manager entry - start ExifTool process"""
        self.exiftool = exiftool.ExifToolHelper()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - terminate ExifTool process"""
        if self.exiftool:
            self.exiftool.terminate()
            
    def load_raw_file(self, file_path: Path) -> Tuple[Optional[object], Optional[str]]:
        """
        Load a RAW or JPEG file
        """
        try:
            if file_path.suffix.lower() in ['.jpg', '.jpeg']:
                # For JPEG, return a mock raw object with the image
                class MockRaw:
                    def __init__(self, image_array):
                        self.image_array = image_array
                    def close(self):
                        pass
                        
                img = Image.open(str(file_path))
                img_array = np.array(img)
                return MockRaw(img_array), None
            else:
                # Original RAW handling
                raw = rawpy.imread(str(file_path))
                logger.debug(f"Successfully loaded RAW file: {file_path}")
                return raw, None
        except Exception as e:
            error_msg = f"Failed to load file {file_path}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
            
    def get_preview_array(self, raw, use_camera_wb: bool = True) -> Optional[np.ndarray]:
        """
        Get a preview array from RAW or JPEG file for analysis
        """
        try:
            if hasattr(raw, 'image_array'):
                # It's our mock JPEG object
                return raw.image_array
            else:
                # Original RAW processing
                rgb = raw.postprocess(
                    use_camera_wb=use_camera_wb,
                    half_size=True,
                    no_auto_bright=False,
                    output_bps=8
                )
                return rgb
        except Exception as e:
            logger.error(f"Failed to generate preview: {str(e)}")
            return None
            
    def extract_metadata(self, file_path: Path) -> Dict:
        """
        Extract comprehensive metadata from file
        """
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'error': None
        }
        
        try:
            if not self.exiftool:
                # Fallback to single-use ExifTool if not in context manager
                with exiftool.ExifToolHelper() as et:
                    tags = et.get_metadata([str(file_path)])[0]
            else:
                tags = self.exiftool.get_metadata([str(file_path)])[0]
                
            # Extract key photographic metadata
            metadata.update({
                # Camera info
                'camera_make': tags.get('EXIF:Make', 'Unknown'),
                'camera_model': tags.get('EXIF:Model', 'Unknown'),
                'lens_model': tags.get('EXIF:LensModel', 'Unknown'),
                
                # Exposure settings
                'iso': tags.get('EXIF:ISO', None),
                'shutter_speed': tags.get('EXIF:ShutterSpeed', None),
                'shutter_speed_value': tags.get('EXIF:ShutterSpeedValue', None),
                'aperture': tags.get('EXIF:FNumber', None),
                'focal_length': tags.get('EXIF:FocalLength', None),
                'focal_length_35mm': tags.get('EXIF:FocalLengthIn35mmFormat', None),
                
                # Other settings
                'exposure_compensation': tags.get('EXIF:ExposureCompensation', None),
                'white_balance': tags.get('EXIF:WhiteBalance', None),
                'focus_mode': tags.get('EXIF:FocusMode', None),
                'af_area_mode': tags.get('EXIF:AFAreaMode', None),
                
                # Timestamps
                'date_taken': tags.get('EXIF:DateTimeOriginal', None),
                'file_modification_date': tags.get('File:FileModificationDate', None),
                
                # Image info
                'image_width': tags.get('EXIF:ImageWidth', None),
                'image_height': tags.get('EXIF:ImageHeight', None),
                'orientation': tags.get('EXIF:Orientation', None),
                
                # Raw metadata
                'raw_metadata': tags
            })
            
            # Parse shutter speed to numeric value if needed
            if metadata['shutter_speed'] and isinstance(metadata['shutter_speed'], str):
                metadata['shutter_speed_numeric'] = self._parse_shutter_speed(metadata['shutter_speed'])
            else:
                metadata['shutter_speed_numeric'] = metadata['shutter_speed_value']
                
        except Exception as e:
            error_msg = f"Failed to extract metadata: {str(e)}"
            logger.error(error_msg)
            metadata['error'] = error_msg
            
        return metadata
    
    def _parse_shutter_speed(self, shutter_speed: str) -> Optional[float]:
        """Parse shutter speed string to numeric value"""
        try:
            if '/' in shutter_speed:
                parts = shutter_speed.split('/')
                return float(parts[0]) / float(parts[1])
            else:
                return float(shutter_speed)
        except:
            logger.warning(f"Could not parse shutter speed: {shutter_speed}")
            return None