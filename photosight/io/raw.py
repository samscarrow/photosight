"""
RAW file processing for PhotoSight
Handles reading Sony .ARW files and extracting metadata
"""

import rawpy
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import exiftool
from datetime import datetime
from photosight.utils.file_protection import protected_analysis, FileProtector

logger = logging.getLogger(__name__)


class RawProcessor:
    """Handles RAW file reading and metadata extraction"""
    
    def __init__(self, enable_protection: bool = True):
        """
        Initialize RAW processor
        
        Args:
            enable_protection: Enable source file protection
        """
        self.exiftool = None
        self.enable_protection = enable_protection
        self.protector = FileProtector(verify_checksums=True, read_only_mode=True) if enable_protection else None
        
    def __enter__(self):
        """Context manager entry - start ExifTool process"""
        self.exiftool = exiftool.ExifTool()
        self.exiftool.run()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - terminate ExifTool process"""
        if self.exiftool:
            self.exiftool.terminate()
            
    def load_raw_file(self, file_path: Path) -> Tuple[Optional[rawpy.RawPy], Optional[str]]:
        """
        Load a RAW file using rawpy with optional protection
        
        Args:
            file_path: Path to RAW file
            
        Returns:
            Tuple of (raw object, error message)
        """
        try:
            file_path = Path(file_path).resolve()
            
            # Protect file if enabled
            if self.enable_protection and self.protector:
                with protected_analysis(file_path):
                    raw = rawpy.imread(str(file_path))
                    logger.debug(f"Successfully loaded protected RAW file: {file_path}")
            else:
                raw = rawpy.imread(str(file_path))
                logger.debug(f"Successfully loaded RAW file: {file_path}")
                
            return raw, None
        except Exception as e:
            error_msg = f"Failed to load RAW file {file_path}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
            
    def get_preview_array(self, raw: rawpy.RawPy, 
                         use_camera_wb: bool = True) -> Optional[np.ndarray]:
        """
        Get a preview array from RAW file for analysis
        
        Args:
            raw: RawPy object
            use_camera_wb: Whether to use camera white balance
            
        Returns:
            RGB array suitable for analysis, or None on error
        """
        try:
            # Use faster postprocessing for preview
            rgb = raw.postprocess(
                use_camera_wb=use_camera_wb,
                half_size=True,  # Faster processing
                no_auto_bright=False,
                output_bps=8
            )
            return rgb
        except Exception as e:
            logger.error(f"Failed to generate preview: {str(e)}")
            return None
            
    def extract_metadata(self, file_path: Path) -> Dict:
        """
        Extract comprehensive metadata from RAW file
        
        Args:
            file_path: Path to RAW file
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'error': None
        }
        
        try:
            if not self.exiftool:
                # Fallback to single-use ExifTool if not in context manager
                with exiftool.ExifTool() as et:
                    result = et.execute_json('-j', str(file_path))
                    tags = result[0] if result else {}
            else:
                result = self.exiftool.execute_json('-j', str(file_path))
                tags = result[0] if result else {}
                
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
                'raw_metadata': tags  # Store complete metadata
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
        """
        Parse shutter speed string to numeric value
        
        Args:
            shutter_speed: Shutter speed string (e.g., "1/60", "2.5")
            
        Returns:
            Numeric shutter speed in seconds
        """
        try:
            if '/' in shutter_speed:
                # Handle fraction format (e.g., "1/60")
                parts = shutter_speed.split('/')
                return float(parts[0]) / float(parts[1])
            else:
                # Handle decimal format (e.g., "2.5")
                return float(shutter_speed)
        except:
            logger.warning(f"Could not parse shutter speed: {shutter_speed}")
            return None
            
    def get_histogram(self, image_array: np.ndarray, 
                     bins: int = 256) -> Dict[str, np.ndarray]:
        """
        Calculate histogram for each color channel
        
        Args:
            image_array: RGB image array
            bins: Number of histogram bins
            
        Returns:
            Dictionary with histograms for each channel
        """
        histograms = {}
        
        # Calculate histogram for each channel
        for i, channel in enumerate(['red', 'green', 'blue']):
            hist, _ = np.histogram(image_array[:, :, i], bins=bins, range=(0, 256))
            histograms[channel] = hist
            
        # Calculate luminance histogram
        # Using ITU-R BT.709 luma coefficients
        luminance = (0.2126 * image_array[:, :, 0] + 
                    0.7152 * image_array[:, :, 1] + 
                    0.0722 * image_array[:, :, 2])
        histograms['luminance'], _ = np.histogram(luminance, bins=bins, range=(0, 256))
        
        return histograms
    
    def check_clipping(self, histograms: Dict[str, np.ndarray], 
                      threshold: float = 0.01) -> Dict[str, bool]:
        """
        Check for highlight/shadow clipping in histograms
        
        Args:
            histograms: Dictionary of histograms
            threshold: Percentage threshold for clipping
            
        Returns:
            Dictionary indicating clipping for each channel
        """
        clipping = {}
        
        for channel, hist in histograms.items():
            total_pixels = np.sum(hist)
            
            # Check shadow clipping (first bin)
            shadow_clip = hist[0] / total_pixels > threshold
            
            # Check highlight clipping (last bin)
            highlight_clip = hist[-1] / total_pixels > threshold
            
            clipping[f'{channel}_shadow_clipped'] = shadow_clip
            clipping[f'{channel}_highlight_clipped'] = highlight_clip
            clipping[f'{channel}_clipped'] = shadow_clip or highlight_clip
            
        return clipping