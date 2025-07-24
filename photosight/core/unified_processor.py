#!/usr/bin/env python3
"""
Unified PhotoSight Processor - Works both locally and on pifive0

This processor bridges the gap between simple local processing and 
full-featured cloud processing with Oracle integration.
"""

import os
import sys
import logging
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import yaml

# Core imports
import rawpy
import cv2
import numpy as np
from dataclasses import dataclass, asdict

# PhotoSight imports
from photosight.processing.yolo_integration import analyze_preview_with_blur
from photosight.processing.raw_processor import RawPostProcessor, ProcessingRecipe

# Optional imports (may not be available locally)
try:
    import oracledb
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False
    
try:
    import exifread
    EXIF_AVAILABLE = True
except ImportError:
    EXIF_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single photo"""
    photo_guid: str
    file_name: str
    status: str  # accepted, rejected
    blur_classification: Optional[str] = None
    people_count: int = 0
    objects_detected: List[str] = None
    output_path: Optional[Path] = None
    rejection_reason: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict = None


class UnifiedPhotoProcessor:
    """
    Unified processor that works in multiple modes:
    - local: Simple file-based processing
    - cloud: Full Oracle integration (requires pifive0)
    - hybrid: Local processing with cloud sync
    """
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "local"):
        """Initialize processor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.mode = mode
        logger.info(f"Initializing PhotoSight processor in {mode} mode")
        
        # Core components (always available)
        self.raw_processor = RawPostProcessor(
            preview_size=800,
            preview_quality=85,
            cache_previews=True
        )
        
        # Optional components
        self.oracle_conn = None
        if mode in ['cloud', 'hybrid'] and ORACLE_AVAILABLE:
            self._init_oracle()
            
    def _init_oracle(self):
        """Initialize Oracle connection if available"""
        try:
            # Try REST API first (works from anywhere)
            oracle_config = self.config.get('oracle', {})
            if oracle_config.get('use_rest_api'):
                self.oracle_conn = OracleRESTClient(oracle_config)
            else:
                # Direct connection (requires proper network access)
                self.oracle_conn = oracledb.connect(
                    user=oracle_config['user'],
                    password=oracle_config['password'],
                    dsn=oracle_config['dsn']
                )
            logger.info("Oracle connection established")
        except Exception as e:
            logger.warning(f"Oracle connection failed: {e}")
            if self.mode == 'cloud':
                raise
            # In hybrid mode, continue without Oracle
            
    def process_photo(self, raw_path: Path, photoshoot_tag: Optional[str] = None,
                     output_dir: Optional[Path] = None) -> ProcessingResult:
        """Process a single photo with unified pipeline"""
        start_time = datetime.now()
        
        # Generate GUID for this photo
        photo_guid = str(uuid.uuid4())
        logger.info(f"Processing {raw_path.name} with GUID {photo_guid}")
        
        try:
            # 1. Generate preview for analysis
            with rawpy.imread(str(raw_path)) as raw:
                preview = raw.postprocess(
                    use_camera_wb=True,
                    half_size=True,
                    output_bps=8
                )
                
            # Resize for YOLO
            h, w = preview.shape[:2]
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                preview = cv2.resize(preview, (int(w*scale), int(h*scale)))
                
            # 2. Run YOLO analysis
            yolo_result, blur_result = analyze_preview_with_blur(preview, self.config)
            
            # 3. Make accept/reject decision
            if not blur_result.meets_quality_threshold:
                return ProcessingResult(
                    photo_guid=photo_guid,
                    file_name=raw_path.name,
                    status='rejected',
                    blur_classification=blur_result.overall_classification,
                    people_count=yolo_result.person_count,
                    objects_detected=[d.class_name for d in yolo_result.detections],
                    rejection_reason=blur_result.overall_classification,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
                
            # 4. Process accepted photo
            if output_dir:
                output_path = output_dir / f"{raw_path.stem}.jpg"
            else:
                output_path = raw_path.parent / f"{raw_path.stem}_processed.jpg"
                
            # Create processing recipe
            recipe = self.raw_processor.create_default_recipe(raw_path)
            recipe.shadows = 35
            recipe.highlights = -20
            recipe.vibrance = 25
            recipe.saturation_factor = 1.05
            
            # Process full resolution
            self.raw_processor.process_raw_file(
                raw_path,
                output_path,
                recipe=recipe
            )
            
            # 5. Extract metadata
            metadata = {
                'photo_guid': photo_guid,
                'photoshoot_tag': photoshoot_tag,
                'capture_date': datetime.fromtimestamp(raw_path.stat().st_mtime).isoformat(),
                'file_size': raw_path.stat().st_size,
                'people_count': yolo_result.person_count,
                'blur_classification': blur_result.overall_classification,
                'objects': [d.class_name for d in yolo_result.detections]
            }
            
            # Extract EXIF if available
            if EXIF_AVAILABLE:
                metadata.update(self._extract_exif(raw_path))
                
            # 6. Store data based on mode
            if self.mode == 'local' or not self.oracle_conn:
                # Save metadata as JSON
                meta_path = output_path.with_suffix('.json')
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            else:
                # Store in Oracle
                self._store_in_oracle(metadata, yolo_result)
                
            return ProcessingResult(
                photo_guid=photo_guid,
                file_name=raw_path.name,
                status='accepted',
                blur_classification=blur_result.overall_classification,
                people_count=yolo_result.person_count,
                objects_detected=[d.class_name for d in yolo_result.detections],
                output_path=output_path,
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing {raw_path.name}: {e}")
            return ProcessingResult(
                photo_guid=photo_guid,
                file_name=raw_path.name,
                status='error',
                rejection_reason=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
    def _extract_exif(self, raw_path: Path) -> Dict:
        """Extract EXIF data from RAW file"""
        exif_data = {}
        try:
            with open(raw_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
            # Extract key fields
            exif_data['camera_make'] = str(tags.get('EXIF Make', ''))
            exif_data['camera_model'] = str(tags.get('EXIF Model', ''))
            exif_data['lens_model'] = str(tags.get('EXIF LensModel', ''))
            exif_data['iso_speed'] = str(tags.get('EXIF ISOSpeedRatings', ''))
            exif_data['aperture'] = str(tags.get('EXIF FNumber', ''))
            exif_data['shutter_speed'] = str(tags.get('EXIF ExposureTime', ''))
            exif_data['focal_length'] = str(tags.get('EXIF FocalLength', ''))
            
            # GPS if available
            if 'GPS GPSLatitude' in tags:
                exif_data['gps_latitude'] = str(tags.get('GPS GPSLatitude', ''))
                exif_data['gps_longitude'] = str(tags.get('GPS GPSLongitude', ''))
                
        except Exception as e:
            logger.warning(f"Could not extract EXIF: {e}")
            
        return exif_data
        
    def _store_in_oracle(self, metadata: Dict, yolo_result) -> None:
        """Store photo and detection data in Oracle"""
        if not self.oracle_conn:
            return
            
        try:
            cursor = self.oracle_conn.cursor()
            
            # Insert photo record
            cursor.execute("""
                INSERT INTO photos (
                    photo_guid, file_name, capture_date, 
                    camera_make, camera_model, photoshoot_tag
                ) VALUES (
                    HEXTORAW(:guid), :file_name, :capture_date,
                    :camera_make, :camera_model, :photoshoot_tag
                )
            """, {
                'guid': metadata['photo_guid'].replace('-', ''),
                'file_name': metadata['file_name'],
                'capture_date': metadata['capture_date'],
                'camera_make': metadata.get('camera_make', 'Unknown'),
                'camera_model': metadata.get('camera_model', 'Unknown'),
                'photoshoot_tag': metadata.get('photoshoot_tag')
            })
            
            # Insert YOLO detections
            for detection in yolo_result.detections:
                detection_id = str(uuid.uuid4()).replace('-', '')
                cursor.execute("""
                    INSERT INTO yolo_detections (
                        detection_id, photo_guid, class_name, confidence,
                        bbox_x, bbox_y, bbox_width, bbox_height
                    ) VALUES (
                        HEXTORAW(:detection_id), HEXTORAW(:photo_guid),
                        :class_name, :confidence,
                        :x, :y, :width, :height
                    )
                """, {
                    'detection_id': detection_id,
                    'photo_guid': metadata['photo_guid'].replace('-', ''),
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'x': detection.bbox[0],
                    'y': detection.bbox[1],
                    'width': detection.bbox[2],
                    'height': detection.bbox[3]
                })
                
            self.oracle_conn.commit()
            logger.info(f"Stored {metadata['file_name']} in Oracle")
            
        except Exception as e:
            logger.error(f"Oracle storage failed: {e}")
            if self.mode == 'cloud':
                raise
                
    def process_batch(self, raw_files: List[Path], photoshoot_tag: Optional[str] = None,
                     output_dir: Optional[Path] = None) -> Dict:
        """Process multiple photos"""
        results = []
        stats = {
            'total': len(raw_files),
            'accepted': 0,
            'rejected': 0,
            'errors': 0
        }
        
        for raw_file in raw_files:
            result = self.process_photo(raw_file, photoshoot_tag, output_dir)
            results.append(result)
            
            if result.status == 'accepted':
                stats['accepted'] += 1
            elif result.status == 'rejected':
                stats['rejected'] += 1
            else:
                stats['errors'] += 1
                
            # Log progress
            processed = stats['accepted'] + stats['rejected'] + stats['errors']
            if processed % 10 == 0:
                logger.info(f"Progress: {processed}/{stats['total']}")
                
        return {
            'results': results,
            'stats': stats,
            'acceptance_rate': stats['accepted'] / stats['total'] * 100
        }


class OracleRESTClient:
    """REST API client for Oracle Cloud (works from anywhere)"""
    
    def __init__(self, config: Dict):
        self.base_url = config['rest_api_url']
        self.auth = (config['user'], config['password'])
        
    def cursor(self):
        return RESTCursor(self.base_url, self.auth)
        
    def commit(self):
        pass  # REST API auto-commits


class RESTCursor:
    """Minimal cursor interface for REST API"""
    
    def __init__(self, base_url: str, auth: Tuple[str, str]):
        self.base_url = base_url
        self.auth = auth
        
    def execute(self, sql: str, params: Dict = None):
        # Implementation would POST SQL to REST endpoint
        pass


if __name__ == "__main__":
    # Example usage
    processor = UnifiedPhotoProcessor(mode="local")
    
    # Process workshop photos
    workshop_files = list(Path("/Volumes/Untitled/DCIM/100MSDCF").glob("*.ARW"))[:5]
    results = processor.process_batch(
        workshop_files,
        photoshoot_tag="test_unified",
        output_dir=Path("./test_output")
    )
    
    print(f"Processed {results['stats']['total']} photos")
    print(f"Acceptance rate: {results['acceptance_rate']:.1f}%")