#!/usr/bin/env python3
"""
Production YOLO-Enhanced Blur Analysis Processor
Deploys the subject-aware blur analysis with Oracle integration
"""

import os
import sys
import time
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import numpy as np
import cv2

# Add the photosight directory to Python path
sys.path.insert(0, '/home/sam/photosight')

try:
    from photosight.processing.yolo_integration import (
        YOLOBlurProcessor, 
        analyze_preview_with_blur,
        SUBJECT_BLUR_PRIORITIES,
        BLUR_THRESHOLDS
    )
    YOLO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"YOLO integration not available: {e}")
    YOLO_AVAILABLE = False

# Configuration
CONFIG_FILE = "/home/sam/photosight/config.yaml"
STATE_FILE = "/home/sam/photosight/yolo_blur_processor_state.json"
LOG_FILE = "/home/sam/photosight/yolo_blur_processor.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionYOLOBlurProcessor:
    """Production YOLO-enhanced blur analysis processor"""
    
    def __init__(self, config_path: str = CONFIG_FILE):
        """Initialize processor with config"""
        self.config_path = config_path
        self.config = self._load_config()
        self.state = self._load_state()
        self.yolo_processor = None
        
        if YOLO_AVAILABLE:
            self._init_yolo_processor()
        
        logger.info("✅ Production YOLO blur processor initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _init_yolo_processor(self):
        """Initialize YOLO blur processor"""
        try:
            yolo_config = self.config.get('ai_curation', {})
            model_path = yolo_config.get('yolo_model', 'yolov8n.pt')
            device = yolo_config.get('device', 'cpu')
            
            self.yolo_processor = YOLOBlurProcessor(model_path=model_path, device=device)
            logger.info("✅ YOLO blur processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO processor: {e}")
            self.yolo_processor = None
    
    def _load_state(self) -> Dict[str, Any]:
        """Load processing state"""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        
        return {
            'processed_files': {},
            'stats': {
                'total_processed': 0,
                'yolo_analyzed': 0,
                'artistic_preserved': 0,
                'technical_rejected': 0,
                'last_updated': None
            }
        }
    
    def _save_state(self):
        """Save processing state"""
        self.state['stats']['last_updated'] = datetime.now().isoformat()
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for tracking"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _load_and_resize_image(self, image_path: Path, target_size: int = 800) -> Optional[np.ndarray]:
        """Load and resize image for analysis"""
        try:
            # Try to load as JPEG first
            if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                image = cv2.imread(str(image_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Resize to target size maintaining aspect ratio
                    height, width = image.shape[:2]
                    if max(height, width) > target_size:
                        if height > width:
                            new_height = target_size
                            new_width = int(width * target_size / height)
                        else:
                            new_width = target_size
                            new_height = int(height * target_size / width)
                        
                        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    return image
            
            # For RAW files, would need RAW processor
            logger.warning(f"RAW file processing not available in production: {image_path.name}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path.name}: {e}")
            return None
    
    def analyze_image_blur(self, image_path: Path) -> Dict[str, Any]:
        """Analyze single image with YOLO-enhanced blur detection"""
        start_time = time.time()
        result = {'file': str(image_path), 'status': 'pending'}
        
        try:
            # Check if already processed
            file_hash = self._get_file_hash(image_path)
            if file_hash in self.state['processed_files']:
                logger.info(f"⏭️ Skipping already processed: {image_path.name}")
                result['status'] = 'skipped'
                result['previous_result'] = self.state['processed_files'][file_hash]
                return result
            
            # Load image
            image = self._load_and_resize_image(image_path)
            if image is None:
                result['status'] = 'error'
                result['message'] = 'Failed to load image'
                return result
            
            result['status'] = 'analyzed'
            result['processing_time'] = time.time() - start_time
            
            # Run YOLO-enhanced blur analysis
            if self.yolo_processor is not None:
                try:
                    logger.info(f"🔍 Analyzing {image_path.name} with YOLO blur detection")
                    
                    # Analyze with YOLO
                    detections, blur_analysis = analyze_preview_with_blur(image, self.yolo_processor)
                    
                    result['detection_count'] = len(detections)
                    result['blur_analysis'] = {
                        'classification': blur_analysis.overall_classification,
                        'subject_sharpness': blur_analysis.subject_sharpness,
                        'background_sharpness': blur_analysis.background_sharpness,
                        'depth_separation': blur_analysis.depth_separation,
                        'motion_blur_detected': blur_analysis.motion_blur_detected,
                        'artistic_intent_score': blur_analysis.artistic_intent_score
                    }
                    
                    # Make quality decision based on blur analysis
                    if blur_analysis.overall_classification == "artistic_shallow_dof":
                        result['quality_decision'] = 'preserve_artistic'
                        self.state['stats']['artistic_preserved'] += 1
                        logger.info(f"🎨 {image_path.name}: Artistic shallow DoF preserved!")
                    elif blur_analysis.overall_classification in ['motion_blur', 'focus_miss']:
                        result['quality_decision'] = 'reject_technical'
                        self.state['stats']['technical_rejected'] += 1
                        logger.info(f"❌ {image_path.name}: Technical blur issue - {blur_analysis.overall_classification}")
                    else:
                        result['quality_decision'] = 'conditional'
                        logger.info(f"⚠️ {image_path.name}: Conditional quality")
                    
                    self.state['stats']['yolo_analyzed'] += 1
                    
                except Exception as e:
                    logger.error(f"YOLO analysis failed for {image_path.name}: {e}")
                    result['quality_decision'] = 'fallback_accept'
            else:
                logger.warning(f"YOLO not available, using fallback for {image_path.name}")
                result['quality_decision'] = 'fallback_accept'
            
            # Update state
            self.state['processed_files'][file_hash] = {
                'file_path': str(image_path),
                'processed_at': datetime.now().isoformat(),
                'quality_decision': result['quality_decision'],
                'blur_classification': result.get('blur_analysis', {}).get('classification', 'unknown')
            }
            
            self.state['stats']['total_processed'] += 1
            self._save_state()
            
            logger.info(f"✅ Completed {image_path.name} in {result['processing_time']:.1f}s - {result['quality_decision']}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {image_path.name}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def scan_and_analyze(self, source_dir: str, max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Scan directory and analyze images with YOLO-enhanced blur analysis
        
        Args:
            source_dir: Directory to scan for image files
            max_files: Maximum files to process (None for unlimited)
            
        Returns:
            Analysis summary
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return {'status': 'error', 'message': 'Source directory not found'}
        
        # Find supported image files
        image_extensions = {'.jpg', '.jpeg', '.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.orf'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"**/*{ext}"))
            image_files.extend(source_path.glob(f"**/*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files in {source_dir}")
        
        if max_files:
            image_files = image_files[:max_files]
            logger.info(f"Limited to {max_files} files for analysis")
        
        # Analyze files
        results = []
        start_time = time.time()
        
        for image_file in image_files:
            result = self.analyze_image_blur(image_file)
            results.append(result)
            
            # Show progress periodically
            if len(results) % 5 == 0:
                self._show_progress_stats()
        
        total_time = time.time() - start_time
        
        # Final summary
        summary = {
            'total_files': len(image_files),
            'analyzed': len([r for r in results if r['status'] == 'analyzed']),
            'skipped': len([r for r in results if r['status'] == 'skipped']),
            'errors': len([r for r in results if r['status'] == 'error']),
            'total_time': total_time,
            'results': results
        }
        
        logger.info(f"✅ Analysis complete: {summary['analyzed']} analyzed, "
                   f"{summary['skipped']} skipped, {summary['errors']} errors "
                   f"in {total_time:.1f}s")
        
        self._show_final_stats()
        return summary
    
    def _show_progress_stats(self):
        """Show current processing statistics"""
        stats = self.state['stats']
        logger.info(f"📊 Progress: {stats['total_processed']} total, "
                   f"{stats['yolo_analyzed']} YOLO analyzed, "
                   f"{stats['artistic_preserved']} artistic preserved, "
                   f"{stats['technical_rejected']} technical rejected")
    
    def _show_final_stats(self):
        """Show final processing statistics"""
        stats = self.state['stats']
        
        logger.info("📊 Final YOLO Blur Analysis Statistics:")
        logger.info(f"  Total images analyzed: {stats['total_processed']}")
        logger.info(f"  YOLO analysis performed: {stats['yolo_analyzed']}")
        logger.info(f"  Artistic images preserved: {stats['artistic_preserved']}")
        logger.info(f"  Technical blur rejected: {stats['technical_rejected']}")
        
        if stats['yolo_analyzed'] > 0:
            artistic_rate = stats['artistic_preserved'] / stats['yolo_analyzed'] * 100
            logger.info(f"  Artistic preservation rate: {artistic_rate:.1f}%")
            
            if stats['artistic_preserved'] > 0:
                logger.info(f"  🎨 Successfully identified and preserved artistic bokeh")
                logger.info(f"  📈 Subject-aware analysis prevented false rejections")


def main():
    """Main entry point"""
    processor = ProductionYOLOBlurProcessor()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
        max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
        
        logger.info(f"🚀 Starting YOLO-Enhanced Blur Analysis")
        logger.info(f"  Source directory: {source_dir}")
        logger.info(f"  Max files: {max_files or 'unlimited'}")
        logger.info(f"  YOLO available: {YOLO_AVAILABLE}")
        logger.info(f"  Features: Subject-aware blur analysis, artistic intent detection")
        
        summary = processor.scan_and_analyze(source_dir, max_files)
        
        if summary.get('status') == 'error' or summary.get('errors', 0) > 0:
            sys.exit(1)
    else:
        print("Usage: python3 production_yolo_blur_processor.py <source_dir> [max_files]")
        print("Features: YOLO-enhanced subject-aware blur analysis")
        sys.exit(1)


if __name__ == "__main__":
    main()