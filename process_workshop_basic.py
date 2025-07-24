#!/usr/bin/env python3
"""
Basic workshop photo processor - Direct RAW to JPEG conversion with YOLO analysis
"""

import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime
import yaml
import rawpy
import cv2
import numpy as np
import json

sys.path.insert(0, '/Users/sam/dev/photosight')
from photosight.processing.yolo_integration import analyze_preview_with_blur

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

SD_CARD_PATH = "/Volumes/Untitled/DCIM/100MSDCF"
OUTPUT_BASE = "/Users/sam/Desktop/photosight_output/enneagram_workshop"
PHOTOSHOOT_TAG = "enneagram_workshop"

def process_raw_simple(raw_file, output_path, metadata=None):
    """Simple RAW to JPEG conversion with basic enhancements"""
    try:
        with rawpy.imread(str(raw_file)) as raw:
            # Process with good defaults
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                bright=1.2,  # Slight brightness boost
                exp_shift=0.5,  # Half stop exposure compensation
                output_bps=8
            )
        
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Basic enhancements
        # Slight contrast boost
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Save with metadata
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        cv2.imwrite(str(output_path), enhanced, encode_params)
        
        # Save metadata separately
        if metadata:
            meta_path = output_path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return False

def main(limit=None):
    """Process workshop photos"""
    # Load config for YOLO
    with open('/Users/sam/dev/photosight/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup directories
    Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)
    accepted_dir = Path(OUTPUT_BASE) / 'accepted'
    rejected_dir = Path(OUTPUT_BASE) / 'rejected'
    accepted_dir.mkdir(exist_ok=True)
    rejected_dir.mkdir(exist_ok=True)
    
    # Get photos
    raw_files = sorted(Path(SD_CARD_PATH).glob("*.ARW"))
    if limit:
        raw_files = raw_files[:limit]
    
    stats = {'total': len(raw_files), 'accepted': 0, 'rejected': 0, 'people': 0}
    
    logger.info(f"🎯 ENNEAGRAM WORKSHOP - Processing {stats['total']} photos")
    logger.info("="*50)
    
    for idx, raw_file in enumerate(raw_files, 1):
        logger.info(f"\n[{idx}/{stats['total']}] {raw_file.name}")
        
        try:
            # Quick preview for YOLO
            with rawpy.imread(str(raw_file)) as raw:
                preview = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8)
                # Resize for YOLO
                h, w = preview.shape[:2]
                if max(h, w) > 800:
                    scale = 800 / max(h, w)
                    preview = cv2.resize(preview, (int(w*scale), int(h*scale)))
            
            # YOLO analysis
            yolo_result, blur_result = analyze_preview_with_blur(preview, config)
            
            # Log findings
            if yolo_result.person_count > 0:
                stats['people'] += 1
                logger.info(f"  ✓ People: {yolo_result.person_count}")
            
            # Decision
            if blur_result.meets_quality_threshold:
                logger.info(f"  ✓ Quality: {blur_result.overall_classification}")
                
                # Process full resolution
                output_path = accepted_dir / f"{raw_file.stem}.jpg"
                metadata = {
                    'photoshoot': PHOTOSHOOT_TAG,
                    'date': '2025-07-23',
                    'people_count': yolo_result.person_count,
                    'blur_class': blur_result.overall_classification,
                    'objects': [d.class_name for d in yolo_result.detections]
                }
                
                if process_raw_simple(raw_file, output_path, metadata):
                    stats['accepted'] += 1
                    logger.info(f"  ✅ SAVED: {output_path.name}")
            else:
                stats['rejected'] += 1
                # Copy RAW to rejected folder
                shutil.copy2(raw_file, rejected_dir / raw_file.name)
                logger.info(f"  ❌ REJECTED: {blur_result.overall_classification}")
                
        except Exception as e:
            logger.error(f"  ⚠️ Error: {e}")
            stats['rejected'] += 1
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("✅ PROCESSING COMPLETE")
    logger.info(f"Accepted: {stats['accepted']}/{stats['total']} ({stats['accepted']/stats['total']*100:.0f}%)")
    logger.info(f"People photos: {stats['people']}")
    logger.info(f"Output: {accepted_dir}")
    
    # Save summary
    with open(Path(OUTPUT_BASE) / "summary.txt", 'w') as f:
        f.write(f"Enneagram Workshop - {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"Total: {stats['total']}\n")
        f.write(f"Accepted: {stats['accepted']} ({stats['accepted']/stats['total']*100:.0f}%)\n")
        f.write(f"People detected: {stats['people']} photos\n")

if __name__ == "__main__":
    # Process all photos
    main()