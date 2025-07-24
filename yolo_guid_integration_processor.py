#!/usr/bin/env python3
"""
PhotoSight YOLO-GUID Integration Processor
Integrates YOLO object detection with GUID-based photo management system
"""

import os
import sys
import json
import uuid
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from requests.auth import HTTPBasicAuth
import time
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_guid_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class YOLODetection:
    """Enhanced YOLO detection with GUID integration"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    subject_priority: float = 0.5
    blur_analysis: Optional[Dict] = None

@dataclass
class IntegratedPhotoAnalysis:
    """Unified photo analysis combining GUID and YOLO data"""
    photo_guid: str
    raw_filename: str
    jpeg_filename: Optional[str]
    yolo_detections: List[YOLODetection]
    blur_classification: str
    overall_confidence: float
    subject_count: int
    processing_timestamp: str
    camera_model: str
    lens_model: str

class YOLOGuidIntegrationProcessor:
    """Processes photos with YOLO analysis and links to GUID records"""
    
    def __init__(self):
        self.base_url = 'https://GFCA71B2AACCE62-PHOTOSIGHTDB.adb.us-chicago-1.oraclecloudapps.com/ords'
        self.username = 'admin'
        self.password = 'REDACTED@321'
        
        # Load YOLO model
        logger.info("🤖 Loading YOLO model...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Subject priority mapping for blur analysis
        self.subject_priorities = {
            'person': 1.0,
            'dog': 0.9,
            'cat': 0.9,
            'bird': 0.8,
            'horse': 0.8,
            'bicycle': 0.6,
            'motorcycle': 0.6,
            'car': 0.5,
            'truck': 0.5,
            'boat': 0.5
        }
        
    def execute_sql(self, sql_statement: str) -> List[Dict]:
        """Execute SQL query via Oracle REST API"""
        try:
            response = requests.post(
                f'{self.base_url}/admin/_/sql',
                auth=HTTPBasicAuth(self.username, self.password),
                json={'statementText': sql_statement},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'resultSet' in result and 'items' in result['resultSet']:
                    return result['resultSet']['items']
            else:
                logger.error(f"SQL query failed: {response.status_code} - {response.text}")
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
        
        return []
    
    def get_guid_records_without_yolo(self, limit: int = 50) -> List[Dict]:
        """Fetch GUID records that don't have YOLO analysis yet"""
        sql = f"""
        SELECT pg.photo_guid, pg.raw_filename, pg.jpeg_filename, pg.camera_model, pg.lens_model
        FROM PHOTOS_GUID pg
        LEFT JOIN YOLO_DETECTIONS yd ON pg.photo_guid = yd.photo_guid
        WHERE yd.photo_guid IS NULL
        AND pg.jpeg_filename IS NOT NULL
        AND ROWNUM <= {limit}
        ORDER BY pg.created_date DESC
        """
        
        return self.execute_sql(sql)
    
    def analyze_image_with_yolo(self, image_path: str) -> Tuple[List[YOLODetection], str]:
        """Analyze image with YOLO and determine blur classification"""
        try:
            # Load and analyze image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return [], "error"
            
            # Run YOLO detection
            results = self.yolo_model(image, conf=0.25)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()
                        
                        # Calculate subject priority
                        priority = self.subject_priorities.get(class_name, 0.3)
                        
                        detection = YOLODetection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                            subject_priority=priority
                        )
                        detections.append(detection)
            
            # Determine blur classification based on subjects
            blur_classification = self._classify_blur_with_subjects(image, detections)
            
            return detections, blur_classification
            
        except Exception as e:
            logger.error(f"YOLO analysis failed for {image_path}: {e}")
            return [], "error"
    
    def _classify_blur_with_subjects(self, image: np.ndarray, detections: List[YOLODetection]) -> str:
        """Classify blur type considering detected subjects"""
        if not detections:
            return "no_subjects_detected"
        
        # Find highest priority subject
        max_priority = max(det.subject_priority for det in detections)
        priority_subjects = [det for det in detections if det.subject_priority >= max_priority * 0.8]
        
        # Analyze blur in subject regions
        subject_sharpness_scores = []
        
        for detection in priority_subjects:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            roi = image[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Calculate Laplacian variance for sharpness
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                subject_sharpness_scores.append(laplacian_var)
        
        if not subject_sharpness_scores:
            return "analysis_failed"
        
        avg_subject_sharpness = np.mean(subject_sharpness_scores)
        
        # Classify based on sharpness threshold
        if avg_subject_sharpness > 500:
            return "sharp_throughout"
        elif avg_subject_sharpness > 200:
            return "mixed_sharpness"
        elif len(priority_subjects) == 1 and avg_subject_sharpness > 100:
            return "artistic_shallow_dof"  # Likely intentional bokeh
        else:
            return "motion_blur"
    
    def store_yolo_analysis(self, analysis: IntegratedPhotoAnalysis) -> bool:
        """Store YOLO analysis results linked to GUID record"""
        try:
            # Prepare detections JSON
            detections_json = json.dumps([asdict(det) for det in analysis.yolo_detections])
            detections_json = detections_json.replace("'", "''")  # Escape quotes
            
            # Insert YOLO analysis
            sql = f"""
            MERGE INTO YOLO_DETECTIONS yd
            USING (SELECT '{analysis.photo_guid}' as photo_guid FROM dual) d
            ON (yd.photo_guid = d.photo_guid)
            WHEN MATCHED THEN UPDATE SET
                detections_json = '{detections_json}',
                blur_classification = '{analysis.blur_classification}',
                overall_confidence = {analysis.overall_confidence},
                subject_count = {analysis.subject_count},
                processing_timestamp = SYSTIMESTAMP
            WHEN NOT MATCHED THEN INSERT (
                photo_guid, detections_json, blur_classification,
                overall_confidence, subject_count, processing_timestamp
            ) VALUES (
                '{analysis.photo_guid}', '{detections_json}', '{analysis.blur_classification}',
                {analysis.overall_confidence}, {analysis.subject_count}, SYSTIMESTAMP
            )
            """
            
            result = self.execute_sql(sql)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store YOLO analysis for {analysis.photo_guid}: {e}")
            return False
    
    def process_guid_records_batch(self, batch_size: int = 10) -> Dict:
        """Process a batch of GUID records with YOLO analysis"""
        logger.info(f"🔍 Fetching {batch_size} GUID records without YOLO analysis")
        
        records = self.get_guid_records_without_yolo(batch_size)
        if not records:
            logger.info("✅ No more records to process")
            return {"processed": 0, "succeeded": 0, "failed": 0}
        
        processed = 0
        succeeded = 0
        failed = 0
        
        for record in records:
            processed += 1
            photo_guid = record['photo_guid']
            jpeg_filename = record.get('jpeg_filename')
            
            if not jpeg_filename:
                logger.warning(f"⚠️  No JPEG filename for GUID {photo_guid}")
                failed += 1
                continue
            
            # Construct JPEG path (adjust based on your storage structure)
            jpeg_path = f"/Users/sam/dev/photosight/processed_jpegs/{jpeg_filename}"
            
            if not os.path.exists(jpeg_path):
                logger.warning(f"⚠️  JPEG not found: {jpeg_path}")
                failed += 1
                continue
            
            logger.info(f"🎯 Processing {photo_guid} - {jpeg_filename}")
            
            # Analyze with YOLO
            detections, blur_classification = self.analyze_image_with_yolo(jpeg_path)
            
            if not detections and blur_classification == "error":
                failed += 1
                continue
            
            # Calculate overall confidence
            overall_confidence = np.mean([det.confidence for det in detections]) if detections else 0.0
            
            # Create integrated analysis
            analysis = IntegratedPhotoAnalysis(
                photo_guid=photo_guid,
                raw_filename=record['raw_filename'],
                jpeg_filename=jpeg_filename,
                yolo_detections=detections,
                blur_classification=blur_classification,
                overall_confidence=overall_confidence,
                subject_count=len(detections),
                processing_timestamp=datetime.now().isoformat(),
                camera_model=record.get('camera_model', ''),
                lens_model=record.get('lens_model', '')
            )
            
            # Store in database
            if self.store_yolo_analysis(analysis):
                succeeded += 1
                logger.info(f"✅ Stored analysis for {photo_guid}: {len(detections)} detections, {blur_classification}")
            else:
                failed += 1
            
            # Rate limiting for Oracle Free Tier
            time.sleep(0.5)
        
        return {
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed
        }
    
    def run_continuous_processing(self, batch_size: int = 10, max_batches: int = 50):
        """Run continuous YOLO-GUID integration processing"""
        logger.info("🚀 Starting YOLO-GUID Integration Processing")
        logger.info(f"📊 Batch size: {batch_size}, Max batches: {max_batches}")
        
        total_processed = 0
        total_succeeded = 0
        total_failed = 0
        batch_count = 0
        
        while batch_count < max_batches:
            batch_count += 1
            logger.info(f"📦 Processing batch {batch_count}/{max_batches}")
            
            results = self.process_guid_records_batch(batch_size)
            
            if results["processed"] == 0:
                logger.info("🎯 All available records processed!")
                break
            
            total_processed += results["processed"]
            total_succeeded += results["succeeded"]
            total_failed += results["failed"]
            
            logger.info(f"📊 Batch {batch_count} results: {results['succeeded']}/{results['processed']} successful")
            
            # Brief pause between batches
            time.sleep(2)
        
        logger.info("🎯 YOLO-GUID Integration Processing Complete!")
        logger.info(f"📈 Final Statistics:")
        logger.info(f"  Total processed: {total_processed}")
        logger.info(f"  Successful: {total_succeeded}")
        logger.info(f"  Failed: {total_failed}")
        logger.info(f"  Success rate: {(total_succeeded/total_processed*100):.1f}%" if total_processed > 0 else "N/A")

def main():
    """Main execution function"""
    processor = YOLOGuidIntegrationProcessor()
    
    # Run processing with configurable parameters
    batch_size = int(os.getenv('BATCH_SIZE', '10'))
    max_batches = int(os.getenv('MAX_BATCHES', '50'))
    
    processor.run_continuous_processing(batch_size, max_batches)

if __name__ == "__main__":
    main()