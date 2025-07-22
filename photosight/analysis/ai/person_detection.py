"""
Person detection using YOLOv8 for photo curation
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from ultralytics import YOLO
import cv2

logger = logging.getLogger(__name__)


class PersonDetector:
    """Detect and analyze people in photos using YOLOv8"""
    
    def __init__(self, model_size: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize person detector
        
        Args:
            model_size: YOLO model size (n/s/m/l/x)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_size = model_size
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_size)
            logger.info(f"Loaded YOLO model: {self.model_size}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
            
    def detect_people(self, image: np.ndarray) -> Dict:
        """
        Detect people in image
        
        Args:
            image: Image array (RGB)
            
        Returns:
            Dictionary with detection results
        """
        results = {
            'people_count': 0,
            'people_boxes': [],
            'people_confidence': [],
            'main_subject': None,
            'group_photo': False,
            'scene_type': 'unknown'
        }
        
        try:
            # Run YOLO detection
            detections = self.model(image, verbose=False)[0]
            
            # Filter for person class (class 0 in COCO)
            person_detections = []
            for box in detections.boxes:
                if box.cls == 0 and box.conf >= self.confidence_threshold:
                    person_detections.append(box)
                    
            results['people_count'] = len(person_detections)
            
            # Extract bounding boxes and confidence scores
            for det in person_detections:
                bbox = det.xyxy[0].cpu().numpy()
                conf = det.conf.cpu().numpy()
                
                results['people_boxes'].append(bbox)
                results['people_confidence'].append(float(conf))
                
            # Analyze scene type
            if results['people_count'] == 0:
                results['scene_type'] = 'no_people'
            elif results['people_count'] == 1:
                results['scene_type'] = 'portrait'
                results['main_subject'] = self._analyze_main_subject(
                    image, results['people_boxes'][0]
                )
            elif results['people_count'] <= 3:
                results['scene_type'] = 'small_group'
                results['group_photo'] = True
            else:
                results['scene_type'] = 'large_group'
                results['group_photo'] = True
                
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            
        return results
        
    def _analyze_main_subject(self, image: np.ndarray, bbox: np.ndarray) -> Dict:
        """
        Analyze the main subject (single person)
        
        Args:
            image: Full image
            bbox: Bounding box of person [x1, y1, x2, y2]
            
        Returns:
            Analysis of main subject
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Calculate subject properties
        subject_width = x2 - x1
        subject_height = y2 - y1
        subject_area = subject_width * subject_height
        image_area = h * w
        
        # Subject size ratio
        size_ratio = subject_area / image_area
        
        # Subject position (center of bbox)
        center_x = (x1 + x2) / 2 / w
        center_y = (y1 + y2) / 2 / h
        
        # Determine shot type based on subject size
        if size_ratio > 0.5:
            shot_type = 'close_up'
        elif size_ratio > 0.25:
            shot_type = 'medium_shot'
        elif size_ratio > 0.1:
            shot_type = 'full_body'
        else:
            shot_type = 'environmental'
            
        return {
            'size_ratio': size_ratio,
            'position': (center_x, center_y),
            'shot_type': shot_type,
            'bbox': bbox.tolist()
        }
        
    def analyze_composition(self, image: np.ndarray, detections: Dict) -> Dict:
        """
        Analyze composition with detected people
        
        Args:
            image: Image array
            detections: Detection results from detect_people
            
        Returns:
            Composition analysis
        """
        h, w = image.shape[:2]
        composition = {
            'people_centered': False,
            'rule_of_thirds': False,
            'balanced': False,
            'leading_lines': False
        }
        
        if detections['people_count'] == 0:
            return composition
            
        # Check if people are centered
        if detections['people_count'] == 1 and detections['main_subject']:
            center_x, center_y = detections['main_subject']['position']
            
            # Check if centered (within 20% of center)
            if 0.4 <= center_x <= 0.6 and 0.4 <= center_y <= 0.6:
                composition['people_centered'] = True
                
            # Check rule of thirds
            thirds_x = [1/3, 2/3]
            thirds_y = [1/3, 2/3]
            
            for tx in thirds_x:
                for ty in thirds_y:
                    if abs(center_x - tx) < 0.1 and abs(center_y - ty) < 0.1:
                        composition['rule_of_thirds'] = True
                        break
                        
        # Check balance for group photos
        elif detections['group_photo']:
            # Calculate center of mass of all people
            total_x = 0
            total_y = 0
            total_weight = 0
            
            for i, bbox in enumerate(detections['people_boxes']):
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2 / w
                center_y = (y1 + y2) / 2 / h
                weight = detections['people_confidence'][i]
                
                total_x += center_x * weight
                total_y += center_y * weight
                total_weight += weight
                
            if total_weight > 0:
                com_x = total_x / total_weight
                com_y = total_y / total_weight
                
                # Check if center of mass is near image center
                if 0.35 <= com_x <= 0.65 and 0.35 <= com_y <= 0.65:
                    composition['balanced'] = True
                    
        return composition
        
    def get_quality_score(self, detections: Dict, composition: Dict) -> float:
        """
        Calculate quality score based on people detection and composition
        
        Args:
            detections: Detection results
            composition: Composition analysis
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        
        # Base score for having people in frame
        if detections['people_count'] > 0:
            score += 0.3
            
            # Bonus for appropriate number of people
            if detections['scene_type'] in ['portrait', 'small_group']:
                score += 0.2
                
            # Composition bonuses
            if composition['rule_of_thirds']:
                score += 0.2
            elif composition['people_centered']:
                score += 0.1
                
            if composition['balanced']:
                score += 0.1
                
            # Confidence bonus
            if detections['people_confidence']:
                avg_confidence = np.mean(detections['people_confidence'])
                score += 0.2 * avg_confidence
                
        return min(1.0, score)