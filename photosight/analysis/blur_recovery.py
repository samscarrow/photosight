"""
Blur recovery module for PhotoSight
Implements two-pass blur detection with subject-aware recovery
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BlurRecoveryAnalyzer:
    """Analyzes rejected blurry photos for potential recovery based on subject focus"""
    
    def __init__(self, config: Dict):
        """Initialize the blur recovery analyzer"""
        self.config = config['technical_filters']['sharpness']
        self.base_threshold = self.config['laplacian_variance_minimum']
        self.recovery_threshold = self.config.get('blur_recovery_threshold', 40.0)
        self.overlap_threshold = self.config.get('subject_focus_overlap_threshold', 0.3)
        self.enable_recovery = self.config.get('enable_blur_recovery', True)
        
        # Initialize YOLO for subject detection if available
        self.yolo_model = None
        if config.get('ai_curation', {}).get('enabled', False):
            try:
                from ultralytics import YOLO
                model_name = config.get('ai_curation', {}).get('yolo_model', 'yolov8n.pt')
                self.yolo_model = YOLO(model_name)
                logger.info(f"Loaded YOLO model for subject detection: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load YOLO model: {e}")
                
        # Face detection cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def analyze_for_recovery(self, image_array: np.ndarray, metadata: Optional[Dict] = None) -> Dict:
        """
        Analyze a blurry photo for potential recovery
        
        Args:
            image_array: RGB image array
            metadata: Image metadata
            
        Returns:
            Recovery analysis results
        """
        if not self.enable_recovery:
            return {'recoverable': False, 'reason': 'recovery_disabled'}
            
        # Step 1: Detect subjects in the image
        subjects = self._detect_subjects(image_array)
        if not subjects:
            return {
                'recoverable': False, 
                'reason': 'no_subjects_detected',
                'subjects': []
            }
            
        # Step 2: Generate focus map
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        focus_map = self._generate_focus_map(gray)
        
        # Step 3: Analyze focus quality in subject areas
        subject_focus_results = []
        for subject in subjects:
            focus_quality = self._analyze_subject_focus(gray, focus_map, subject)
            subject_focus_results.append(focus_quality)
            
        # Step 4: Determine if photo is recoverable
        recoverable, recovery_score = self._determine_recoverability(subject_focus_results)
        
        return {
            'recoverable': recoverable,
            'recovery_score': recovery_score,
            'subjects': subjects,
            'subject_focus_results': subject_focus_results,
            'focus_map': focus_map,
            'reason': 'subjects_in_focus' if recoverable else 'subjects_not_in_focus'
        }
        
    def _detect_subjects(self, image_array: np.ndarray) -> List[Dict]:
        """Detect subjects in the image with bias for human faces"""
        subjects = []
        
        # Priority 1: Detect faces using OpenCV cascade
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            subjects.append({
                'type': 'face',
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.9,  # High confidence for face detection
                'priority': 1
            })
            
        # Priority 2: Use YOLO for person detection if available
        if self.yolo_model:
            try:
                results = self.yolo_model(image_array, verbose=False)
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Check if detected object is a person (class 0 in COCO)
                            if int(box.cls) == 0:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = float(box.conf)
                                
                                # Check if this person bbox overlaps with any face
                                is_face_overlap = False
                                for face_subject in subjects:
                                    if face_subject['type'] == 'face':
                                        if self._calculate_iou(face_subject['bbox'], [x1, y1, x2, y2]) > 0.3:
                                            is_face_overlap = True
                                            break
                                            
                                if not is_face_overlap:
                                    subjects.append({
                                        'type': 'person',
                                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                        'confidence': confidence,
                                        'priority': 2
                                    })
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
                
        # Priority 3: Detect high-contrast regions as potential subjects
        if not subjects:
            subjects.extend(self._detect_high_contrast_regions(gray))
            
        return subjects
        
    def _detect_high_contrast_regions(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect high-contrast regions that might be subjects"""
        regions = []
        
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        h, w = gray_image.shape
        min_area = (h * w) * 0.01  # At least 1% of image
        max_area = (h * w) * 0.5   # At most 50% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'type': 'high_contrast_region',
                    'bbox': [x, y, x + w, y + h],
                    'confidence': 0.5,
                    'priority': 3
                })
                
        # Sort by area and keep top 5
        regions.sort(key=lambda r: (r['bbox'][2] - r['bbox'][0]) * (r['bbox'][3] - r['bbox'][1]), reverse=True)
        return regions[:5]
        
    def _generate_focus_map(self, gray_image: np.ndarray, window_size: int = 32) -> np.ndarray:
        """Generate a detailed focus map using windowed Laplacian variance"""
        h, w = gray_image.shape
        
        # Use smaller stride for more detailed map
        stride = window_size // 4
        
        # Initialize focus map
        map_h = (h - window_size) // stride + 1
        map_w = (w - window_size) // stride + 1
        focus_map = np.zeros((map_h, map_w))
        
        # Calculate local sharpness
        for i in range(0, h - window_size, stride):
            for j in range(0, w - window_size, stride):
                window = gray_image[i:i+window_size, j:j+window_size]
                
                # Use Laplacian variance as sharpness measure
                laplacian = cv2.Laplacian(window, cv2.CV_64F)
                variance = laplacian.var()
                
                focus_map[i//stride, j//stride] = variance
                
        # Resize back to original dimensions
        focus_map_resized = cv2.resize(focus_map, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to 0-1 range
        if focus_map_resized.max() > 0:
            focus_map_resized = focus_map_resized / focus_map_resized.max()
            
        return focus_map_resized
        
    def _analyze_subject_focus(self, gray_image: np.ndarray, focus_map: np.ndarray, 
                              subject: Dict) -> Dict:
        """Analyze focus quality in subject area"""
        x1, y1, x2, y2 = subject['bbox']
        
        # Extract subject region from gray image and focus map
        subject_gray = gray_image[y1:y2, x1:x2]
        subject_focus = focus_map[y1:y2, x1:x2]
        
        # Calculate sharpness metrics
        laplacian = cv2.Laplacian(subject_gray, cv2.CV_64F)
        subject_sharpness = laplacian.var()
        
        # Calculate focus statistics
        mean_focus = np.mean(subject_focus)
        max_focus = np.max(subject_focus)
        focus_coverage = np.sum(subject_focus > 0.5) / subject_focus.size  # Percentage of well-focused pixels
        
        # Check if subject meets recovery threshold
        is_sharp = subject_sharpness >= self.recovery_threshold
        
        # Calculate high-focus zone overlap
        high_focus_mask = focus_map > 0.6  # Top 40% focus areas
        subject_mask = np.zeros_like(focus_map, dtype=bool)
        subject_mask[y1:y2, x1:x2] = True
        
        overlap_area = np.sum(high_focus_mask & subject_mask)
        subject_area = np.sum(subject_mask)
        overlap_ratio = overlap_area / subject_area if subject_area > 0 else 0
        
        return {
            'subject_type': subject['type'],
            'subject_bbox': subject['bbox'],
            'subject_confidence': subject['confidence'],
            'subject_priority': subject['priority'],
            'sharpness_variance': subject_sharpness,
            'mean_focus': mean_focus,
            'max_focus': max_focus,
            'focus_coverage': focus_coverage,
            'is_sharp': is_sharp,
            'high_focus_overlap': overlap_ratio,
            'meets_overlap_threshold': overlap_ratio >= self.overlap_threshold
        }
        
    def _determine_recoverability(self, subject_focus_results: List[Dict]) -> Tuple[bool, float]:
        """Determine if photo is recoverable based on subject analysis"""
        if not subject_focus_results:
            return False, 0.0
            
        # Sort by priority (faces first, then people, then other regions)
        sorted_results = sorted(subject_focus_results, key=lambda x: x['subject_priority'])
        
        # Check if any high-priority subject is sharp
        for result in sorted_results:
            if result['is_sharp'] and result['meets_overlap_threshold']:
                # Calculate recovery score based on subject quality
                recovery_score = self._calculate_recovery_score(result)
                return True, recovery_score
                
        # If no individual subject qualifies, check if multiple subjects together justify recovery
        sharp_subjects = [r for r in sorted_results if r['is_sharp']]
        if len(sharp_subjects) >= 2:
            avg_overlap = np.mean([r['high_focus_overlap'] for r in sharp_subjects])
            if avg_overlap >= self.overlap_threshold * 0.8:  # Slightly lower threshold for multiple subjects
                recovery_score = np.mean([self._calculate_recovery_score(r) for r in sharp_subjects])
                return True, recovery_score
                
        return False, 0.0
        
    def _calculate_recovery_score(self, focus_result: Dict) -> float:
        """Calculate a recovery confidence score (0-100)"""
        # Weight different factors
        weights = {
            'sharpness': 0.3,
            'focus_coverage': 0.2,
            'overlap': 0.3,
            'subject_confidence': 0.2
        }
        
        # Normalize sharpness score
        sharpness_score = min(focus_result['sharpness_variance'] / self.base_threshold, 1.0)
        
        # Calculate weighted score
        score = (
            sharpness_score * weights['sharpness'] +
            focus_result['focus_coverage'] * weights['focus_coverage'] +
            focus_result['high_focus_overlap'] * weights['overlap'] +
            focus_result['subject_confidence'] * weights['subject_confidence']
        ) * 100
        
        # Boost score for faces
        if focus_result['subject_type'] == 'face':
            score *= 1.2
            
        return min(score, 100.0)
        
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = bbox1_area + bbox2_area - intersection
        
        return intersection / union if union > 0 else 0.0


def process_blur_recovery(config: Dict, rejected_folder: Path, accepted_folder: Path, 
                         file_manager, raw_processor, num_threads: int = 4) -> Dict:
    """
    Process blur recovery for rejected photos
    
    Args:
        config: Configuration dictionary
        rejected_folder: Path to rejected/blurry folder
        accepted_folder: Path to accepted folder
        file_manager: FileManager instance
        raw_processor: RawProcessor instance
        num_threads: Number of processing threads
        
    Returns:
        Recovery statistics
    """
    analyzer = BlurRecoveryAnalyzer(config)
    
    # Find all blurry rejected files
    blurry_files = list(rejected_folder.glob("*.ARW")) + list(rejected_folder.glob("*.arw"))
    
    if not blurry_files:
        logger.info("No blurry files found for recovery analysis")
        return {'total': 0, 'recovered': 0, 'failed': 0}
        
    logger.info(f"Found {len(blurry_files)} blurry files for recovery analysis")
    
    recovered_files = []
    failed_files = []
    
    def analyze_file(file_path):
        """Analyze a single file for recovery"""
        try:
            # Load image
            raw, error = raw_processor.load_raw_file(file_path)
            if error:
                return None, f"Failed to load: {error}"
                
            preview = raw_processor.get_preview_array(raw)
            if preview is None:
                return None, "Failed to generate preview"
                
            # Extract metadata
            metadata = raw_processor.extract_metadata(file_path)
            
            # Analyze for recovery
            recovery_result = analyzer.analyze_for_recovery(preview, metadata)
            
            if recovery_result['recoverable']:
                return file_path, recovery_result
            else:
                return None, recovery_result['reason']
                
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None, str(e)
            
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(analyze_file, f): f for f in blurry_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing blur recovery"):
            file_path = futures[future]
            try:
                result, info = future.result()
                if result:
                    recovered_files.append((result, info))
                else:
                    failed_files.append((file_path, info))
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                failed_files.append((file_path, str(e)))
                
    # Move recovered files
    logger.info(f"Recovered {len(recovered_files)} photos from blur rejection")
    
    for file_path, recovery_info in recovered_files:
        try:
            # Calculate destination
            relative_path = file_path.relative_to(rejected_folder)
            destination = accepted_folder / relative_path
            
            # Move file and any sidecars
            file_manager.organize_file(file_path, destination.parent, move_sidecars=True)
            
            logger.info(f"Recovered: {file_path.name} (score: {recovery_info['recovery_score']:.1f})")
            
        except Exception as e:
            logger.error(f"Error moving recovered file {file_path}: {e}")
            
    return {
        'total': len(blurry_files),
        'recovered': len(recovered_files),
        'failed': len(failed_files),
        'recovery_rate': len(recovered_files) / len(blurry_files) if blurry_files else 0
    }