"""
YOLO Integration Module for PhotoSight
Provides early object detection during preview generation
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO (ultralytics) not available. Early object detection disabled.")

logger = logging.getLogger(__name__)


# Subject priority weights for blur analysis
SUBJECT_BLUR_PRIORITIES = {
    'person': 1.0,      # People must be sharp
    'dog': 0.9,         # Pets very high priority
    'cat': 0.9,         # Pets very high priority
    'bird': 0.8,        # Animals high priority
    'horse': 0.8,       # Animals high priority
    'car': 0.7,         # Vehicles medium-high priority
    'motorcycle': 0.7,  # Vehicles medium-high priority
    'bicycle': 0.6,     # Objects medium priority
    'sports ball': 0.5, # Action objects medium priority
    'bottle': 0.3,      # Small objects lower priority
    'cup': 0.3,         # Small objects lower priority
    'book': 0.2,        # Background objects low priority
}

# Blur quality thresholds
BLUR_THRESHOLDS = {
    'subject_sharp_min': 80.0,      # Minimum Laplacian variance for sharp subjects
    'subject_acceptable_min': 50.0,  # Minimum acceptable sharpness
    'background_soft_max': 40.0,     # Maximum for artistic background blur
    'motion_blur_uniform': 0.7,      # Uniformity threshold for motion blur detection
}


@dataclass
class YOLODetection:
    """Single YOLO detection result"""
    class_name: str
    confidence: float
    bbox_x: float  # Normalized coordinates (0-1)
    bbox_y: float
    bbox_width: float
    bbox_height: float
    
    # Blur analysis properties
    subject_priority: float = 0.0  # Priority weight for blur analysis (0-1)
    depth_layer: str = "unknown"   # foreground/midground/background
    
    def get_pixel_bbox(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized bbox to pixel coordinates"""
        x1 = int(self.bbox_x * img_width)
        y1 = int(self.bbox_y * img_height)
        x2 = int((self.bbox_x + self.bbox_width) * img_width)
        y2 = int((self.bbox_y + self.bbox_height) * img_height)
        return x1, y1, x2, y2


@dataclass
class YOLOAnalysis:
    """Complete YOLO analysis result"""
    detections: List[YOLODetection]
    person_count: int
    object_score: float  # Overall interest score (0-1)
    has_people: bool
    dominant_objects: List[str]  # Most prominent object classes
    processing_time_ms: float
    
    # Blur analysis properties
    priority_subjects: List[YOLODetection] = None  # High-priority objects for sharpness
    subject_regions: List[Tuple[int, int, int, int]] = None  # Subject bounding boxes (pixels)
    background_regions: List[Tuple[int, int, int, int]] = None  # Background regions
    depth_layers: Dict[str, List[YOLODetection]] = None  # Objects by depth layer
    
    def __post_init__(self):
        """Initialize computed properties after creation"""
        if self.priority_subjects is None:
            self.priority_subjects = []
        if self.subject_regions is None:
            self.subject_regions = []
        if self.background_regions is None:
            self.background_regions = []
        if self.depth_layers is None:
            self.depth_layers = {"foreground": [], "midground": [], "background": []}


@dataclass
class SubjectBlurAnalysis:
    """Blur analysis results for subject-aware processing"""
    subject_blur_scores: Dict[str, float]  # Per-subject sharpness scores
    background_blur_score: float           # Background region sharpness
    overall_classification: str            # artistic_bokeh, motion_blur, sharp_throughout, etc.
    meets_quality_threshold: bool          # Pass/fail for image quality
    priority_subject_sharp: bool           # Are high-priority subjects sharp?
    blur_type_confidence: float           # Confidence in blur classification (0-1)
    
    # Detailed metrics
    subject_details: Dict[str, Dict] = None      # Per-subject analysis details
    background_details: Dict[str, Any] = None    # Background analysis details
    artistic_intent_score: float = 0.0          # Likelihood of intentional blur (0-1)
    
    def __post_init__(self):
        if self.subject_details is None:
            self.subject_details = {}
        if self.background_details is None:
            self.background_details = {}


class YOLOBlurProcessor:
    """Enhanced YOLO processor with subject-aware blur analysis"""
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        """Initialize YOLO processor with blur analysis capabilities"""
        self.yolo_processor = YOLOProcessor(model_path, device)
    
    def analyze_with_blur_context(self, preview_image: np.ndarray, 
                                confidence_threshold: float = 0.25) -> Tuple[YOLOAnalysis, SubjectBlurAnalysis]:
        """Analyze image with both object detection and subject-aware blur analysis"""
        # Get YOLO detections
        yolo_analysis = self.yolo_processor.analyze_preview(preview_image, confidence_threshold)
        
        # Enhance YOLO analysis with blur context
        self._enhance_yolo_for_blur(yolo_analysis, preview_image)
        
        # Perform subject-aware blur analysis
        blur_analysis = self._analyze_subject_blur(preview_image, yolo_analysis)
        
        return yolo_analysis, blur_analysis
    
    def _enhance_yolo_for_blur(self, yolo_analysis: YOLOAnalysis, image: np.ndarray):
        """Enhance YOLO analysis with blur-relevant metadata"""
        img_height, img_width = image.shape[:2]
        
        # Assign priority weights and depth layers
        for detection in yolo_analysis.detections:
            # Set subject priority for blur analysis
            detection.subject_priority = SUBJECT_BLUR_PRIORITIES.get(detection.class_name, 0.1)
            
            # Simple depth layering based on size and position
            bbox_area = detection.bbox_width * detection.bbox_height
            center_y = detection.bbox_y + detection.bbox_height / 2
            
            if bbox_area > 0.15:  # Large objects likely foreground
                detection.depth_layer = "foreground"
            elif center_y > 0.7:  # Objects in lower portion often foreground
                detection.depth_layer = "foreground"
            elif bbox_area < 0.05:  # Small objects likely background
                detection.depth_layer = "background"
            else:
                detection.depth_layer = "midground"
        
        # Populate computed properties
        yolo_analysis.priority_subjects = [d for d in yolo_analysis.detections 
                                         if d.subject_priority >= 0.5]
        
        yolo_analysis.subject_regions = [
            detection.get_pixel_bbox(img_width, img_height) 
            for detection in yolo_analysis.priority_subjects
        ]
        
        # Group by depth layers
        for detection in yolo_analysis.detections:
            yolo_analysis.depth_layers[detection.depth_layer].append(detection)
    
    def _analyze_subject_blur(self, image: np.ndarray, 
                            yolo_analysis: YOLOAnalysis) -> SubjectBlurAnalysis:
        """Perform subject-aware blur analysis"""
        img_height, img_width = image.shape[:2]
        
        # Convert to grayscale for blur analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        subject_blur_scores = {}
        subject_details = {}
        
        # Analyze each priority subject
        for i, detection in enumerate(yolo_analysis.priority_subjects):
            x1, y1, x2, y2 = detection.get_pixel_bbox(img_width, img_height)
            
            # Extract subject region with some padding
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_width, x2 + pad)
            y2 = min(img_height, y2 + pad)
            
            subject_region = gray[y1:y2, x1:x2]
            
            if subject_region.size > 0:
                # Calculate multiple blur metrics
                laplacian_var = cv2.Laplacian(subject_region, cv2.CV_64F).var()
                sobel_var = self._calculate_sobel_variance(subject_region)
                
                # Combined sharpness score
                sharpness_score = (laplacian_var + sobel_var) / 2
                
                subject_key = f"{detection.class_name}_{i}"
                subject_blur_scores[subject_key] = sharpness_score
                
                subject_details[subject_key] = {
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'priority': detection.subject_priority,
                    'laplacian_variance': laplacian_var,
                    'sobel_variance': sobel_var,
                    'sharpness_score': sharpness_score,
                    'bbox': (x1, y1, x2, y2),
                    'depth_layer': detection.depth_layer
                }
        
        # Analyze background regions
        background_blur_score = self._analyze_background_blur(gray, yolo_analysis)
        
        # Classify blur type and artistic intent
        classification, confidence, artistic_score = self._classify_blur_type(
            subject_blur_scores, background_blur_score, yolo_analysis
        )
        
        # Determine if meets quality threshold
        priority_sharp = self._are_priority_subjects_sharp(subject_blur_scores, yolo_analysis)
        meets_threshold = self._meets_quality_threshold(
            subject_blur_scores, background_blur_score, classification, priority_sharp
        )
        
        return SubjectBlurAnalysis(
            subject_blur_scores=subject_blur_scores,
            background_blur_score=background_blur_score,
            overall_classification=classification,
            meets_quality_threshold=meets_threshold,
            priority_subject_sharp=priority_sharp,
            blur_type_confidence=confidence,
            subject_details=subject_details,
            background_details={
                'background_sharpness': background_blur_score,
                'background_region_count': len(yolo_analysis.background_regions)
            },
            artistic_intent_score=artistic_score
        )
    
    def _calculate_sobel_variance(self, image_region: np.ndarray) -> float:
        """Calculate Sobel gradient variance for sharpness measurement"""
        sobel_x = cv2.Sobel(image_region, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_region, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        return float(sobel_combined.var())
    
    def _analyze_background_blur(self, gray_image: np.ndarray, 
                               yolo_analysis: YOLOAnalysis) -> float:
        """Analyze blur in background regions (areas not covered by subjects)"""
        img_height, img_width = gray_image.shape
        
        # Create mask of subject regions
        subject_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        for detection in yolo_analysis.detections:
            x1, y1, x2, y2 = detection.get_pixel_bbox(img_width, img_height)
            subject_mask[y1:y2, x1:x2] = 255
        
        # Analyze regions not covered by subjects
        background_mask = cv2.bitwise_not(subject_mask)
        background_pixels = gray_image[background_mask > 0]
        
        if len(background_pixels) > 100:  # Need sufficient pixels for analysis
            # Create temporary image of just background pixels for Laplacian
            background_region = gray_image.copy()
            background_region[subject_mask > 0] = np.mean(background_pixels)
            
            laplacian_var = cv2.Laplacian(background_region, cv2.CV_64F).var()
            return float(laplacian_var)
        else:
            return 0.0  # Not enough background to analyze
    
    def _classify_blur_type(self, subject_scores: Dict[str, float], 
                          background_score: float, 
                          yolo_analysis: YOLOAnalysis) -> Tuple[str, float, float]:
        """Classify the type of blur and artistic intent"""
        if not subject_scores:
            # Fallback classification when no subjects detected
            return self._classify_without_subjects(background_score)
        
        # Calculate average subject sharpness
        weighted_subject_scores = []
        for detection in yolo_analysis.priority_subjects:
            subject_key = f"{detection.class_name}_{yolo_analysis.priority_subjects.index(detection)}"
            if subject_key in subject_scores:
                score = subject_scores[subject_key]
                weight = detection.subject_priority
                weighted_subject_scores.append(score * weight)
        
        if not weighted_subject_scores:
            avg_subject_sharpness = 0.0
        else:
            avg_subject_sharpness = np.mean(weighted_subject_scores)
        
        # Classification logic
        subject_sharp = avg_subject_sharpness > BLUR_THRESHOLDS['subject_sharp_min']
        background_soft = background_score < BLUR_THRESHOLDS['background_soft_max']
        
        confidence = 0.7  # Base confidence
        artistic_score = 0.0
        
        if subject_sharp and background_soft:
            classification = "artistic_shallow_dof"
            artistic_score = 0.8
            confidence = 0.9
        elif subject_sharp and background_score > BLUR_THRESHOLDS['subject_sharp_min']:
            classification = "sharp_throughout"
            confidence = 0.8
        elif avg_subject_sharpness < BLUR_THRESHOLDS['subject_acceptable_min']:
            # Check for motion blur (uniform softness)
            if abs(avg_subject_sharpness - background_score) < 20.0:
                classification = "motion_blur"
                confidence = 0.7
            else:
                classification = "focus_miss"
                confidence = 0.6
        else:
            classification = "mixed_sharpness"
            confidence = 0.5
        
        return classification, confidence, artistic_score
    
    def _classify_without_subjects(self, background_score: float) -> Tuple[str, float, float]:
        """Classify blur when no subjects are detected (fallback method)"""
        # Use global image sharpness to classify
        confidence = 0.6  # Lower confidence without subject context
        artistic_score = 0.0
        
        if background_score > 200.0:  # Very sharp (high variance)
            return "sharp_throughout", confidence, artistic_score
        elif background_score > 80.0:  # Moderately sharp
            return "mixed_sharpness", confidence, artistic_score
        elif background_score > 30.0:  # Somewhat soft
            return "focus_miss", confidence, artistic_score
        else:  # Very soft (< 30.0)
            # Very soft images are likely motion blur
            return "motion_blur", confidence, artistic_score
    
    def _are_priority_subjects_sharp(self, subject_scores: Dict[str, float], 
                                   yolo_analysis: YOLOAnalysis) -> bool:
        """Check if high-priority subjects meet sharpness requirements"""
        if not subject_scores:
            return False
        
        # Check each priority subject
        for detection in yolo_analysis.priority_subjects:
            if detection.subject_priority >= 0.8:  # High priority subjects
                subject_key = f"{detection.class_name}_{yolo_analysis.priority_subjects.index(detection)}"
                if subject_key in subject_scores:
                    score = subject_scores[subject_key]
                    if score < BLUR_THRESHOLDS['subject_sharp_min']:
                        return False
        
        return True
    
    def _meets_quality_threshold(self, subject_scores: Dict[str, float],
                               background_score: float, classification: str,
                               priority_sharp: bool) -> bool:
        """Determine if image meets overall quality threshold"""
        # Artistic bokeh is acceptable even with soft background
        if classification == "artistic_shallow_dof":
            return priority_sharp
        
        # Sharp throughout is always good
        if classification == "sharp_throughout":
            return True
        
        # Motion blur and focus miss are generally poor quality
        if classification in ["motion_blur", "focus_miss"]:
            return False
        
        # Mixed sharpness: depends on priority subjects or background if no subjects
        if subject_scores:
            return priority_sharp
        else:
            # Fallback: use background sharpness when no subjects detected
            return background_score > BLUR_THRESHOLDS['subject_acceptable_min']


class YOLOProcessor:
    """YOLO object detection processor optimized for preview-stage processing"""
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        """
        Initialize YOLO processor
        
        Args:
            model_path: Path to YOLO model (default: yolov8n.pt for speed)
            device: Processing device ("cpu" or "cuda")
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.is_initialized = False
        
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available - object detection disabled")
            return
            
        self._initialize_model()
    
    def _initialize_model(self):
        """Lazy initialization of YOLO model"""
        if self.is_initialized or not YOLO_AVAILABLE:
            return
            
        try:
            logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.is_initialized = True
            logger.info("✅ YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def analyze_preview(self, preview_image: np.ndarray, 
                       confidence_threshold: float = 0.25) -> YOLOAnalysis:
        """
        Analyze preview image for objects and people
        
        Args:
            preview_image: RGB numpy array (800px max dimension)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            YOLOAnalysis with detection results and scores
        """
        import time
        start_time = time.time()
        
        if not self.is_initialized or self.model is None:
            # Return empty analysis if YOLO not available
            return YOLOAnalysis(
                detections=[],
                person_count=0,
                object_score=0.0,
                has_people=False,
                dominant_objects=[],
                processing_time_ms=0.0
            )
        
        try:
            # Ensure image is in correct format (HWC, RGB, uint8)
            if preview_image.dtype != np.uint8:
                preview_image = (preview_image * 255).astype(np.uint8)
            
            # Run YOLO detection
            results = self.model(preview_image, conf=confidence_threshold, verbose=False)
            
            detections = []
            person_count = 0
            object_classes = {}
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection data with proper indexing
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[cls]
                        
                        # Count object classes
                        object_classes[class_name] = object_classes.get(class_name, 0) + 1
                        
                        # Count people
                        if class_name == 'person':
                            person_count += 1
                        
                        # Normalize bounding box coordinates
                        img_height, img_width = preview_image.shape[:2]
                        bbox_x = float(xyxy[0] / img_width)
                        bbox_y = float(xyxy[1] / img_height)
                        bbox_width = float((xyxy[2] - xyxy[0]) / img_width)
                        bbox_height = float((xyxy[3] - xyxy[1]) / img_height)
                        
                        detections.append(YOLODetection(
                            class_name=class_name,
                            confidence=conf,
                            bbox_x=bbox_x,
                            bbox_y=bbox_y,
                            bbox_width=bbox_width,
                            bbox_height=bbox_height
                        ))
            
            # Calculate object interest score
            object_score = self._calculate_object_score(detections, object_classes)
            
            # Get dominant objects (top 3 by count)
            dominant_objects = sorted(object_classes.keys(), 
                                    key=lambda x: object_classes[x], 
                                    reverse=True)[:3]
            
            processing_time = (time.time() - start_time) * 1000
            
            return YOLOAnalysis(
                detections=detections,
                person_count=person_count,
                object_score=object_score,
                has_people=person_count > 0,
                dominant_objects=dominant_objects,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"YOLO analysis failed: {e}")
            return YOLOAnalysis(
                detections=[],
                person_count=0,
                object_score=0.0,
                has_people=False,
                dominant_objects=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_object_score(self, detections: List[YOLODetection], 
                               object_classes: Dict[str, int]) -> float:
        """
        Calculate overall interest score based on detected objects
        
        Args:
            detections: List of all detections
            object_classes: Count of each object class
            
        Returns:
            Score from 0.0 to 1.0 indicating image interest level
        """
        if not detections:
            return 0.0
        
        score = 0.0
        
        # Base score from detection count (more objects = more interesting)
        detection_score = min(len(detections) * 0.1, 0.5)  # Cap at 0.5
        score += detection_score
        
        # Bonus for people (people photos are often more interesting)
        if 'person' in object_classes:
            person_bonus = min(object_classes['person'] * 0.2, 0.4)  # Cap at 0.4
            score += person_bonus
        
        # Bonus for high-confidence detections
        high_conf_count = sum(1 for d in detections if d.confidence > 0.7)
        confidence_bonus = min(high_conf_count * 0.1, 0.3)  # Cap at 0.3
        score += confidence_bonus
        
        # Bonus for diverse object types
        diversity_bonus = min(len(object_classes) * 0.05, 0.15)  # Cap at 0.15
        score += diversity_bonus
        
        # Bonus for interesting object classes
        interesting_classes = {'person', 'car', 'dog', 'cat', 'bird', 'horse', 'boat', 'airplane'}
        interesting_count = sum(1 for cls in object_classes.keys() if cls in interesting_classes)
        interesting_bonus = min(interesting_count * 0.1, 0.2)  # Cap at 0.2
        score += interesting_bonus
        
        return min(score, 1.0)  # Cap total score at 1.0
    
    def should_continue_processing(self, analysis: YOLOAnalysis, 
                                 threshold: float = 0.3) -> bool:
        """
        Determine if image is interesting enough to continue full processing
        
        Args:
            analysis: YOLO analysis result
            threshold: Minimum score to continue processing
            
        Returns:
            True if should continue with full processing
        """
        return analysis.object_score >= threshold
    
    def get_detection_summary(self, analysis: YOLOAnalysis) -> str:
        """Get human-readable summary of detections"""
        if not analysis.detections:
            return "No objects detected"
        
        summary_parts = []
        
        if analysis.has_people:
            if analysis.person_count == 1:
                summary_parts.append("1 person")
            else:
                summary_parts.append(f"{analysis.person_count} people")
        
        # Add other dominant objects (excluding person)
        other_objects = [obj for obj in analysis.dominant_objects if obj != 'person']
        if other_objects:
            summary_parts.extend(other_objects[:2])  # Top 2 non-person objects
        
        total_detections = len(analysis.detections)
        if total_detections > sum(1 for obj in analysis.dominant_objects):
            summary_parts.append(f"{total_detections} total objects")
        
        return ", ".join(summary_parts) if summary_parts else f"{total_detections} objects"


# Global YOLO processor instances (lazy-loaded)
_yolo_processor: Optional[YOLOProcessor] = None
_yolo_blur_processor: Optional[YOLOBlurProcessor] = None


def get_yolo_processor(model_path: str = "yolov8n.pt", device: str = "cpu") -> YOLOProcessor:
    """Get or create global YOLO processor instance"""
    global _yolo_processor
    
    if _yolo_processor is None:
        _yolo_processor = YOLOProcessor(model_path=model_path, device=device)
    
    return _yolo_processor


def get_yolo_blur_processor(model_path: str = "yolov8n.pt", device: str = "cpu") -> YOLOBlurProcessor:
    """Get or create global YOLO blur processor instance"""
    global _yolo_blur_processor
    
    if _yolo_blur_processor is None:
        _yolo_blur_processor = YOLOBlurProcessor(model_path=model_path, device=device)
    
    return _yolo_blur_processor


def analyze_preview_yolo(preview_image: np.ndarray, 
                        config: Dict[str, Any]) -> YOLOAnalysis:
    """
    Convenience function to analyze preview with YOLO using config settings
    
    Args:
        preview_image: RGB numpy array
        config: PhotoSight configuration dict
        
    Returns:
        YOLOAnalysis result
    """
    yolo_config = config.get('ai_curation', {})
    model_path = yolo_config.get('yolo_model', 'yolov8n.pt')
    device = yolo_config.get('device', 'cpu')
    confidence = yolo_config.get('person_detection', {}).get('confidence_threshold', 0.25)
    
    processor = get_yolo_processor(model_path=model_path, device=device)
    return processor.analyze_preview(preview_image, confidence_threshold=confidence)


def analyze_preview_with_blur(preview_image: np.ndarray, 
                            config: Dict[str, Any]) -> Tuple[YOLOAnalysis, SubjectBlurAnalysis]:
    """
    Analyze preview with both YOLO detection and subject-aware blur analysis
    
    Args:
        preview_image: RGB numpy array
        config: PhotoSight configuration dict
        
    Returns:
        Tuple of (YOLOAnalysis, SubjectBlurAnalysis)
    """
    yolo_config = config.get('ai_curation', {})
    model_path = yolo_config.get('yolo_model', 'yolov8n.pt')
    device = yolo_config.get('device', 'cpu')
    confidence = yolo_config.get('person_detection', {}).get('confidence_threshold', 0.25)
    
    processor = YOLOBlurProcessor(model_path=model_path, device=device)
    return processor.analyze_with_blur_context(preview_image, confidence_threshold=confidence)