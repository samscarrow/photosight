"""
Subject-aware intelligent cropping system for PhotoSight

Detects people and faces in photos and suggests crops that improve composition
using techniques like the rule of thirds.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class CropSuggestion:
    """Represents a suggested crop with its properties"""
    x: int
    y: int
    width: int
    height: int
    score: float
    reasoning: str
    composition_scores: Dict[str, float]


class SmartCropper:
    """Subject-aware intelligent cropping system"""
    
    def __init__(self, 
                 yolo_model: str = 'yolov8n.pt',
                 min_subject_ratio: float = 0.05,
                 max_subject_ratio: float = 0.6,
                 thirds_tolerance: float = 0.1):
        """
        Initialize smart cropper
        
        Args:
            yolo_model: Path to YOLO model for object detection
            min_subject_ratio: Minimum ratio of subject size to image area
            max_subject_ratio: Maximum ratio of subject size to image area
            thirds_tolerance: Tolerance for rule of thirds alignment (0-1)
        """
        self.yolo_model = YOLO(yolo_model)
        self.min_subject_ratio = min_subject_ratio
        self.max_subject_ratio = max_subject_ratio
        self.thirds_tolerance = thirds_tolerance
        
        # Person-related YOLO class IDs
        self.person_classes = {0}  # 'person' class in COCO dataset
        
    def suggest_crops(self, image: np.ndarray, 
                     aspect_ratios: Optional[List[Tuple[int, int]]] = None) -> List[CropSuggestion]:
        """
        Suggest intelligent crops for the image
        
        Args:
            image: RGB image array
            aspect_ratios: List of (width, height) aspect ratios to consider
            
        Returns:
            List of crop suggestions sorted by score
        """
        if aspect_ratios is None:
            # Default aspect ratios
            aspect_ratios = [
                (3, 2),   # Classic 35mm
                (4, 3),   # Traditional
                (16, 9),  # Widescreen
                (1, 1),   # Square
                (4, 5),   # Portrait
                (2, 3),   # Vertical classic
            ]
        
        h, w = image.shape[:2]
        
        # Detect subjects in image
        subjects = self._detect_subjects(image)
        
        if not subjects:
            # No subjects detected, suggest center crops
            return self._suggest_center_crops(w, h, aspect_ratios)
        
        # Detect faces for finer positioning
        faces = self._detect_faces(image)
        
        # Generate crop suggestions
        suggestions = []
        
        for aspect_ratio in aspect_ratios:
            ar_width, ar_height = aspect_ratio
            
            # Try different crop sizes
            for scale in [0.5, 0.67, 0.8, 0.9, 1.0]:
                # Calculate crop dimensions maintaining aspect ratio
                if w / h > ar_width / ar_height:
                    # Image is wider than aspect ratio
                    crop_height = int(h * scale)
                    crop_width = int(crop_height * ar_width / ar_height)
                else:
                    # Image is taller than aspect ratio
                    crop_width = int(w * scale)
                    crop_height = int(crop_width * ar_height / ar_width)
                
                if crop_width > w or crop_height > h:
                    continue
                
                # Generate crops for this size
                crops = self._generate_subject_aware_crops(
                    image, subjects, faces, crop_width, crop_height
                )
                suggestions.extend(crops)
        
        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x.score, reverse=True)
        
        # Filter out overlapping suggestions
        filtered = self._filter_overlapping_crops(suggestions)
        
        return filtered[:10]  # Return top 10 suggestions
    
    def _detect_subjects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect subjects (people) in the image
        
        Args:
            image: RGB image array
            
        Returns:
            List of detected subjects with bounding boxes
        """
        results = self.yolo_model(image, verbose=False)
        subjects = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls)
                    if class_id in self.person_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        subjects.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(box.conf),
                            'class': 'person'
                        })
        
        return subjects
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in the image using OpenCV
        
        Args:
            image: RGB image array
            
        Returns:
            List of detected faces with bounding boxes
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use OpenCV's face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': 1.0,  # OpenCV doesn't provide confidence
                'class': 'face'
            })
        
        return face_list
    
    def _generate_subject_aware_crops(self, image: np.ndarray, 
                                    subjects: List[Dict], 
                                    faces: List[Dict],
                                    crop_width: int, 
                                    crop_height: int) -> List[CropSuggestion]:
        """
        Generate crop suggestions based on detected subjects
        
        Args:
            image: RGB image array
            subjects: List of detected subjects
            faces: List of detected faces
            crop_width: Desired crop width
            crop_height: Desired crop height
            
        Returns:
            List of crop suggestions
        """
        h, w = image.shape[:2]
        suggestions = []
        
        # Combine subjects and faces, prioritizing faces
        all_subjects = faces + subjects
        
        for subject in all_subjects:
            bbox = subject['bbox']
            subj_x1, subj_y1, subj_x2, subj_y2 = bbox
            subj_center_x = (subj_x1 + subj_x2) // 2
            subj_center_y = (subj_y1 + subj_y2) // 2
            subj_width = subj_x2 - subj_x1
            subj_height = subj_y2 - subj_y1
            
            # Check if subject is too small or too large
            subj_area_ratio = (subj_width * subj_height) / (w * h)
            if subj_area_ratio < self.min_subject_ratio or subj_area_ratio > self.max_subject_ratio:
                continue
            
            # Try different positions based on rule of thirds
            thirds_positions = [
                (crop_width // 3, crop_height // 3),        # Top-left third
                (2 * crop_width // 3, crop_height // 3),    # Top-right third
                (crop_width // 3, 2 * crop_height // 3),    # Bottom-left third
                (2 * crop_width // 3, 2 * crop_height // 3), # Bottom-right third
                (crop_width // 2, crop_height // 2),         # Center
            ]
            
            for third_x, third_y in thirds_positions:
                # Calculate crop position to place subject at thirds intersection
                crop_x = subj_center_x - third_x
                crop_y = subj_center_y - third_y
                
                # Ensure crop is within image bounds
                crop_x = max(0, min(w - crop_width, crop_x))
                crop_y = max(0, min(h - crop_height, crop_y))
                
                # Check if subject is fully contained in crop
                if (crop_x <= subj_x1 and crop_y <= subj_y1 and 
                    crop_x + crop_width >= subj_x2 and crop_y + crop_height >= subj_y2):
                    
                    # Calculate composition scores
                    crop_region = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
                    scores = self._calculate_composition_scores(
                        crop_region, 
                        (subj_x1 - crop_x, subj_y1 - crop_y, subj_x2 - crop_x, subj_y2 - crop_y),
                        subject['class']
                    )
                    
                    # Calculate overall score
                    overall_score = self._calculate_overall_score(scores, subject['confidence'])
                    
                    # Determine reasoning
                    reasoning = self._generate_reasoning(scores, subject['class'], third_x, third_y, crop_width, crop_height)
                    
                    suggestions.append(CropSuggestion(
                        x=crop_x,
                        y=crop_y,
                        width=crop_width,
                        height=crop_height,
                        score=overall_score,
                        reasoning=reasoning,
                        composition_scores=scores
                    ))
        
        return suggestions
    
    def _calculate_composition_scores(self, crop: np.ndarray, 
                                    subject_bbox: Tuple[int, int, int, int],
                                    subject_class: str) -> Dict[str, float]:
        """
        Calculate various composition scores for a crop
        
        Args:
            crop: Cropped image region
            subject_bbox: Subject bounding box within crop (x1, y1, x2, y2)
            subject_class: Type of subject ('face' or 'person')
            
        Returns:
            Dictionary of composition scores
        """
        h, w = crop.shape[:2]
        subj_x1, subj_y1, subj_x2, subj_y2 = subject_bbox
        subj_center_x = (subj_x1 + subj_x2) / 2
        subj_center_y = (subj_y1 + subj_y2) / 2
        
        scores = {}
        
        # Rule of thirds score
        thirds_x = [w / 3, 2 * w / 3]
        thirds_y = [h / 3, 2 * h / 3]
        
        # Find closest thirds intersection
        min_dist = float('inf')
        for tx in thirds_x:
            for ty in thirds_y:
                dist = np.sqrt((subj_center_x - tx)**2 + (subj_center_y - ty)**2)
                min_dist = min(min_dist, dist)
        
        # Normalize distance to score (0-1)
        max_dist = np.sqrt((w/2)**2 + (h/2)**2)
        scores['rule_of_thirds'] = 1.0 - (min_dist / max_dist)
        
        # Subject size score (not too small, not too large)
        subj_area = (subj_x2 - subj_x1) * (subj_y2 - subj_y1)
        crop_area = w * h
        size_ratio = subj_area / crop_area
        
        if subject_class == 'face':
            # Faces should be 5-25% of frame
            optimal_ratio = 0.15
            tolerance = 0.1
        else:
            # Full bodies should be 15-50% of frame
            optimal_ratio = 0.3
            tolerance = 0.2
        
        scores['subject_size'] = np.exp(-((size_ratio - optimal_ratio) / tolerance)**2)
        
        # Balance score (subject shouldn't be too close to edges)
        edge_distances = [
            subj_x1,  # Left edge
            subj_y1,  # Top edge
            w - subj_x2,  # Right edge
            h - subj_y2   # Bottom edge
        ]
        min_edge_dist = min(edge_distances)
        scores['balance'] = min(1.0, min_edge_dist / (min(w, h) * 0.1))
        
        # Golden ratio score (optional, for aesthetic appeal)
        golden_ratio = 1.618
        aspect_ratio = w / h
        scores['golden_ratio'] = np.exp(-((aspect_ratio - golden_ratio) / 0.5)**2)
        
        # Leading space (for subjects looking/moving in a direction)
        # Simple heuristic: more space in front than behind
        if subj_center_x < w / 2:
            # Subject on left, should have space on right
            front_space = w - subj_x2
            back_space = subj_x1
        else:
            # Subject on right, should have space on left
            front_space = subj_x1
            back_space = w - subj_x2
        
        if front_space + back_space > 0:
            scores['leading_space'] = front_space / (front_space + back_space)
        else:
            scores['leading_space'] = 0.5
        
        return scores
    
    def _calculate_overall_score(self, composition_scores: Dict[str, float], 
                               subject_confidence: float) -> float:
        """
        Calculate overall crop score from individual components
        
        Args:
            composition_scores: Dictionary of composition scores
            subject_confidence: Detection confidence of the subject
            
        Returns:
            Overall score (0-1)
        """
        # Weights for different aspects
        weights = {
            'rule_of_thirds': 0.3,
            'subject_size': 0.25,
            'balance': 0.2,
            'golden_ratio': 0.1,
            'leading_space': 0.15
        }
        
        # Calculate weighted average
        total_score = 0
        total_weight = 0
        
        for aspect, weight in weights.items():
            if aspect in composition_scores:
                total_score += composition_scores[aspect] * weight
                total_weight += weight
        
        if total_weight > 0:
            avg_score = total_score / total_weight
        else:
            avg_score = 0.5
        
        # Factor in subject detection confidence
        final_score = avg_score * (0.7 + 0.3 * subject_confidence)
        
        return final_score
    
    def _generate_reasoning(self, scores: Dict[str, float], subject_class: str,
                          third_x: int, third_y: int, crop_width: int, crop_height: int) -> str:
        """
        Generate human-readable reasoning for the crop suggestion
        
        Args:
            scores: Composition scores
            subject_class: Type of subject
            third_x, third_y: Thirds position used
            crop_width, crop_height: Crop dimensions
            
        Returns:
            Reasoning string
        """
        reasons = []
        
        # Rule of thirds
        if scores.get('rule_of_thirds', 0) > 0.8:
            if third_x < crop_width / 2:
                horiz = "left"
            elif third_x > crop_width / 2:
                horiz = "right"
            else:
                horiz = "center"
                
            if third_y < crop_height / 2:
                vert = "upper"
            elif third_y > crop_height / 2:
                vert = "lower"
            else:
                vert = "middle"
                
            reasons.append(f"{subject_class.capitalize()} positioned at {vert}-{horiz} third")
        
        # Subject size
        if scores.get('subject_size', 0) > 0.8:
            reasons.append(f"Optimal {subject_class} prominence in frame")
        
        # Balance
        if scores.get('balance', 0) > 0.8:
            reasons.append("Well-balanced composition")
        
        # Leading space
        if scores.get('leading_space', 0) > 0.7:
            reasons.append("Good leading space")
        
        # Aspect ratio
        ar = crop_width / crop_height
        if abs(ar - 1.618) < 0.1:
            reasons.append("Golden ratio aspect")
        elif abs(ar - 1.5) < 0.1:
            reasons.append("Classic 3:2 aspect")
        elif abs(ar - 1.33) < 0.1:
            reasons.append("Traditional 4:3 aspect")
        elif abs(ar - 1.78) < 0.1:
            reasons.append("Cinematic 16:9 aspect")
        
        return "; ".join(reasons) if reasons else "Standard crop"
    
    def _suggest_center_crops(self, width: int, height: int, 
                            aspect_ratios: List[Tuple[int, int]]) -> List[CropSuggestion]:
        """
        Suggest center crops when no subjects are detected
        
        Args:
            width: Image width
            height: Image height
            aspect_ratios: List of aspect ratios to try
            
        Returns:
            List of center crop suggestions
        """
        suggestions = []
        
        for ar_width, ar_height in aspect_ratios:
            # Calculate crop size
            if width / height > ar_width / ar_height:
                crop_height = height
                crop_width = int(height * ar_width / ar_height)
            else:
                crop_width = width
                crop_height = int(width * ar_height / ar_width)
            
            if crop_width > width or crop_height > height:
                continue
            
            # Center the crop
            crop_x = (width - crop_width) // 2
            crop_y = (height - crop_height) // 2
            
            suggestions.append(CropSuggestion(
                x=crop_x,
                y=crop_y,
                width=crop_width,
                height=crop_height,
                score=0.5,  # Neutral score for center crops
                reasoning=f"Center crop with {ar_width}:{ar_height} aspect ratio",
                composition_scores={'center_crop': 1.0}
            ))
        
        return suggestions
    
    def _filter_overlapping_crops(self, suggestions: List[CropSuggestion], 
                                iou_threshold: float = 0.8) -> List[CropSuggestion]:
        """
        Filter out highly overlapping crop suggestions
        
        Args:
            suggestions: List of crop suggestions sorted by score
            iou_threshold: Maximum IoU to consider crops different
            
        Returns:
            Filtered list of suggestions
        """
        if not suggestions:
            return []
        
        filtered = [suggestions[0]]  # Keep the highest scoring one
        
        for suggestion in suggestions[1:]:
            # Check overlap with all kept suggestions
            keep = True
            for kept in filtered:
                iou = self._calculate_iou(
                    (suggestion.x, suggestion.y, suggestion.width, suggestion.height),
                    (kept.x, kept.y, kept.width, kept.height)
                )
                if iou > iou_threshold:
                    keep = False
                    break
            
            if keep:
                filtered.append(suggestion)
        
        return filtered
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union of two boxes
        
        Args:
            box1: (x, y, width, height)
            box2: (x, y, width, height)
            
        Returns:
            IoU value (0-1)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def apply_crop(self, image: np.ndarray, crop: CropSuggestion) -> np.ndarray:
        """
        Apply a crop suggestion to an image
        
        Args:
            image: Original image
            crop: Crop suggestion to apply
            
        Returns:
            Cropped image
        """
        return image[crop.y:crop.y + crop.height, crop.x:crop.x + crop.width]
    
    def visualize_suggestions(self, image: np.ndarray, 
                            suggestions: List[CropSuggestion],
                            max_show: int = 5) -> np.ndarray:
        """
        Visualize crop suggestions on the image
        
        Args:
            image: Original image
            suggestions: List of crop suggestions
            max_show: Maximum number of suggestions to show
            
        Returns:
            Image with crop suggestions drawn
        """
        vis_image = image.copy()
        h, w = image.shape[:2]
        
        # Define colors for different suggestions (best to worst)
        colors = [
            (0, 255, 0),    # Green
            (0, 200, 255),  # Yellow
            (0, 165, 255),  # Orange  
            (0, 0, 255),    # Red
            (255, 0, 255),  # Magenta
        ]
        
        for i, suggestion in enumerate(suggestions[:max_show]):
            color = colors[min(i, len(colors) - 1)]
            
            # Draw rectangle
            cv2.rectangle(
                vis_image,
                (suggestion.x, suggestion.y),
                (suggestion.x + suggestion.width, suggestion.y + suggestion.height),
                color, 2
            )
            
            # Draw thirds grid inside the crop
            third_w = suggestion.width // 3
            third_h = suggestion.height // 3
            
            # Vertical lines
            for j in range(1, 3):
                cv2.line(
                    vis_image,
                    (suggestion.x + j * third_w, suggestion.y),
                    (suggestion.x + j * third_w, suggestion.y + suggestion.height),
                    color, 1
                )
            
            # Horizontal lines
            for j in range(1, 3):
                cv2.line(
                    vis_image,
                    (suggestion.x, suggestion.y + j * third_h),
                    (suggestion.x + suggestion.width, suggestion.y + j * third_h),
                    color, 1
                )
            
            # Add score text
            text = f"#{i+1}: {suggestion.score:.2f}"
            cv2.putText(
                vis_image, text,
                (suggestion.x + 5, suggestion.y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        return vis_image