"""
Face and pose analysis using MediaPipe for photo quality assessment
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """Analyze faces for quality and expression using MediaPipe"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize face analyzer
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Long range model
            min_detection_confidence=min_detection_confidence
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
    def analyze_faces(self, image: np.ndarray) -> Dict:
        """
        Analyze faces in image
        
        Args:
            image: Image array (RGB)
            
        Returns:
            Dictionary with face analysis results
        """
        results = {
            'face_count': 0,
            'faces': [],
            'overall_quality': 0.0,
            'has_eyes_open': True,
            'has_smiles': False,
            'good_expressions': False
        }
        
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image
                
            h, w = image_rgb.shape[:2]
            
            # Detect faces
            detection_results = self.face_detection.process(image_rgb)
            
            if detection_results.detections:
                results['face_count'] = len(detection_results.detections)
                
                # Analyze each face
                for detection in detection_results.detections:
                    face_analysis = self._analyze_single_face(
                        image_rgb, detection, h, w
                    )
                    results['faces'].append(face_analysis)
                    
                # Aggregate results
                if results['faces']:
                    # Check if all faces have eyes open
                    results['has_eyes_open'] = all(
                        face['eyes_open'] for face in results['faces']
                    )
                    
                    # Check if any face is smiling
                    results['has_smiles'] = any(
                        face['smiling'] for face in results['faces']
                    )
                    
                    # Good expressions if most faces look good
                    good_faces = sum(
                        1 for face in results['faces'] 
                        if face['expression_quality'] > 0.6
                    )
                    results['good_expressions'] = good_faces >= len(results['faces']) * 0.7
                    
                    # Overall quality is average of all faces
                    results['overall_quality'] = np.mean([
                        face['quality_score'] for face in results['faces']
                    ])
                    
        except Exception as e:
            logger.error(f"Error in face analysis: {e}")
            
        return results
        
    def _analyze_single_face(self, image: np.ndarray, detection, 
                           height: int, width: int) -> Dict:
        """
        Analyze a single detected face
        
        Args:
            image: Full image
            detection: MediaPipe detection result
            height: Image height
            width: Image width
            
        Returns:
            Analysis of single face
        """
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative bbox to absolute coordinates
        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)
        
        # Ensure bbox is within image bounds
        x = max(0, x)
        y = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        
        # Crop face region with some padding
        padding = int(0.2 * min(w, h))
        x1_pad = max(0, x - padding)
        y1_pad = max(0, y - padding)
        x2_pad = min(width, x2 + padding)
        y2_pad = min(height, y2 + padding)
        
        face_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Run face mesh on cropped face
        mesh_results = self.face_mesh.process(face_crop)
        
        face_data = {
            'bbox': [x, y, x2, y2],
            'confidence': detection.score[0] if detection.score else 0.5,
            'size_ratio': (w * h) / (width * height),
            'eyes_open': True,
            'smiling': False,
            'sharpness': 0.0,
            'expression_quality': 0.0,
            'quality_score': 0.0
        }
        
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
            
            # Analyze eyes
            face_data['eyes_open'] = self._check_eyes_open(landmarks)
            
            # Analyze smile
            face_data['smiling'] = self._check_smile(landmarks)
            
            # Calculate face sharpness
            face_data['sharpness'] = self._calculate_sharpness(face_crop)
            
            # Expression quality
            face_data['expression_quality'] = self._calculate_expression_quality(
                face_data['eyes_open'], 
                face_data['smiling'],
                landmarks
            )
            
        # Calculate overall quality score
        face_data['quality_score'] = self._calculate_face_quality_score(face_data)
        
        return face_data
        
    def _check_eyes_open(self, landmarks) -> bool:
        """Check if eyes are open based on landmarks"""
        # Eye landmarks indices for eye aspect ratio
        # Left eye: 33, 160, 158, 133, 153, 144
        # Right eye: 362, 385, 387, 263, 373, 380
        
        def eye_aspect_ratio(eye_points):
            # Simplified EAR calculation
            vertical_1 = np.linalg.norm(
                np.array([eye_points[1].x, eye_points[1].y]) - 
                np.array([eye_points[5].x, eye_points[5].y])
            )
            vertical_2 = np.linalg.norm(
                np.array([eye_points[2].x, eye_points[2].y]) - 
                np.array([eye_points[4].x, eye_points[4].y])
            )
            horizontal = np.linalg.norm(
                np.array([eye_points[0].x, eye_points[0].y]) - 
                np.array([eye_points[3].x, eye_points[3].y])
            )
            
            if horizontal > 0:
                return (vertical_1 + vertical_2) / (2.0 * horizontal)
            return 0
            
        # Get eye landmarks
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        try:
            left_eye = [landmarks.landmark[i] for i in left_eye_indices]
            right_eye = [landmarks.landmark[i] for i in right_eye_indices]
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # Eyes are considered open if EAR > threshold
            # Lowered threshold and more lenient logic for natural variations
            ear_threshold = 0.12
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Eyes open if average EAR is good, or both individual EARs are reasonable
            return (avg_ear > ear_threshold) or (left_ear > 0.10 and right_ear > 0.10)
            
        except:
            return True  # Default to open if calculation fails
            
    def _check_smile(self, landmarks) -> bool:
        """Check if face is smiling based on mouth landmarks"""
        # Mouth corner indices
        left_corner = landmarks.landmark[61]
        right_corner = landmarks.landmark[291]
        
        # Upper lip center
        upper_lip = landmarks.landmark[13]
        
        # Lower lip center
        lower_lip = landmarks.landmark[14]
        
        # Calculate mouth width
        mouth_width = np.linalg.norm(
            np.array([right_corner.x, right_corner.y]) - 
            np.array([left_corner.x, left_corner.y])
        )
        
        # Calculate mouth height
        mouth_height = np.linalg.norm(
            np.array([lower_lip.x, lower_lip.y]) - 
            np.array([upper_lip.x, upper_lip.y])
        )
        
        # Check if corners are raised (smile)
        corner_height = (left_corner.y + right_corner.y) / 2
        lip_center_height = (upper_lip.y + lower_lip.y) / 2
        
        # Smile if corners are higher than center and mouth is wide
        is_smiling = (
            corner_height < lip_center_height and  # Y increases downward
            mouth_width > mouth_height * 2.5
        )
        
        return is_smiling
        
    def _calculate_sharpness(self, face_image: np.ndarray) -> float:
        """Calculate face sharpness using Laplacian variance"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range
        # Typical sharp faces have variance > 100
        normalized = min(1.0, variance / 200.0)
        return normalized
        
    def _calculate_expression_quality(self, eyes_open: bool, 
                                    smiling: bool, landmarks) -> float:
        """Calculate overall expression quality"""
        score = 0.0
        
        # Eyes open is crucial
        if eyes_open:
            score += 0.5
            
        # Smile is a bonus
        if smiling:
            score += 0.3
            
        # Check for neutral/pleasant expression if not smiling
        if not smiling and eyes_open:
            score += 0.2
            
        return score
        
    def _calculate_face_quality_score(self, face_data: Dict) -> float:
        """Calculate overall face quality score"""
        score = 0.0
        
        # Confidence weight
        score += 0.2 * face_data['confidence']
        
        # Size weight (faces should be reasonably sized)
        if 0.02 < face_data['size_ratio'] < 0.5:
            score += 0.2
        elif 0.01 < face_data['size_ratio'] < 0.6:
            score += 0.1
            
        # Sharpness weight
        score += 0.3 * face_data['sharpness']
        
        # Expression weight
        score += 0.3 * face_data['expression_quality']
        
        return min(1.0, score)