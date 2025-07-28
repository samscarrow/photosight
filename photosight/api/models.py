"""
API Data Models and Schemas

Defines request/response models for the PhotoSight API.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json


class APIStatus(Enum):
    """API response status codes."""
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"
    PARTIAL = "partial"


class SessionState(Enum):
    """Editing session states."""
    CREATED = "created"
    ACTIVE = "active"
    PROCESSING = "processing"
    SAVING = "saving"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class APIResponse:
    """Standard API response wrapper."""
    status: APIStatus
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'status': self.status.value,
            'data': self.data,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'request_id': self.request_id
        }


@dataclass
class ErrorResponse(APIResponse):
    """Error response with additional context."""
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        self.status = APIStatus.ERROR
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'error_code': self.error_code,
            'error_details': self.error_details
        })
        return result


@dataclass
class SessionInfo:
    """Editing session information."""
    session_id: str
    state: SessionState
    created_at: datetime
    updated_at: datetime
    image_path: Optional[str] = None
    recipe_id: Optional[str] = None
    preview_level: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'image_path': self.image_path,
            'recipe_id': self.recipe_id,
            'preview_level': self.preview_level,
            'metadata': self.metadata
        }


@dataclass 
class ProcessingRequest:
    """Request for image processing."""
    session_id: str
    recipe: Dict[str, Any]
    preview_level: str = "medium"
    progressive: bool = True
    priority: str = "normal"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingRequest':
        return cls(
            session_id=data['session_id'],
            recipe=data['recipe'],
            preview_level=data.get('preview_level', 'medium'),
            progressive=data.get('progressive', True),
            priority=data.get('priority', 'normal')
        )


@dataclass
class BatchProcessingRequest:
    """Request for batch processing."""
    images: List[str]
    recipe: Dict[str, Any]
    output_format: str = "jpeg"
    output_quality: int = 90
    priority: str = "normal"
    webhook_url: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchProcessingRequest':
        return cls(
            images=data['images'],
            recipe=data['recipe'],
            output_format=data.get('output_format', 'jpeg'),
            output_quality=data.get('output_quality', 90),
            priority=data.get('priority', 'normal'),
            webhook_url=data.get('webhook_url')
        )


@dataclass
class BatchJobStatus:
    """Batch processing job status."""
    job_id: str
    state: str  # queued, processing, completed, failed
    total_images: int
    processed_images: int
    failed_images: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def progress(self) -> float:
        """Calculate progress percentage."""
        if self.total_images == 0:
            return 0.0
        return (self.processed_images + self.failed_images) / self.total_images * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'state': self.state,
            'progress': self.progress,
            'total_images': self.total_images,
            'processed_images': self.processed_images,
            'failed_images': self.failed_images,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'results': self.results,
            'errors': self.errors
        }


@dataclass
class PreviewUpdate:
    """Real-time preview update message."""
    session_id: str
    preview_level: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    image_data: Optional[str] = None  # Base64 encoded
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'preview_update',
            'session_id': self.session_id,
            'preview_level': self.preview_level,
            'timestamp': self.timestamp.isoformat(),
            'image_data': self.image_data,
            'metadata': self.metadata
        }


@dataclass
class ProcessingProgress:
    """Processing progress update."""
    session_id: str
    stage: str  # loading, processing, saving
    progress: float  # 0-100
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'processing_progress',
            'session_id': self.session_id,
            'stage': self.stage,
            'progress': self.progress,
            'message': self.message,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SystemStats:
    """System performance statistics."""
    active_sessions: int
    queued_tasks: int
    processing_tasks: int
    memory_usage_mb: float
    cache_hit_rate: float
    average_processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'active_sessions': self.active_sessions,
            'queued_tasks': self.queued_tasks,
            'processing_tasks': self.processing_tasks,
            'memory_usage_mb': self.memory_usage_mb,
            'cache_hit_rate': self.cache_hit_rate,
            'average_processing_time_ms': self.average_processing_time_ms,
            'timestamp': self.timestamp.isoformat()
        }


class RecipeValidator:
    """Validates processing recipe structure."""
    
    REQUIRED_FIELDS = []
    OPTIONAL_FIELDS = [
        'exposure_adjustment', 'contrast', 'highlights', 'shadows',
        'whites', 'blacks', 'vibrance', 'saturation', 'temperature_adjustment',
        'tint_adjustment', 'clarity', 'texture', 'vignette_amount',
        'noise_reduction', 'adjustment_layers', 'color_grading_preset'
    ]
    
    @classmethod
    def validate(cls, recipe: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a processing recipe.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(recipe, dict):
            return False, "Recipe must be a dictionary"
        
        # Check for unknown fields
        known_fields = set(cls.REQUIRED_FIELDS + cls.OPTIONAL_FIELDS)
        unknown_fields = set(recipe.keys()) - known_fields
        if unknown_fields and 'metadata' not in unknown_fields:
            # Allow metadata field for custom data
            unknown_fields.discard('metadata')
            if unknown_fields:
                return False, f"Unknown fields: {', '.join(unknown_fields)}"
        
        # Validate numeric ranges
        numeric_validations = {
            'exposure_adjustment': (-5.0, 5.0),
            'contrast': (-100, 100),
            'highlights': (-100, 100),
            'shadows': (-100, 100),
            'whites': (-100, 100),
            'blacks': (-100, 100),
            'vibrance': (-100, 100),
            'saturation': (-100, 100),
            'clarity': (-100, 100),
            'texture': (-100, 100),
            'vignette_amount': (-100, 100),
            'temperature_adjustment': (-100, 100),
            'tint_adjustment': (-100, 100)
        }
        
        for field, (min_val, max_val) in numeric_validations.items():
            if field in recipe:
                value = recipe[field]
                if not isinstance(value, (int, float)):
                    return False, f"{field} must be numeric"
                if value < min_val or value > max_val:
                    return False, f"{field} must be between {min_val} and {max_val}"
        
        # Validate noise reduction if present
        if 'noise_reduction' in recipe:
            nr = recipe['noise_reduction']
            if not isinstance(nr, dict):
                return False, "noise_reduction must be a dictionary"
            
            nr_fields = ['luminance_amount', 'chrominance_amount', 'detail_preservation']
            for field in nr_fields:
                if field in nr:
                    value = nr[field]
                    if not isinstance(value, (int, float)) or value < 0 or value > 100:
                        return False, f"noise_reduction.{field} must be between 0 and 100"
        
        # Validate adjustment layers if present
        if 'adjustment_layers' in recipe:
            layers = recipe['adjustment_layers']
            if not isinstance(layers, list):
                return False, "adjustment_layers must be a list"
            
            for i, layer in enumerate(layers):
                if not isinstance(layer, dict):
                    return False, f"adjustment_layers[{i}] must be a dictionary"
                if 'name' not in layer:
                    return False, f"adjustment_layers[{i}] must have a name"
                if 'adjustments' not in layer:
                    return False, f"adjustment_layers[{i}] must have adjustments"
        
        return True, None


# WebSocket message types
WS_MESSAGE_TYPES = {
    'SUBSCRIBE': 'subscribe',
    'UNSUBSCRIBE': 'unsubscribe',
    'PREVIEW_UPDATE': 'preview_update',
    'PROGRESS_UPDATE': 'progress_update',
    'ERROR': 'error',
    'HEARTBEAT': 'heartbeat',
    'SESSION_STATE': 'session_state'
}