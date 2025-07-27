"""
Unified Pydantic models for PhotoSight API with comprehensive validation

This module replaces both the dataclass-based models.py and models_pydantic.py
to provide a single source of truth for API data structures with robust validation.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any, Union, Literal, Tuple
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum


# Base Configuration
class PhotoSightBaseModel(BaseModel):
    """Base model with common configuration for PhotoSight models"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    )


# ========== Enums ==========

class APIStatus(str, Enum):
    """API response status codes."""
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"
    PARTIAL = "partial"


class SessionState(str, Enum):
    """Editing session states."""
    CREATED = "created"
    ACTIVE = "active"
    PROCESSING = "processing"
    SAVING = "saving"
    CLOSED = "closed"
    ERROR = "error"


class PhotoStatus(str, Enum):
    """Photo processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, Enum):
    """Batch job status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportFormat(str, Enum):
    """Export file formats"""
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    DNG = "dng"


class ColorSpace(str, Enum):
    """Color space options"""
    SRGB = "sRGB"
    ADOBE_RGB = "AdobeRGB"
    PROPHOTO_RGB = "ProPhotoRGB"


class Priority(str, Enum):
    """Processing priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


# ========== Base Models ==========

class APIResponse(PhotoSightBaseModel):
    """Standard API response wrapper."""
    status: APIStatus
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ErrorResponse(APIResponse):
    """Error response with additional context."""
    status: APIStatus = APIStatus.ERROR
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class PaginatedResponse(PhotoSightBaseModel):
    """Paginated response wrapper"""
    items: List[Any]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    per_page: int = Field(ge=1, le=100)
    has_next: bool
    has_prev: bool


# ========== Processing Models ==========

class ProcessingRecipe(PhotoSightBaseModel):
    """Complete processing recipe with validation"""
    
    # Basic adjustments
    exposure: float = Field(default=0.0, ge=-5.0, le=5.0, description="Exposure adjustment in stops")
    contrast: float = Field(default=0.0, ge=-100.0, le=100.0)
    highlights: float = Field(default=0.0, ge=-100.0, le=100.0)
    shadows: float = Field(default=0.0, ge=-100.0, le=100.0)
    whites: float = Field(default=0.0, ge=-100.0, le=100.0)
    blacks: float = Field(default=0.0, ge=-100.0, le=100.0)
    
    # Color adjustments
    vibrance: float = Field(default=0.0, ge=-100.0, le=100.0)
    saturation: float = Field(default=0.0, ge=-100.0, le=100.0)
    temperature: float = Field(default=0.0, ge=-10000.0, le=10000.0)
    tint: float = Field(default=0.0, ge=-100.0, le=100.0)
    
    # Presence adjustments
    clarity: float = Field(default=0.0, ge=-100.0, le=100.0)
    texture: float = Field(default=0.0, ge=-100.0, le=100.0)
    vignette_amount: float = Field(default=0.0, ge=-100.0, le=100.0)
    
    # Geometry (optional crop)
    crop: Optional[Dict[str, float]] = None
    
    # Advanced options
    auto_tone: bool = Field(default=False)
    
    @field_validator('crop')
    @classmethod
    def validate_crop(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if v is None:
            return v
        required_keys = {'x', 'y', 'width', 'height'}
        if not all(k in v for k in required_keys):
            raise ValueError(f"Crop must contain keys: {required_keys}")
        if not all(0 <= v[k] <= 1 for k in required_keys):
            raise ValueError("Crop values must be between 0 and 1")
        if v['width'] <= 0 or v['height'] <= 0:
            raise ValueError("Crop width and height must be positive")
        return v


class ProcessingRequest(BaseModel):
    """Request to process a single photo"""
    photo_id: UUID
    recipe: ProcessingRecipe
    priority: Priority = Priority.NORMAL
    webhook_url: Optional[str] = None


class BatchProcessingRequest(BaseModel):
    """Request to process multiple photos"""
    photo_ids: List[UUID] = Field(..., min_length=1, max_length=1000)
    recipe: ProcessingRecipe
    priority: Priority = Priority.NORMAL
    job_name: Optional[str] = None
    webhook_url: Optional[str] = None
    
    @field_validator('photo_ids')
    @classmethod
    def validate_unique_ids(cls, v: List[UUID]) -> List[UUID]:
        if len(set(v)) != len(v):
            raise ValueError("photo_ids must be unique")
        return v


# ========== Session Models ==========

class SessionInfo(BaseModel):
    """Editing session information"""
    session_id: UUID = Field(default_factory=uuid4)
    photo_id: UUID
    state: SessionState = SessionState.CREATED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    current_recipe: Optional[ProcessingRecipe] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PreviewRequest(BaseModel):
    """Request for preview generation"""
    recipe: ProcessingRecipe
    size: Literal["small", "medium", "large"] = "medium"
    quality: int = Field(default=85, ge=50, le=100)


class PreviewUpdate(BaseModel):
    """Preview update notification"""
    session_id: UUID
    preview_url: str
    processing_time: float
    recipe: ProcessingRecipe
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ========== Progress Models ==========

class ProcessingProgress(BaseModel):
    """Processing progress update"""
    session_id: Optional[UUID] = None
    job_id: Optional[UUID] = None
    photo_id: Optional[UUID] = None
    progress: float = Field(..., ge=0.0, le=100.0)
    stage: str
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchJobStatus(BaseModel):
    """Batch job status"""
    job_id: UUID
    status: JobStatus
    total_photos: int
    processed_photos: int
    failed_photos: int
    progress: float = Field(..., ge=0.0, le=100.0)
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


# ========== Photo Models ==========

class PhotoMetadata(BaseModel):
    """Photo metadata"""
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens: Optional[str] = None
    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    shutter_speed: Optional[str] = None
    iso: Optional[int] = None
    taken_at: Optional[datetime] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None


class PhotoInfo(BaseModel):
    """Photo information"""
    photo_id: UUID
    filename: str
    file_size: int
    width: int
    height: int
    format: str
    status: PhotoStatus
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Optional[PhotoMetadata] = None
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    processing_recipe: Optional[ProcessingRecipe] = None


# ========== Export Models ==========

class ExportSettings(BaseModel):
    """Export settings"""
    format: ExportFormat = ExportFormat.JPEG
    quality: int = Field(default=90, ge=50, le=100)
    color_space: ColorSpace = ColorSpace.SRGB
    resize: Optional[Dict[str, int]] = None
    output_sharpening: bool = True
    embed_metadata: bool = True
    
    @field_validator('resize')
    @classmethod
    def validate_resize(cls, v: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
        if v is None:
            return v
        if 'width' in v and v['width'] <= 0:
            raise ValueError("Width must be positive")
        if 'height' in v and v['height'] <= 0:
            raise ValueError("Height must be positive")
        if not ('width' in v or 'height' in v):
            raise ValueError("Resize must specify at least width or height")
        return v


class ExportRequest(BaseModel):
    """Export request"""
    photo_id: UUID
    recipe: ProcessingRecipe
    settings: ExportSettings
    destination_path: Optional[str] = None


# ========== WebSocket Models ==========

class WSMessage(BaseModel):
    """WebSocket message"""
    type: str
    event: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSSubscribeRequest(BaseModel):
    """WebSocket subscription request"""
    session_id: Optional[UUID] = None
    job_id: Optional[UUID] = None
    events: List[str] = Field(default_factory=lambda: ["preview_update", "progress", "complete", "error"])


# Export the most commonly used models for easy import
__all__ = [
    'APIStatus', 'SessionState', 'PhotoStatus', 'JobStatus', 
    'ExportFormat', 'ColorSpace', 'Priority',
    'APIResponse', 'ErrorResponse',
    'ProcessingRecipe', 'ProcessingRequest', 'BatchProcessingRequest',
    'SessionInfo', 'PreviewRequest', 'PreviewUpdate',
    'ProcessingProgress', 'BatchJobStatus',
    'PhotoMetadata', 'PhotoInfo',
    'ExportSettings', 'ExportRequest',
    'WSMessage', 'WSSubscribeRequest'
]