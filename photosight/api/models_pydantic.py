"""
Pydantic models for PhotoSight API with automatic validation
"""

from pydantic import BaseModel, Field, validator, ValidationError
from typing import Optional, List, Dict, Any, Union, Literal
from datetime import datetime
from uuid import UUID
import uuid
from enum import Enum


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


# Sub-models for nested structures

class WhiteBalance(BaseModel):
    """White balance settings"""
    temperature: int = Field(default=5500, ge=2000, le=10000, description="Color temperature in Kelvin")
    tint: int = Field(default=0, ge=-100, le=100, description="Green-magenta tint adjustment")


class ToneCurvePoint(BaseModel):
    """Single point on a tone curve"""
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class LensCorrections(BaseModel):
    """Lens correction settings"""
    enable_profile: bool = Field(default=True, description="Enable lens profile corrections")
    chromatic_aberration: bool = Field(default=True, description="Remove chromatic aberration")
    vignetting: bool = Field(default=True, description="Correct vignetting")
    distortion: bool = Field(default=True, description="Correct lens distortion")


class NoiseReduction(BaseModel):
    """Noise reduction settings"""
    luminance: float = Field(default=0, ge=0, le=100, description="Luminance noise reduction")
    color: float = Field(default=25, ge=0, le=100, description="Color noise reduction")
    detail: float = Field(default=50, ge=0, le=100, description="Detail preservation")


class Sharpening(BaseModel):
    """Sharpening settings"""
    amount: float = Field(default=25, ge=0, le=150, description="Sharpening amount")
    radius: float = Field(default=1.0, ge=0.5, le=3.0, description="Sharpening radius")
    detail: float = Field(default=25, ge=0, le=100, description="Detail level")
    masking: float = Field(default=0, ge=0, le=100, description="Edge masking")


class ResizeSettings(BaseModel):
    """Image resize settings"""
    enabled: bool = Field(default=False)
    width: Optional[int] = Field(None, gt=0, le=10000)
    height: Optional[int] = Field(None, gt=0, le=10000)
    unit: Literal["pixels", "percent"] = Field(default="pixels")
    
    @validator('width', 'height')
    def validate_dimensions(cls, v, values):
        if values.get('enabled') and not v:
            raise ValueError("Width and height required when resize is enabled")
        return v


class ExportSettings(BaseModel):
    """Export settings for processed images"""
    format: ExportFormat = Field(default=ExportFormat.JPEG)
    quality: int = Field(default=90, ge=1, le=100, description="JPEG quality (1-100)")
    color_space: ColorSpace = Field(default=ColorSpace.SRGB)
    resize: Optional[ResizeSettings] = None
    metadata: bool = Field(default=True, description="Include metadata in exported file")
    watermark: Optional[str] = Field(None, description="Watermark text")


# Main models

class ProcessingRecipe(BaseModel):
    """Complete processing recipe with all adjustments"""
    # Basic adjustments
    exposure: float = Field(default=0, ge=-5, le=5, description="Exposure adjustment in stops")
    contrast: float = Field(default=0, ge=-100, le=100)
    highlights: float = Field(default=0, ge=-100, le=100)
    shadows: float = Field(default=0, ge=-100, le=100)
    whites: float = Field(default=0, ge=-100, le=100)
    blacks: float = Field(default=0, ge=-100, le=100)
    
    # Presence adjustments
    clarity: float = Field(default=0, ge=-100, le=100)
    vibrance: float = Field(default=0, ge=-100, le=100)
    saturation: float = Field(default=0, ge=-100, le=100)
    
    # Color and tone
    white_balance: WhiteBalance = Field(default_factory=WhiteBalance)
    tone_curve: Optional[List[ToneCurvePoint]] = Field(None, min_items=2, max_items=20)
    
    # Detail adjustments
    lens_corrections: LensCorrections = Field(default_factory=LensCorrections)
    noise_reduction: NoiseReduction = Field(default_factory=NoiseReduction)
    sharpening: Sharpening = Field(default_factory=Sharpening)
    
    # Export settings
    export_settings: ExportSettings = Field(default_factory=ExportSettings)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class PhotoMetadata(BaseModel):
    """Photo metadata extracted from EXIF"""
    width: int
    height: int
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    iso: Optional[int] = Field(None, ge=25, le=102400)
    aperture: Optional[float] = Field(None, ge=0.7, le=64)
    shutter_speed: Optional[str] = None
    focal_length: Optional[float] = Field(None, ge=1, le=2000)
    date_taken: Optional[datetime] = None
    gps_latitude: Optional[float] = Field(None, ge=-90, le=90)
    gps_longitude: Optional[float] = Field(None, ge=-180, le=180)


class ProcessingResults(BaseModel):
    """Results from photo processing"""
    preview_url: str
    full_res_url: Optional[str] = None
    histogram: Optional[Dict[str, List[int]]] = None
    quality_metrics: Optional[Dict[str, float]] = None
    processing_time_ms: int = Field(ge=0)


class Photo(BaseModel):
    """Complete photo information"""
    id: UUID = Field(default_factory=uuid.uuid4)
    filename: str
    file_path: str
    status: PhotoStatus = PhotoStatus.PENDING
    metadata: Optional[PhotoMetadata] = None
    processing_results: Optional[ProcessingResults] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    project_id: Optional[UUID] = None
    tags: List[str] = Field(default_factory=list)


class Session(BaseModel):
    """Editing session information"""
    session_id: UUID = Field(default_factory=uuid.uuid4)
    photo_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    current_recipe: ProcessingRecipe = Field(default_factory=ProcessingRecipe)
    preview_url: Optional[str] = None
    undo_stack_size: int = Field(default=0, ge=0)
    redo_stack_size: int = Field(default=0, ge=0)
    is_dirty: bool = Field(default=False)


class BatchJobResult(BaseModel):
    """Result for a single item in a batch job"""
    photo_id: UUID
    status: PhotoStatus
    output_path: Optional[str] = None
    error: Optional[str] = None
    processed_at: Optional[datetime] = None


class BatchJob(BaseModel):
    """Batch processing job"""
    job_id: UUID = Field(default_factory=uuid.uuid4)
    status: JobStatus = JobStatus.PENDING
    total_items: int = Field(gt=0)
    processed_items: int = Field(default=0, ge=0)
    failed_items: int = Field(default=0, ge=0)
    progress: float = Field(default=0, ge=0, le=100)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: List[BatchJobResult] = Field(default_factory=list)
    priority: Priority = Priority.NORMAL
    
    @validator('progress')
    def calculate_progress(cls, v, values):
        total = values.get('total_items', 1)
        processed = values.get('processed_items', 0)
        return (processed / total) * 100 if total > 0 else 0


# API Request/Response models

class UploadRequest(BaseModel):
    """File upload request"""
    auto_process: bool = Field(default=False)
    project_id: Optional[UUID] = None
    tags: List[str] = Field(default_factory=list)


class ProcessingRequest(BaseModel):
    """Request to process a photo"""
    recipe: ProcessingRecipe = Field(default_factory=ProcessingRecipe)
    create_session: bool = Field(default=True)
    output_format: Optional[ExportFormat] = None


class BatchProcessingRequest(BaseModel):
    """Request for batch processing"""
    photo_ids: List[UUID] = Field(min_items=1, max_items=1000)
    recipe: ProcessingRecipe = Field(default_factory=ProcessingRecipe)
    output_format: ExportFormat = ExportFormat.JPEG
    priority: Priority = Priority.NORMAL
    
    @validator('photo_ids')
    def validate_unique_ids(cls, v):
        if len(v) != len(set(v)):
            raise ValueError("Duplicate photo IDs not allowed")
        return v


class SessionUpdateRequest(BaseModel):
    """Request to update a session"""
    recipe: Optional[ProcessingRecipe] = None
    action: Optional[Literal["undo", "redo", "reset"]] = None


# API Response models

class APIResponse(BaseModel):
    """Standard API response wrapper"""
    status: Literal["success", "error"]
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: str
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    items: List[Any]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    per_page: int = Field(ge=1, le=100)
    has_next: bool
    has_prev: bool


# WebSocket message models

class WSMessage(BaseModel):
    """Base WebSocket message"""
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PreviewUpdateMessage(WSMessage):
    """Preview update WebSocket message"""
    type: Literal["preview_update"] = "preview_update"
    session_id: UUID
    preview_url: str
    processing_time_ms: int


class ProgressUpdateMessage(WSMessage):
    """Progress update WebSocket message"""
    type: Literal["progress_update"] = "progress_update"
    job_id: UUID
    progress: float = Field(ge=0, le=100)
    processed_items: int
    total_items: int
    status: JobStatus


class ErrorMessage(WSMessage):
    """Error WebSocket message"""
    type: Literal["error"] = "error"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None