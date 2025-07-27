"""
Database models for PhotoSight.

Defines the SQLAlchemy ORM models for photos, analysis results,
processing recipes, and project management.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, 
    ForeignKey, Index, BigInteger, Enum, Table
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class ProcessingStatus(enum.Enum):
    """Photo processing status."""
    PENDING = "pending"
    PROCESSED = "processed"
    REJECTED = "rejected"
    ERROR = "error"


class FileStatus(enum.Enum):
    """File sync status for hybrid local/cloud storage."""
    LOCAL_ONLY = "local_only"           # File exists only on local machine
    CLOUD_ONLY = "cloud_only"           # File exists only in cloud (metadata only)
    SYNCED = "synced"                   # File exists both locally and cloud metadata matches
    LOCAL_MODIFIED = "local_modified"   # Local file changed after cloud sync
    CLOUD_MODIFIED = "cloud_modified"   # Cloud metadata changed after local sync
    MISSING = "missing"                 # File missing from expected location
    CONFLICT = "conflict"               # Both local and cloud modified


class ProjectStatus(enum.Enum):
    """Project status."""
    PLANNING = "Planning"
    ACTIVE = "Active"
    ON_HOLD = "On Hold"
    COMPLETED = "Completed"
    ARCHIVED = "Archived"


class ProjectPhase(enum.Enum):
    """Project workflow phase."""
    CAPTURE = "Capture"
    IMPORT = "Import"
    CULL = "Cull"
    EDIT = "Edit"
    REVIEW = "Review"
    EXPORT = "Export"
    DELIVER = "Deliver"


class TaskStatus(enum.Enum):
    """Task status."""
    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    REVIEW = "Code Review"
    COMPLETED = "Done"
    BLOCKED = "Blocked"


class TaskPriority(enum.Enum):
    """Task priority levels."""
    CRITICAL = "P0-Critical"
    HIGH = "P1-High"
    MEDIUM = "P2-Medium"
    LOW = "P3-Low"


# Association table for many-to-many relationship between photos and tasks
photo_tasks = Table(
    'photo_tasks',
    Base.metadata,
    Column('photo_id', Integer, ForeignKey('photos.id'), primary_key=True),
    Column('task_id', Integer, ForeignKey('tasks.id'), primary_key=True)
)

# Association model for photos and album tags with extra metadata
class PhotoAlbumTag(Base):
    """Association between photos and album tags with metadata."""
    __tablename__ = 'photo_album_tags'
    
    photo_id = Column(Integer, ForeignKey('photos.id'), primary_key=True)
    album_tag_id = Column(Integer, ForeignKey('album_tags.id'), primary_key=True)
    added_at = Column(DateTime, default=func.now())
    added_by = Column(String(100), default='system')
    sort_order = Column(Integer, default=0)
    association_metadata = Column(JSONB, default={})
    
    # Relationships
    photo = relationship("Photo", back_populates="album_associations")
    album_tag = relationship("AlbumTag", back_populates="photo_associations")

# Association table for many-to-many relationship between projects and photos
project_photos = Table(
    'project_photos',
    Base.metadata,
    Column('project_id', Integer, ForeignKey('projects.id'), primary_key=True),
    Column('photo_id', Integer, ForeignKey('photos.id'), primary_key=True),
    Column('added_at', DateTime, default=func.now()),
    Column('notes', Text)
)

# Association table for many-to-many relationship between photos and keywords
photo_keywords = Table(
    'photo_keywords',
    Base.metadata,
    Column('photo_id', Integer, ForeignKey('photos.id'), primary_key=True),
    Column('keyword_id', Integer, ForeignKey('keywords.id'), primary_key=True),
    Column('added_at', DateTime, default=func.now()),
    Column('source', String(50))  # 'manual', 'ai', 'import'
)

# Association table for many-to-many relationship between photos and collections
photo_collections = Table(
    'photo_collections',
    Base.metadata,
    Column('photo_id', Integer, ForeignKey('photos.id'), primary_key=True),
    Column('collection_id', Integer, ForeignKey('collections.id'), primary_key=True),
    Column('added_at', DateTime, default=func.now()),
    Column('sort_order', Integer, default=0)
)


class ProjectPhoto(Base):
    """Association between projects and photos with metadata."""
    __tablename__ = 'project_photos'
    __table_args__ = {'extend_existing': True}
    
    project_id = Column(Integer, ForeignKey('projects.id'), primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id'), primary_key=True)
    added_at = Column(DateTime, default=func.now())
    notes = Column(Text)
    
    # Relationships
    project = relationship("Project")
    photo = relationship("Photo")


class Project(Base):
    """Photography project (shoot, event, assignment)."""
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True, index=True)
    description = Column(Text)
    client_name = Column(String(200), index=True)
    project_type = Column(String(50))  # wedding, portrait, commercial, landscape, etc.
    
    # Status and workflow
    status = Column(Enum(ProjectStatus), default=ProjectStatus.PLANNING, index=True)
    phase = Column(Enum(ProjectPhase), default=ProjectPhase.CAPTURE, index=True)
    priority = Column(Enum(TaskPriority), default=TaskPriority.MEDIUM, index=True)
    
    # Dates
    shoot_date = Column(DateTime)
    due_date = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime)
    
    # Project details
    budget = Column(Float)
    location = Column(String(500))
    notes = Column(Text)
    
    # Deliverables tracking
    expected_photos = Column(Integer)
    delivered_photos = Column(Integer, default=0)
    
    # Recipe association
    default_recipe_id = Column(Integer, ForeignKey('processing_recipes.id'), nullable=True)
    
    # Custom metadata
    meta_data = Column(JSONB, default={})
    
    # Relationships
    photos = relationship("Photo", secondary=project_photos, back_populates="projects", overlaps="photo,project")
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")
    default_recipe = relationship("ProcessingRecipe", foreign_keys=[default_recipe_id])
    
    # Indexes
    __table_args__ = (
        Index('idx_project_status_due', 'status', 'due_date'),
        Index('idx_project_phase_priority', 'phase', 'priority'),
        Index('idx_project_client', 'client_name'),
    )


class Task(Base):
    """Task within a project."""
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False, index=True)
    parent_task_id = Column(Integer, ForeignKey('tasks.id'), index=True)  # For subtasks
    
    name = Column(String(200), nullable=False)
    description = Column(Text)
    task_type = Column(String(50))  # cull, edit, review, export, deliver, etc.
    
    # Status and priority
    status = Column(Enum(TaskStatus), default=TaskStatus.TODO, index=True)
    priority = Column(Enum(TaskPriority), default=TaskPriority.MEDIUM, index=True)
    
    # Assignment
    assigned_to = Column(String(100))  # User/team member name
    
    # Dates
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    due_date = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Effort tracking
    estimated_hours = Column(Float)
    actual_hours = Column(Float)
    
    # Dependencies
    depends_on_task_id = Column(Integer, ForeignKey('tasks.id'))
    
    # Recipe association (can override project default)
    recipe_id = Column(Integer, ForeignKey('processing_recipes.id'), nullable=True)
    
    # Notes and metadata
    notes = Column(Text)
    meta_data = Column(JSONB, default={})
    
    # Relationships
    project = relationship("Project", back_populates="tasks")
    photos = relationship("Photo", secondary=photo_tasks, back_populates="tasks", overlaps="photo,task")
    subtasks = relationship("Task", foreign_keys=[parent_task_id], remote_side=[id], backref="parent_task")
    depends_on = relationship("Task", foreign_keys=[depends_on_task_id], remote_side=[id])
    recipe = relationship("ProcessingRecipe", foreign_keys=[recipe_id])
    
    # Indexes
    __table_args__ = (
        Index('idx_task_project_status', 'project_id', 'status'),
        Index('idx_task_assigned_status', 'assigned_to', 'status'),
        Index('idx_task_due_date', 'due_date'),
    )


class Photo(Base):
    """Main photo entity with EXIF metadata."""
    __tablename__ = 'photos'
    
    # Primary identification
    id = Column(Integer, primary_key=True)
    file_path = Column(Text, unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=False, index=True)
    file_size = Column(BigInteger)
    checksum = Column(String(64), unique=True, index=True)
    
    # Image dimensions
    image_width = Column(Integer)
    image_height = Column(Integer)
    
    # Timestamps
    date_taken = Column(DateTime, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Camera and lens information (promoted for fast queries)
    camera_make = Column(String(100), index=True)
    camera_model = Column(String(100), index=True)
    lens_model = Column(String(200), index=True)
    
    # Core exposure settings (promoted for fast queries)
    iso = Column(Integer, index=True)
    aperture = Column(Float, index=True)  # f-number (e.g., 2.8)
    shutter_speed_numeric = Column(Float, index=True)  # in seconds (e.g., 0.001 for 1/1000)
    shutter_speed_display = Column(String(50))  # human-readable (e.g., "1/1000")
    focal_length = Column(Float, index=True)  # in mm
    focal_length_35mm = Column(Integer)  # 35mm equivalent
    
    # Additional technical metadata
    exposure_compensation = Column(Float)
    flash_fired = Column(Boolean, index=True)
    white_balance = Column(String(50))
    metering_mode = Column(String(50))
    exposure_mode = Column(String(50))
    focus_mode = Column(String(50))
    focus_distance = Column(Float)
    
    # Enhanced EXIF fields for camera profile matching
    camera_serial = Column(String(100))  # For specific camera profile matching
    lens_serial = Column(String(100))    # For lens-specific corrections
    color_temp_kelvin = Column(Integer, index=True)  # Extracted WB temperature
    
    # Image quality and format
    bit_depth = Column(Integer)
    compression = Column(String(50))
    color_space = Column(String(50))
    
    # GPS coordinates (if available)
    gps_latitude = Column(Float, index=True)
    gps_longitude = Column(Float, index=True)
    gps_altitude = Column(Float)
    
    # Processing status
    processing_status = Column(
        Enum(ProcessingStatus), 
        default=ProcessingStatus.PENDING, 
        index=True
    )
    rejection_reason = Column(String(100))
    
    # Hybrid sync fields
    file_status = Column(
        Enum(FileStatus),
        default=FileStatus.LOCAL_ONLY,
        nullable=False,
        index=True
    )
    machine_id = Column(String(100), index=True)  # Which machine has the file
    storage_path = Column(Text)  # Cloud storage path if applicable
    last_sync_at = Column(DateTime)
    sync_version = Column(Integer, default=0)  # Increment on each sync
    file_modified_at = Column(DateTime)  # Track file system modification time
    
    # Complete EXIF data stored as JSONB for flexibility
    exif_data = Column(JSONB, default={})
    
    # Raw metadata from processing pipeline
    raw_meta_data = Column(JSONB, default={})
    
    # Camera profile association
    camera_profile_id = Column(Integer, ForeignKey('camera_profiles.id'), nullable=True, index=True)
    
    # Relationships
    camera_profile = relationship("CameraProfile", back_populates="photos")
    projects = relationship("Project", secondary=project_photos, back_populates="photos", overlaps="photo,project")
    tasks = relationship("Task", secondary=photo_tasks, back_populates="photos", overlaps="photo,task")
    analysis_results = relationship("AnalysisResult", back_populates="photo", cascade="all, delete-orphan")
    face_detections = relationship("FaceDetection", back_populates="photo", cascade="all, delete-orphan")
    composition_analyses = relationship("CompositionAnalysis", back_populates="photo", cascade="all, delete-orphan")
    photo_recipe = relationship("PhotoRecipe", back_populates="photo", uselist=False, cascade="all, delete-orphan")
    yolo_detection_runs = relationship("YOLODetectionRun", back_populates="photo", cascade="all, delete-orphan")
    yolo_detections = relationship("YOLODetection", back_populates="photo", cascade="all, delete-orphan")
    yolo_stats = relationship("YOLODetectionStats", back_populates="photo", uselist=False, cascade="all, delete-orphan")
    
    # Metadata relationships
    keywords = relationship("Keyword", secondary=photo_keywords, back_populates="photos")
    collections = relationship("Collection", secondary=photo_collections, back_populates="photos")
    iptc_metadata = relationship("IPTCMetadata", back_populates="photo", uselist=False, cascade="all, delete-orphan")
    
    # Album associations through the association object
    album_associations = relationship("PhotoAlbumTag", back_populates="photo", cascade="all, delete-orphan")
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_photo_camera_lens', 'camera_model', 'lens_model'),
        Index('idx_photo_settings', 'iso', 'aperture', 'focal_length'),
        Index('idx_photo_date_camera', 'date_taken', 'camera_model'),
        Index('idx_photo_gps', 'gps_latitude', 'gps_longitude'),
        Index('idx_photo_processing_status', 'processing_status'),
        Index('idx_photo_sync_status', 'file_status', 'machine_id'),
        Index('idx_photo_sync_version', 'sync_version', 'last_sync_at'),
        Index('idx_photo_camera_profile', 'camera_profile_id', 'iso'),
        Index('idx_photo_wb_temp', 'color_temp_kelvin', 'camera_model'),
        Index('idx_photo_flash_iso', 'flash_fired', 'iso'),
    )


class AnalysisResult(Base):
    """Results from technical and AI analysis."""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id'), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False)  # 'technical', 'ai_curation', 'similarity'
    
    # Technical quality scores (0-1)
    sharpness_score = Column(Float)
    exposure_quality = Column(Float)
    contrast_score = Column(Float)
    color_accuracy = Column(Float)
    noise_level = Column(Float)
    
    # AI-based scores (0-1)
    overall_ai_score = Column(Float)
    person_detected = Column(Boolean)
    face_quality_score = Column(Float)
    composition_score = Column(Float)
    emotional_impact = Column(Float)
    technical_excellence = Column(Float)
    
    # White balance analysis results
    wb_estimated_temp = Column(Integer)      # Kelvin
    wb_confidence = Column(Float)            # 0-1 confidence score
    wb_method_used = Column(String(50))      # Which WB algorithm was used
    color_cast_strength = Column(Float)      # 0-1 color cast intensity
    dominant_cast = Column(String(20))       # 'red', 'blue', 'green', etc.
    
    # Tone mapping analysis
    dynamic_range_score = Column(Float)      # 0-1 scene dynamic range
    zone_distribution = Column(JSONB)        # Ansel Adams zone distribution
    highlight_clipping = Column(Float)       # 0-1 percentage clipped
    shadow_clipping = Column(Float)          # 0-1 percentage clipped
    
    # Camera profile matching
    camera_profile_used_id = Column(Integer, ForeignKey('camera_profiles.id'))
    profile_confidence = Column(Float)       # How well profile matched
    
    # Processing performance
    processing_time_ms = Column(Integer)     # Processing duration
    algorithm_versions = Column(JSONB)       # Track which algorithm versions used
    
    # Detailed analysis data
    analysis_data = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    photo = relationship("Photo", back_populates="analysis_results")
    camera_profile_used = relationship("CameraProfile", foreign_keys=[camera_profile_used_id])
    
    # Composite indexes for efficient queries
    __table_args__ = (
        Index('idx_analysis_type_scores', 'analysis_type', 'overall_ai_score', 'sharpness_score'),
        Index('idx_analysis_wb_temp_confidence', 'wb_estimated_temp', 'wb_confidence'),
        Index('idx_analysis_clipping', 'highlight_clipping', 'shadow_clipping'),
        Index('idx_analysis_processing_time', 'processing_time_ms', 'created_at'),
        # Critical index for window function performance
        Index('idx_analysis_photo_id_desc', 'photo_id', 'id', postgresql_using='btree'),
    )


class ProcessingRecipe(Base):
    """Processing recipes/presets for batch operations."""
    __tablename__ = 'processing_recipes'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    category = Column(String(50))  # wedding, portrait, landscape, commercial, etc.
    
    # Recipe parameters stored as JSON
    parameters = Column(JSONB, nullable=False)
    
    # Tone mapping settings (separate for easier queries)
    tone_curve_type = Column(String(20), default='parametric', index=True)  # 'parametric', 'spline', 'custom'
    tone_settings = Column(JSONB)  # ToneMapperSettings serialized
    
    # Versioning
    version = Column(String(20), default="1.0")
    parent_recipe_id = Column(Integer, ForeignKey('processing_recipes.id'), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    
    # Usage tracking
    times_used = Column(Integer, default=0)
    last_used = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(100))
    
    # Recipe performance metrics
    avg_processing_time = Column(Float)
    success_rate = Column(Float)
    
    # Recipe applicability
    suitable_for = Column(JSONB, default={})  # {"scene_types": ["portrait"], "lighting": ["natural"]}
    
    # Relationships  
    parent_recipe = relationship("ProcessingRecipe", remote_side=[id])
    child_recipes = relationship("ProcessingRecipe", back_populates="parent_recipe")
    
    # Indexes
    __table_args__ = (
        Index('idx_recipe_category_active', 'category', 'is_active'),
        Index('idx_recipe_usage', 'times_used', 'last_used'),
    )


class CameraProfile(Base):
    """Camera-specific processing profiles for optimal results."""
    __tablename__ = 'camera_profiles'
    
    id = Column(Integer, primary_key=True)
    camera_make = Column(String(100), nullable=False, index=True)
    camera_model = Column(String(100), nullable=False, index=True)
    
    # Profile parameters
    high_iso_threshold = Column(Integer, default=1600)
    highlight_headroom = Column(Float, default=0.0)    # Additional highlight recovery capability
    shadow_lift_bias = Column(Float, default=0.0)      # Shadow lifting preference
    wb_confidence_boost = Column(Float, default=1.0)   # WB algorithm confidence multiplier
    color_grading_bias = Column(String(50))             # Preferred color grading preset
    
    # Noise characteristics
    base_iso = Column(Integer, default=100)
    noise_floor = Column(Float, default=0.001)
    noise_curve_params = Column(JSONB)  # Noise model parameters
    
    # Color characteristics  
    color_matrix = Column(JSONB)        # Camera-specific color calibration
    wb_presets = Column(JSONB)          # White balance presets for this camera
    
    # Profile metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(100), default='system')
    is_active = Column(Boolean, default=True, index=True)
    confidence_score = Column(Float, default=0.8)  # How reliable is this profile
    
    # Usage statistics
    times_used = Column(Integer, default=0)
    last_used = Column(DateTime)
    
    # Relationships - photos using this profile
    photos = relationship("Photo", back_populates="camera_profile")
    
    # Indexes for fast camera matching
    __table_args__ = (
        Index('idx_camera_profile_make_model', 'camera_make', 'camera_model'),
        Index('idx_camera_profile_active', 'is_active', 'confidence_score'),
    )


class BatchSession(Base):
    """Track batch processing sessions."""
    __tablename__ = 'batch_sessions'
    
    id = Column(Integer, primary_key=True)
    session_name = Column(String(200))
    input_path = Column(Text)
    output_path = Column(Text)
    
    # Session info
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    total_photos = Column(Integer)
    processed_photos = Column(Integer, default=0)
    accepted_photos = Column(Integer, default=0)
    rejected_photos = Column(Integer, default=0)
    
    # Configuration used
    config_used = Column(JSONB)
    
    # Summary and notes
    summary_report = Column(JSONB)
    notes = Column(Text)


class FaceDetection(Base):
    """Face detection results for photos."""
    __tablename__ = 'face_detections'
    
    id = Column(Integer, primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id'), nullable=False, index=True)
    
    # Face location in image
    x = Column(Integer)
    y = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    
    # Face quality metrics
    blur_score = Column(Float)
    emotion_scores = Column(JSONB)  # joy, sadness, anger, etc.
    
    # Face attributes
    age_estimate = Column(Integer)
    gender_estimate = Column(String(20))
    
    # Relationships
    photo = relationship("Photo", back_populates="face_detections")


class SimilarityGroup(Base):
    """Groups of similar photos."""
    __tablename__ = 'similarity_groups'
    
    id = Column(Integer, primary_key=True)
    group_type = Column(String(50))  # 'duplicate', 'burst', 'similar_composition'
    confidence = Column(Float)
    
    # Group metadata
    created_at = Column(DateTime, default=func.now())
    representative_photo_id = Column(Integer, ForeignKey('photos.id'))
    
    # Relationships
    photo_similarities = relationship("PhotoSimilarity", back_populates="group")


class PhotoSimilarity(Base):
    """Many-to-many relationship for photo similarity."""
    __tablename__ = 'photo_similarities'
    
    id = Column(Integer, primary_key=True)
    group_id = Column(Integer, ForeignKey('similarity_groups.id'), nullable=False, index=True)
    photo_id = Column(Integer, ForeignKey('photos.id'), nullable=False, index=True)
    
    # Similarity metrics
    visual_similarity = Column(Float)
    time_difference = Column(Float)  # seconds between shots
    
    # Relationships
    group = relationship("SimilarityGroup", back_populates="photo_similarities")


class CompositionAnalysis(Base):
    """Detailed composition analysis results."""
    __tablename__ = 'composition_analyses'
    
    id = Column(Integer, primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id'), nullable=False, index=True)
    
    # Composition rules
    rule_of_thirds_score = Column(Float)
    symmetry_score = Column(Float)
    leading_lines_score = Column(Float)
    
    # Visual elements
    dominant_colors = Column(JSONB)  # Array of RGB values
    color_harmony_score = Column(Float)
    
    # Subject detection
    main_subject_position = Column(JSONB)  # {x, y, width, height}
    background_blur_quality = Column(Float)
    
    # Relationships
    photo = relationship("Photo", back_populates="composition_analyses")


class PhotoRecipe(Base):
    """Per-photo processing recipes."""
    __tablename__ = 'photo_recipes'
    
    id = Column(Integer, primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id'), nullable=False, unique=True, index=True)
    
    # Recipe data stored as JSON
    recipe_data = Column(JSONB, nullable=False)
    
    # Tracking
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    applied_at = Column(DateTime)  # When recipe was applied to generate output
    
    # Optional reference to preset recipe
    preset_recipe_id = Column(Integer, ForeignKey('processing_recipes.id'), nullable=True)
    
    # Relationships
    photo = relationship("Photo", back_populates="photo_recipe")
    preset_recipe = relationship("ProcessingRecipe")


class YOLOModel(Base):
    """YOLO model registry."""
    __tablename__ = 'yolo_models'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50))  # yolov8n, yolov8s, yolov8m, etc.
    model_path = Column(Text)
    classes = Column(JSONB, nullable=False)  # List of class names
    input_size = Column(Integer)  # Expected input size (640, 1280, etc.)
    description = Column(Text)
    performance_metrics = Column(JSONB)  # mAP, speed, etc.
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    detection_runs = relationship("YOLODetectionRun", back_populates="model")
    
    __table_args__ = (
        Index('idx_yolo_models_active', 'is_active', 'model_name'),
    )


class YOLODetectionRun(Base):
    """YOLO detection processing runs."""
    __tablename__ = 'yolo_detection_runs'
    
    id = Column(Integer, primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id', ondelete='CASCADE'), nullable=False)
    model_id = Column(Integer, ForeignKey('yolo_models.id'))
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    confidence_threshold = Column(Float, default=0.5)
    processing_time_ms = Column(Integer)
    detection_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    machine_id = Column(String(100))
    status = Column(String(20), default='completed')
    error_message = Column(Text)
    
    # Relationships
    photo = relationship("Photo", back_populates="yolo_detection_runs")
    model = relationship("YOLOModel", back_populates="detection_runs")
    detections = relationship("YOLODetection", back_populates="run", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_yolo_runs_photo', 'photo_id'),
        Index('idx_yolo_runs_created', 'created_at'),
        Index('idx_yolo_runs_model', 'model_name', 'model_version'),
    )


class YOLODetection(Base):
    """Individual YOLO object detections."""
    __tablename__ = 'yolo_detections'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('yolo_detection_runs.id', ondelete='CASCADE'), nullable=False)
    photo_id = Column(Integer, ForeignKey('photos.id', ondelete='CASCADE'), nullable=False)
    class_name = Column(String(100), nullable=False)
    class_id = Column(Integer)
    confidence = Column(Float, nullable=False)
    
    # Bounding box coordinates (normalized 0-1)
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_width = Column(Float, nullable=False)
    bbox_height = Column(Float, nullable=False)
    
    # Pixel coordinates for convenience
    pixel_x = Column(Integer)
    pixel_y = Column(Integer)
    pixel_width = Column(Integer)
    pixel_height = Column(Integer)
    
    # Additional detection data
    segmentation_mask = Column(JSONB)
    keypoints = Column(JSONB)
    track_id = Column(String(50))  # For object tracking
    attributes = Column(JSONB)  # Extra attributes
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    run = relationship("YOLODetectionRun", back_populates="detections")
    photo = relationship("Photo", back_populates="yolo_detections")
    
    __table_args__ = (
        Index('idx_yolo_detections_run', 'run_id'),
        Index('idx_yolo_detections_photo', 'photo_id'),
        Index('idx_yolo_detections_class', 'class_name', 'confidence'),
        Index('idx_yolo_detections_confidence', 'confidence'),
        Index('idx_yolo_detections_track', 'track_id'),
    )


class YOLODetectionStats(Base):
    """Aggregated YOLO detection statistics per photo."""
    __tablename__ = 'yolo_detection_stats'
    
    id = Column(Integer, primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id', ondelete='CASCADE'), nullable=False, unique=True)
    latest_run_id = Column(Integer, ForeignKey('yolo_detection_runs.id', ondelete='SET NULL'))
    person_count = Column(Integer, default=0)
    face_count = Column(Integer, default=0)
    animal_count = Column(Integer, default=0)
    vehicle_count = Column(Integer, default=0)
    object_diversity = Column(Integer, default=0)  # Number of unique classes
    total_detections = Column(Integer, default=0)
    avg_confidence = Column(Float)
    max_confidence = Column(Float)
    scene_complexity = Column(Float)  # Derived metric
    primary_subject = Column(String(100))  # Most prominent object
    class_distribution = Column(JSONB)  # {"person": 2, "dog": 1, ...}
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    photo = relationship("Photo", back_populates="yolo_stats", uselist=False)
    latest_run = relationship("YOLODetectionRun")
    
    __table_args__ = (
        Index('idx_yolo_stats_photo', 'photo_id'),
        Index('idx_yolo_stats_people', 'person_count'),
        Index('idx_yolo_stats_diversity', 'object_diversity'),
        Index('idx_yolo_stats_subject', 'primary_subject'),
    )


class Keyword(Base):
    """Keywords/tags for photos."""
    __tablename__ = 'keywords'
    
    id = Column(Integer, primary_key=True)
    keyword = Column(String(200), unique=True, nullable=False, index=True)
    category = Column(String(100), index=True)  # Optional keyword categorization
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    created_by = Column(String(100))  # 'user', 'ai', 'import'
    
    # Relationships
    photos = relationship("Photo", secondary=photo_keywords, back_populates="keywords")
    
    __table_args__ = (
        Index('idx_keyword_usage', 'usage_count', 'last_used'),
        Index('idx_keyword_category', 'category', 'keyword'),
    )


class Collection(Base):
    """Photo collections for organization."""
    __tablename__ = 'collections'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text)
    parent_collection_id = Column(Integer, ForeignKey('collections.id'), index=True)
    
    # Collection type
    collection_type = Column(String(50), default='manual')  # 'manual', 'smart', 'album'
    smart_criteria = Column(JSONB)  # For smart collections
    
    # Display settings
    sort_by = Column(String(50), default='date_taken')
    sort_order = Column(String(10), default='asc')
    cover_photo_id = Column(Integer, ForeignKey('photos.id'))
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(100))
    
    # Privacy/sharing
    is_public = Column(Boolean, default=False)
    share_token = Column(String(100), unique=True)
    
    # Statistics
    photo_count = Column(Integer, default=0)
    
    # Relationships
    photos = relationship("Photo", secondary=photo_collections, back_populates="collections")
    parent_collection = relationship("Collection", remote_side=[id])
    subcollections = relationship("Collection", back_populates="parent_collection")
    cover_photo = relationship("Photo", foreign_keys=[cover_photo_id])
    
    __table_args__ = (
        Index('idx_collection_type', 'collection_type', 'name'),
        Index('idx_collection_parent', 'parent_collection_id'),
        Index('idx_collection_share', 'share_token', 'is_public'),
    )


class IPTCMetadata(Base):
    """IPTC/XMP metadata for photos."""
    __tablename__ = 'iptc_metadata'
    
    id = Column(Integer, primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id'), nullable=False, unique=True, index=True)
    
    # Core IPTC fields
    title = Column(String(500))
    caption = Column(Text)  # Also known as description
    headline = Column(String(500))
    
    # Creator information
    creator = Column(String(200))  # Photographer/artist name
    creator_job_title = Column(String(200))
    creator_address = Column(String(500))
    creator_city = Column(String(100))
    creator_region = Column(String(100))  # State/Province
    creator_postal_code = Column(String(50))
    creator_country = Column(String(100))
    creator_phone = Column(String(100))
    creator_email = Column(String(200))
    creator_website = Column(String(500))
    
    # Copyright and rights
    copyright_notice = Column(String(500))
    rights_usage_terms = Column(Text)
    
    # Content description
    subject_code = Column(JSONB)  # IPTC subject codes
    scene_code = Column(JSONB)  # IPTC scene codes
    genre = Column(JSONB)  # List of genres
    
    # Location (more detailed than GPS)
    sublocation = Column(String(200))  # e.g., "Central Park"
    city = Column(String(100))
    region = Column(String(100))  # State/Province
    country = Column(String(100))
    country_code = Column(String(10))
    
    # Event information
    event = Column(String(200))
    
    # Instructions
    instructions = Column(Text)  # Special instructions
    
    # Dates
    date_created = Column(DateTime)  # When the photo was taken
    
    # Source
    source = Column(String(200))
    credit = Column(String(200))
    
    # Additional metadata
    job_id = Column(String(100))
    
    # XMP sidecar sync
    xmp_sidecar_path = Column(Text)
    xmp_last_synced = Column(DateTime)
    xmp_sync_status = Column(String(50))  # 'synced', 'modified', 'missing'
    
    # Metadata tracking
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    modified_by = Column(String(100))
    
    # Full IPTC/XMP data as JSON for fields we don't explicitly track
    extended_metadata = Column(JSONB, default={})
    
    # Relationships
    photo = relationship("Photo", back_populates="iptc_metadata", uselist=False)
    
    __table_args__ = (
        Index('idx_iptc_creator', 'creator'),
        Index('idx_iptc_location', 'city', 'region', 'country'),
        Index('idx_iptc_event', 'event'),
        Index('idx_iptc_sync', 'xmp_sync_status', 'xmp_last_synced'),
    )


class AlbumTag(Base):
    """
    Album tags for organizing photos into dynamic albums.
    
    Albums are created automatically when tags are assigned to photos.
    Photos can belong to multiple albums through tags.
    """
    __tablename__ = 'album_tags'
    
    id = Column(Integer, primary_key=True)
    tag_name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200))
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(100), default='system')
    is_public = Column(Boolean, default=False, index=True)
    
    # Metadata for album configuration and settings
    album_metadata = Column(JSONB, default={})
    
    # Statistics (can be updated by triggers or batch jobs)
    photo_count = Column(Integer, default=0)
    last_photo_added = Column(DateTime)
    
    # Relationships through association object
    photo_associations = relationship("PhotoAlbumTag", back_populates="album_tag", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_album_tags_updated', 'updated_at'),
        Index('idx_album_tags_public_updated', 'is_public', 'updated_at'),
    )