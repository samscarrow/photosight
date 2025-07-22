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


class ProjectStatus(enum.Enum):
    """Project status."""
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class TaskStatus(enum.Enum):
    """Task status."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class TaskPriority(enum.Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# Association table for many-to-many relationship between photos and tasks
photo_tasks = Table(
    'photo_tasks',
    Base.metadata,
    Column('photo_id', Integer, ForeignKey('photos.id'), primary_key=True),
    Column('task_id', Integer, ForeignKey('tasks.id'), primary_key=True)
)


class Project(Base):
    """Photography project (shoot, event, assignment)."""
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True, index=True)
    description = Column(Text)
    client_name = Column(String(200), index=True)
    project_type = Column(String(50))  # wedding, portrait, commercial, landscape, etc.
    
    # Dates
    shoot_date = Column(DateTime)
    due_date = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime)
    
    # Status and metadata
    status = Column(Enum(ProjectStatus), default=ProjectStatus.PLANNING, index=True)
    budget = Column(Float)
    location = Column(String(500))
    notes = Column(Text)
    
    # Deliverables tracking
    expected_photos = Column(Integer)
    delivered_photos = Column(Integer, default=0)
    
    # Custom metadata
    meta_data = Column(JSONB, default={})
    
    # Relationships
    photos = relationship("Photo", back_populates="project", cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_project_status_due', 'status', 'due_date'),
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
    
    # Notes and metadata
    notes = Column(Text)
    meta_data = Column(JSONB, default={})
    
    # Relationships
    project = relationship("Project", back_populates="tasks")
    photos = relationship("Photo", secondary=photo_tasks, back_populates="tasks")
    subtasks = relationship("Task", backref="parent_task", remote_side=[id])
    depends_on = relationship("Task", foreign_keys=[depends_on_task_id], remote_side=[id])
    
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
    
    # Project relationship
    project_id = Column(Integer, ForeignKey('projects.id'), index=True)
    
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
    
    # Complete EXIF data stored as JSONB for flexibility
    exif_data = Column(JSONB, default={})
    
    # Raw metadata from processing pipeline
    raw_meta_data = Column(JSONB, default={})
    
    # Relationships
    project = relationship("Project", back_populates="photos")
    tasks = relationship("Task", secondary=photo_tasks, back_populates="photos")
    analysis_results = relationship("AnalysisResult", back_populates="photo", cascade="all, delete-orphan")
    face_detections = relationship("FaceDetection", back_populates="photo", cascade="all, delete-orphan")
    composition_analyses = relationship("CompositionAnalysis", back_populates="photo", cascade="all, delete-orphan")
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_photo_camera_lens', 'camera_model', 'lens_model'),
        Index('idx_photo_settings', 'iso', 'aperture', 'focal_length'),
        Index('idx_photo_date_camera', 'date_taken', 'camera_model'),
        Index('idx_photo_gps', 'gps_latitude', 'gps_longitude'),
        Index('idx_photo_project_status', 'project_id', 'processing_status'),
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
    
    # Detailed analysis data
    analysis_data = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    photo = relationship("Photo", back_populates="analysis_results")
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_analysis_type_scores', 'analysis_type', 'overall_ai_score', 'sharpness_score'),
    )


class ProcessingRecipe(Base):
    """Processing recipes/presets for batch operations."""
    __tablename__ = 'processing_recipes'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    
    # Recipe parameters stored as JSON
    parameters = Column(JSONB, nullable=False)
    
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