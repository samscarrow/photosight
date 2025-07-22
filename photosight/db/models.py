"""
SQLAlchemy models for PhotoSight database.

Defines the database schema for storing photo metadata,
analysis results, processing recipes, and batch processing information.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, 
    Text, BigInteger, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class Photo(Base):
    """Core photo metadata and file information"""
    __tablename__ = 'photos'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(Text, unique=True, nullable=False)
    filename = Column(String(255), nullable=False)
    file_size = Column(BigInteger)
    checksum = Column(String(64))  # SHA-256 hash
    
    # Camera metadata
    camera_make = Column(String(100))
    camera_model = Column(String(100))
    lens_model = Column(String(200))
    focal_length = Column(Float)
    aperture = Column(Float)
    shutter_speed = Column(String(50))  # e.g., "1/200"
    iso_value = Column(Integer)
    
    # Timestamps
    captured_at = Column(DateTime)
    imported_at = Column(DateTime, default=datetime.utcnow)
    last_analyzed = Column(DateTime)
    
    # Processing status
    processing_status = Column(String(50), default='pending')  # pending, analyzed, processed, rejected
    rejection_reason = Column(Text)
    
    # EXIF and metadata as JSON
    exif_data = Column(JSONB)
    color_profile = Column(String(100))
    
    # Relationships
    analysis_results = relationship("AnalysisResult", back_populates="photo", cascade="all, delete-orphan")
    processing_recipes = relationship("ProcessingRecipe", back_populates="photo", cascade="all, delete-orphan")
    face_detections = relationship("FaceDetection", back_populates="photo", cascade="all, delete-orphan")
    similarity_relationships = relationship("PhotoSimilarity", back_populates="photo", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_photos_status', 'processing_status'),
        Index('idx_photos_captured', 'captured_at'),
        Index('idx_photos_iso', 'iso_value'),
        Index('idx_photos_focal_length', 'focal_length'),
    )


class AnalysisResult(Base):
    """Technical and AI analysis results for photos"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id'), nullable=False)
    analysis_type = Column(String(50), nullable=False)  # technical, ai_curation, similarity
    analysis_version = Column(String(20))  # Track analysis algorithm versions
    
    # Technical analysis scores
    sharpness_score = Column(Float)
    blur_score = Column(Float)
    exposure_quality = Column(Float)
    contrast_score = Column(Float)
    saturation_score = Column(Float)
    noise_level = Column(Float)
    
    # AI curation scores
    overall_ai_score = Column(Float)
    person_detected = Column(Boolean, default=False)
    person_count = Column(Integer, default=0)
    face_quality_score = Column(Float)
    composition_score = Column(Float)
    aesthetic_score = Column(Float)
    
    # Histogram and exposure analysis
    histogram_data = Column(JSONB)  # RGB histograms
    black_clip_percentage = Column(Float)
    white_clip_percentage = Column(Float)
    mean_brightness = Column(Float)
    shadow_percentage = Column(Float)
    highlight_percentage = Column(Float)
    
    # Additional analysis data
    analysis_data = Column(JSONB)  # Flexible storage for additional metrics
    
    # Processing metadata
    processing_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    photo = relationship("Photo", back_populates="analysis_results")
    
    # Indexes
    __table_args__ = (
        Index('idx_analysis_photo_type', 'photo_id', 'analysis_type'),
        Index('idx_analysis_overall_score', 'overall_ai_score'),
        Index('idx_analysis_sharpness', 'sharpness_score'),
        UniqueConstraint('photo_id', 'analysis_type', 'analysis_version', name='uq_photo_analysis_version'),
    )


class ProcessingRecipe(Base):
    """RAW processing recipes and iterations"""
    __tablename__ = 'processing_recipes'
    
    id = Column(Integer, primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id'), nullable=False)
    recipe_name = Column(String(200))
    iteration_number = Column(Integer, default=1)
    
    # Recipe parameters as JSON
    recipe_data = Column(JSONB, nullable=False)
    recipe_hash = Column(String(64))  # SHA-256 of recipe_data for deduplication
    
    # Processing results
    output_file_path = Column(Text)
    processing_time_ms = Column(Integer)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Preview information
    preview_generated = Column(Boolean, default=False)
    preview_path = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))  # processing engine version
    
    # Relationship
    photo = relationship("Photo", back_populates="processing_recipes")
    
    # Indexes
    __table_args__ = (
        Index('idx_recipe_photo_iteration', 'photo_id', 'iteration_number'),
        Index('idx_recipe_hash', 'recipe_hash'),
        Index('idx_recipe_success', 'success'),
    )


class BatchSession(Base):
    """Batch processing session tracking"""
    __tablename__ = 'batch_sessions'
    
    id = Column(Integer, primary_key=True)
    session_name = Column(String(200))
    
    # Session metadata
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    total_photos = Column(Integer, default=0)
    processed_photos = Column(Integer, default=0)
    accepted_photos = Column(Integer, default=0)
    rejected_photos = Column(Integer, default=0)
    
    # Processing statistics
    total_processing_time_ms = Column(BigInteger)
    average_processing_time_ms = Column(Float)
    
    # Configuration and settings
    config_snapshot = Column(JSONB)  # Store config used for this batch
    processing_stages = Column(JSONB)  # Which stages were run
    
    # Status and results
    status = Column(String(50), default='running')  # running, completed, failed, cancelled
    error_count = Column(Integer, default=0)
    summary_report = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class FaceDetection(Base):
    """Individual face detection results"""
    __tablename__ = 'face_detections'
    
    id = Column(Integer, primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id'), nullable=False)
    
    # Bounding box coordinates (normalized 0-1)
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_width = Column(Float, nullable=False)
    bbox_height = Column(Float, nullable=False)
    
    # Face quality metrics
    confidence_score = Column(Float)
    sharpness = Column(Float)
    brightness = Column(Float)
    size_pixels = Column(Integer)
    
    # Face characteristics
    age_estimate = Column(Integer)
    emotion_scores = Column(JSONB)  # happiness, surprise, etc.
    pose_angles = Column(JSONB)  # pitch, yaw, roll
    
    # Processing metadata
    detection_model = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    photo = relationship("Photo", back_populates="face_detections")
    
    # Indexes
    __table_args__ = (
        Index('idx_face_photo_confidence', 'photo_id', 'confidence_score'),
        Index('idx_face_quality', 'sharpness', 'brightness'),
    )


class SimilarityGroup(Base):
    """Groups of similar photos"""
    __tablename__ = 'similarity_groups'
    
    id = Column(Integer, primary_key=True)
    group_hash = Column(String(64), unique=True)  # Hash of similarity features
    similarity_threshold = Column(Float)
    photo_count = Column(Integer, default=0)
    
    # Representative photo (best quality in group)
    representative_photo_id = Column(Integer, ForeignKey('photos.id'))
    
    # Group metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    algorithm_version = Column(String(20))
    
    # Relationships
    photos = relationship("PhotoSimilarity", back_populates="group")


class PhotoSimilarity(Base):
    """Many-to-many relationship for photo similarities"""
    __tablename__ = 'photo_similarities'
    
    photo_id = Column(Integer, ForeignKey('photos.id'), primary_key=True)
    group_id = Column(Integer, ForeignKey('similarity_groups.id'), primary_key=True)
    similarity_score = Column(Float)  # How similar to group representative
    perceptual_hash = Column(String(64))  # Photo's perceptual hash
    
    is_representative = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    photo = relationship("Photo", back_populates="similarity_relationships")
    group = relationship("SimilarityGroup", back_populates="photos")
    
    # Indexes
    __table_args__ = (
        Index('idx_similarity_score', 'similarity_score'),
        Index('idx_similarity_hash', 'perceptual_hash'),
    )


class CompositionAnalysis(Base):
    """Detailed composition analysis results"""
    __tablename__ = 'composition_analysis'
    
    id = Column(Integer, primary_key=True)
    photo_id = Column(Integer, ForeignKey('photos.id'), nullable=False)
    
    # Rule of thirds analysis
    rule_of_thirds_score = Column(Float)
    subject_placement_score = Column(Float)
    
    # Visual balance
    balance_score = Column(Float)
    symmetry_score = Column(Float)
    
    # Leading lines and patterns
    leading_lines_detected = Column(Boolean, default=False)
    leading_lines_score = Column(Float)
    pattern_detection_score = Column(Float)
    
    # Color composition
    color_harmony_score = Column(Float)
    dominant_colors = Column(JSONB)  # RGB values of dominant colors
    color_distribution = Column(JSONB)
    
    # Depth and layers
    depth_score = Column(Float)
    foreground_background_separation = Column(Float)
    
    # Overall composition
    overall_composition_score = Column(Float)
    
    # Analysis metadata
    analysis_version = Column(String(20))
    processing_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_composition_photo', 'photo_id'),
        Index('idx_composition_overall', 'overall_composition_score'),
        Index('idx_composition_rule_thirds', 'rule_of_thirds_score'),
    )