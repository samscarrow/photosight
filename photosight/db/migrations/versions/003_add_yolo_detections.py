"""Add YOLO object detection tables

Revision ID: 003_add_yolo_detections
Revises: 002_add_project_management
Create Date: 2025-01-23 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '003_add_yolo_detections'
down_revision = '002_add_project_management'
branch_labels = None
depends_on = None


def upgrade():
    """Create YOLO detection tables."""
    
    # Create YOLO detection runs table
    op.create_table('yolo_detection_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('photo_id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=False),
        sa.Column('confidence_threshold', sa.Float(), default=0.5),
        sa.Column('processing_time_ms', sa.Integer()),
        sa.Column('detection_count', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('machine_id', sa.String(100)),
        sa.Column('status', sa.String(20), default='completed'),
        sa.Column('error_message', sa.Text()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['photo_id'], ['photos.id'], ondelete='CASCADE')
    )
    op.create_index('idx_yolo_runs_photo', 'yolo_detection_runs', ['photo_id'])
    op.create_index('idx_yolo_runs_created', 'yolo_detection_runs', ['created_at'])
    op.create_index('idx_yolo_runs_model', 'yolo_detection_runs', ['model_name', 'model_version'])
    
    # Create YOLO detections table
    op.create_table('yolo_detections',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('photo_id', sa.Integer(), nullable=False),
        sa.Column('class_name', sa.String(100), nullable=False),
        sa.Column('class_id', sa.Integer()),
        sa.Column('confidence', sa.Float(), nullable=False),
        # Bounding box coordinates (normalized 0-1)
        sa.Column('bbox_x', sa.Float(), nullable=False),
        sa.Column('bbox_y', sa.Float(), nullable=False),
        sa.Column('bbox_width', sa.Float(), nullable=False),
        sa.Column('bbox_height', sa.Float(), nullable=False),
        # Pixel coordinates for convenience
        sa.Column('pixel_x', sa.Integer()),
        sa.Column('pixel_y', sa.Integer()),
        sa.Column('pixel_width', sa.Integer()),
        sa.Column('pixel_height', sa.Integer()),
        # Additional detection data
        sa.Column('segmentation_mask', postgresql.JSONB()),
        sa.Column('keypoints', postgresql.JSONB()),
        sa.Column('track_id', sa.String(50)),  # For object tracking
        sa.Column('attributes', postgresql.JSONB()),  # Extra attributes
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['run_id'], ['yolo_detection_runs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['photo_id'], ['photos.id'], ondelete='CASCADE')
    )
    op.create_index('idx_yolo_detections_run', 'yolo_detections', ['run_id'])
    op.create_index('idx_yolo_detections_photo', 'yolo_detections', ['photo_id'])
    op.create_index('idx_yolo_detections_class', 'yolo_detections', ['class_name', 'confidence'])
    op.create_index('idx_yolo_detections_confidence', 'yolo_detections', ['confidence'])
    op.create_index('idx_yolo_detections_track', 'yolo_detections', ['track_id'])
    
    # Create aggregated detection statistics table
    op.create_table('yolo_detection_stats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('photo_id', sa.Integer(), nullable=False, unique=True),
        sa.Column('latest_run_id', sa.Integer()),
        sa.Column('person_count', sa.Integer(), default=0),
        sa.Column('face_count', sa.Integer(), default=0),
        sa.Column('animal_count', sa.Integer(), default=0),
        sa.Column('vehicle_count', sa.Integer(), default=0),
        sa.Column('object_diversity', sa.Integer(), default=0),  # Number of unique classes
        sa.Column('total_detections', sa.Integer(), default=0),
        sa.Column('avg_confidence', sa.Float()),
        sa.Column('max_confidence', sa.Float()),
        sa.Column('scene_complexity', sa.Float()),  # Derived metric
        sa.Column('primary_subject', sa.String(100)),  # Most prominent object
        sa.Column('class_distribution', postgresql.JSONB()),  # {"person": 2, "dog": 1, ...}
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['photo_id'], ['photos.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['latest_run_id'], ['yolo_detection_runs.id'], ondelete='SET NULL')
    )
    op.create_index('idx_yolo_stats_photo', 'yolo_detection_stats', ['photo_id'])
    op.create_index('idx_yolo_stats_people', 'yolo_detection_stats', ['person_count'])
    op.create_index('idx_yolo_stats_diversity', 'yolo_detection_stats', ['object_diversity'])
    op.create_index('idx_yolo_stats_subject', 'yolo_detection_stats', ['primary_subject'])
    
    # Create YOLO model registry table
    op.create_table('yolo_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=False),
        sa.Column('model_type', sa.String(50)),  # yolov8n, yolov8s, yolov8m, etc.
        sa.Column('model_path', sa.Text()),
        sa.Column('classes', postgresql.JSONB(), nullable=False),  # List of class names
        sa.Column('input_size', sa.Integer()),  # Expected input size (640, 1280, etc.)
        sa.Column('description', sa.Text()),
        sa.Column('performance_metrics', postgresql.JSONB()),  # mAP, speed, etc.
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name', 'model_version', name='uq_yolo_model_version')
    )
    op.create_index('idx_yolo_models_active', 'yolo_models', ['is_active', 'model_name'])
    
    # Add foreign key from detection_runs to models
    op.add_column('yolo_detection_runs', 
        sa.Column('model_id', sa.Integer(), sa.ForeignKey('yolo_models.id'))
    )
    
    # Create view for latest detections per photo
    op.execute("""
        CREATE OR REPLACE VIEW v_latest_yolo_detections AS
        SELECT DISTINCT ON (d.photo_id, d.class_name)
            d.*,
            r.model_name,
            r.model_version,
            r.created_at as run_date
        FROM yolo_detections d
        JOIN yolo_detection_runs r ON d.run_id = r.id
        WHERE r.status = 'completed'
        ORDER BY d.photo_id, d.class_name, r.created_at DESC, d.confidence DESC;
    """)


def downgrade():
    """Drop YOLO detection tables."""
    op.execute("DROP VIEW IF EXISTS v_latest_yolo_detections;")
    op.drop_table('yolo_detections')
    op.drop_table('yolo_detection_stats')
    op.drop_table('yolo_detection_runs')
    op.drop_table('yolo_models')