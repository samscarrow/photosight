"""Add promoted EXIF fields to photos table

Revision ID: 001
Revises: 
Create Date: 2025-07-22 15:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add promoted EXIF columns to photos table."""
    
    # First, create the base photos table if it doesn't exist
    op.create_table(
        'photos',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=True),
        sa.Column('checksum', sa.String(64), nullable=True),
        
        # Promoted EXIF fields for efficient querying
        sa.Column('image_width', sa.Integer(), nullable=True),
        sa.Column('image_height', sa.Integer(), nullable=True),
        sa.Column('date_taken', sa.DateTime(), nullable=True),
        sa.Column('camera_make', sa.String(100), nullable=True),
        sa.Column('camera_model', sa.String(100), nullable=True),
        sa.Column('lens_model', sa.String(200), nullable=True),
        sa.Column('iso', sa.Integer(), nullable=True),
        sa.Column('aperture', sa.Float(), nullable=True),
        sa.Column('shutter_speed_numeric', sa.Float(), nullable=True),
        sa.Column('shutter_speed_display', sa.String(50), nullable=True),
        sa.Column('focal_length', sa.Float(), nullable=True),
        sa.Column('focal_length_35mm', sa.Integer(), nullable=True),
        
        # Additional technical metadata
        sa.Column('flash_fired', sa.Boolean(), nullable=True),
        sa.Column('white_balance', sa.String(50), nullable=True),
        sa.Column('metering_mode', sa.String(50), nullable=True),
        sa.Column('exposure_mode', sa.String(50), nullable=True),
        sa.Column('exposure_compensation', sa.Float(), nullable=True),
        
        # Image quality and format
        sa.Column('bit_depth', sa.Integer(), nullable=True),
        sa.Column('compression', sa.String(50), nullable=True),
        sa.Column('color_space', sa.String(50), nullable=True),
        
        # GPS coordinates
        sa.Column('gps_latitude', sa.Float(), nullable=True),
        sa.Column('gps_longitude', sa.Float(), nullable=True),
        sa.Column('gps_altitude', sa.Float(), nullable=True),
        
        # Timestamps
        sa.Column('imported_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_analyzed', sa.DateTime(), nullable=True),
        
        # Processing status
        sa.Column('processing_status', sa.String(50), server_default='pending', nullable=True),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        
        # Complete EXIF and metadata as JSONB
        sa.Column('exif_data', JSONB, nullable=True),
        sa.Column('raw_metadata', JSONB, nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('file_path')
    )
    
    # Create indexes for efficient querying
    op.create_index('idx_photos_status', 'photos', ['processing_status'])
    op.create_index('idx_photos_date_taken', 'photos', ['date_taken'])
    op.create_index('idx_photos_camera_model', 'photos', ['camera_model'])
    op.create_index('idx_photos_lens_model', 'photos', ['lens_model'])
    op.create_index('idx_photos_iso', 'photos', ['iso'])
    op.create_index('idx_photos_aperture', 'photos', ['aperture'])
    op.create_index('idx_photos_focal_length', 'photos', ['focal_length'])
    op.create_index('idx_photos_focal_length_35mm', 'photos', ['focal_length_35mm'])
    op.create_index('idx_photos_dimensions', 'photos', ['image_width', 'image_height'])
    op.create_index('idx_photos_gps', 'photos', ['gps_latitude', 'gps_longitude'])
    op.create_index('idx_photos_imported', 'photos', ['imported_at'])
    
    # Composite indexes for common queries
    op.create_index('idx_photos_camera_iso', 'photos', ['camera_model', 'iso'])
    op.create_index('idx_photos_lens_focal', 'photos', ['lens_model', 'focal_length'])
    op.create_index('idx_photos_date_camera', 'photos', ['date_taken', 'camera_model'])


def downgrade() -> None:
    """Remove promoted EXIF fields and revert to basic schema."""
    
    # Drop indexes first
    op.drop_index('idx_photos_date_camera', 'photos')
    op.drop_index('idx_photos_lens_focal', 'photos')
    op.drop_index('idx_photos_camera_iso', 'photos')
    op.drop_index('idx_photos_imported', 'photos')
    op.drop_index('idx_photos_gps', 'photos')
    op.drop_index('idx_photos_dimensions', 'photos')
    op.drop_index('idx_photos_focal_length_35mm', 'photos')
    op.drop_index('idx_photos_focal_length', 'photos')
    op.drop_index('idx_photos_aperture', 'photos')
    op.drop_index('idx_photos_iso', 'photos')
    op.drop_index('idx_photos_lens_model', 'photos')
    op.drop_index('idx_photos_camera_model', 'photos')
    op.drop_index('idx_photos_date_taken', 'photos')
    op.drop_index('idx_photos_status', 'photos')
    
    # Drop the entire photos table (this is destructive!)
    op.drop_table('photos')