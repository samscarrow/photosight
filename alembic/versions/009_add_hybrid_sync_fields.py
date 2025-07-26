"""add hybrid sync fields

Revision ID: 005_add_hybrid_sync_fields
Revises: 004_fix_project_schema
Create Date: 2025-01-22 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '009_add_hybrid_sync_fields'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade():
    # Create FileStatus enum type
    op.execute("CREATE TYPE filestatus AS ENUM ('local_only', 'cloud_only', 'synced', 'local_modified', 'cloud_modified', 'missing', 'conflict')")
    
    # Add new columns to photos table
    op.add_column('photos', sa.Column('file_status', sa.Enum('local_only', 'cloud_only', 'synced', 'local_modified', 'cloud_modified', 'missing', 'conflict', name='filestatus'), nullable=False, server_default='local_only'))
    op.add_column('photos', sa.Column('machine_id', sa.String(length=100), nullable=True))
    op.add_column('photos', sa.Column('storage_path', sa.Text(), nullable=True))
    op.add_column('photos', sa.Column('last_sync_at', sa.DateTime(), nullable=True))
    op.add_column('photos', sa.Column('sync_version', sa.Integer(), nullable=True, server_default='0'))
    op.add_column('photos', sa.Column('file_modified_at', sa.DateTime(), nullable=True))
    
    # Create indexes for sync fields
    op.create_index('idx_photo_sync_status', 'photos', ['file_status', 'machine_id'])
    op.create_index('idx_photo_sync_version', 'photos', ['sync_version', 'last_sync_at'])
    op.create_index(op.f('ix_photos_machine_id'), 'photos', ['machine_id'])
    op.create_index(op.f('ix_photos_file_status'), 'photos', ['file_status'])
    
    # Remove server defaults after adding columns
    op.alter_column('photos', 'file_status', server_default=None)
    op.alter_column('photos', 'sync_version', server_default=None)


def downgrade():
    # Drop indexes
    op.drop_index(op.f('ix_photos_file_status'), table_name='photos')
    op.drop_index(op.f('ix_photos_machine_id'), table_name='photos')
    op.drop_index('idx_photo_sync_version', table_name='photos')
    op.drop_index('idx_photo_sync_status', table_name='photos')
    
    # Drop columns
    op.drop_column('photos', 'file_modified_at')
    op.drop_column('photos', 'sync_version')
    op.drop_column('photos', 'last_sync_at')
    op.drop_column('photos', 'storage_path')
    op.drop_column('photos', 'machine_id')
    op.drop_column('photos', 'file_status')
    
    # Drop enum type
    op.execute('DROP TYPE filestatus')