"""Add photo_recipes table

Revision ID: 003
Revises: 002
Create Date: 2025-01-19

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '007'
down_revision = '003_add_yolo_detections'
branch_labels = None
depends_on = None


def upgrade():
    # Create photo_recipes table
    op.create_table('photo_recipes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('photo_id', sa.Integer(), nullable=False),
        sa.Column('recipe_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('applied_at', sa.DateTime(), nullable=True),
        sa.Column('preset_recipe_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['photo_id'], ['photos.id'], ),
        sa.ForeignKeyConstraint(['preset_recipe_id'], ['processing_recipes.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_photo_recipes_photo_id'), 'photo_recipes', ['photo_id'], unique=True)


def downgrade():
    op.drop_index(op.f('ix_photo_recipes_photo_id'), table_name='photo_recipes')
    op.drop_table('photo_recipes')