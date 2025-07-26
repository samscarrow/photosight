"""Add project management tables

Revision ID: 002
Revises: 001
Create Date: 2025-07-22

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # Create enum types
    op.execute("CREATE TYPE projectstatus AS ENUM ('planning', 'active', 'on_hold', 'completed', 'archived')")
    op.execute("CREATE TYPE taskstatus AS ENUM ('todo', 'in_progress', 'review', 'completed', 'blocked')")
    op.execute("CREATE TYPE taskpriority AS ENUM ('low', 'medium', 'high', 'urgent')")
    
    # Create projects table
    op.create_table('projects',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('client_name', sa.String(length=200), nullable=True),
        sa.Column('project_type', sa.String(length=50), nullable=True),
        sa.Column('shoot_date', sa.DateTime(), nullable=True),
        sa.Column('due_date', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('status', postgresql.ENUM('planning', 'active', 'on_hold', 'completed', 'archived', name='projectstatus'), nullable=True),
        sa.Column('budget', sa.Float(), nullable=True),
        sa.Column('location', sa.String(length=500), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('expected_photos', sa.Integer(), nullable=True),
        sa.Column('delivered_photos', sa.Integer(), nullable=True, default=0),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_project_client', 'projects', ['client_name'], unique=False)
    op.create_index('idx_project_status_due', 'projects', ['status', 'due_date'], unique=False)
    op.create_index(op.f('ix_projects_name'), 'projects', ['name'], unique=True)
    op.create_index(op.f('ix_projects_status'), 'projects', ['status'], unique=False)
    
    # Create tasks table
    op.create_table('tasks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('parent_task_id', sa.Integer(), nullable=True),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('task_type', sa.String(length=50), nullable=True),
        sa.Column('status', postgresql.ENUM('todo', 'in_progress', 'review', 'completed', 'blocked', name='taskstatus'), nullable=True),
        sa.Column('priority', postgresql.ENUM('low', 'medium', 'high', 'urgent', name='taskpriority'), nullable=True),
        sa.Column('assigned_to', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('due_date', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('estimated_hours', sa.Float(), nullable=True),
        sa.Column('actual_hours', sa.Float(), nullable=True),
        sa.Column('depends_on_task_id', sa.Integer(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['depends_on_task_id'], ['tasks.id'], ),
        sa.ForeignKeyConstraint(['parent_task_id'], ['tasks.id'], ),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_task_assigned_status', 'tasks', ['assigned_to', 'status'], unique=False)
    op.create_index('idx_task_due_date', 'tasks', ['due_date'], unique=False)
    op.create_index('idx_task_project_status', 'tasks', ['project_id', 'status'], unique=False)
    op.create_index(op.f('ix_tasks_depends_on_task_id'), 'tasks', ['depends_on_task_id'], unique=False)
    op.create_index(op.f('ix_tasks_parent_task_id'), 'tasks', ['parent_task_id'], unique=False)
    op.create_index(op.f('ix_tasks_priority'), 'tasks', ['priority'], unique=False)
    op.create_index(op.f('ix_tasks_project_id'), 'tasks', ['project_id'], unique=False)
    op.create_index(op.f('ix_tasks_status'), 'tasks', ['status'], unique=False)
    
    # Create photo_tasks association table
    op.create_table('photo_tasks',
        sa.Column('photo_id', sa.Integer(), nullable=False),
        sa.Column('task_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['photo_id'], ['photos.id'], ),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
        sa.PrimaryKeyConstraint('photo_id', 'task_id')
    )
    
    # Add project_id column to photos table
    op.add_column('photos', sa.Column('project_id', sa.Integer(), nullable=True))
    op.create_index('idx_photo_project_status', 'photos', ['project_id', 'processing_status'], unique=False)
    op.create_index(op.f('ix_photos_project_id'), 'photos', ['project_id'], unique=False)
    op.create_foreign_key('fk_photos_project', 'photos', 'projects', ['project_id'], ['id'])


def downgrade():
    # Drop foreign key and indexes from photos table
    op.drop_constraint('fk_photos_project', 'photos', type_='foreignkey')
    op.drop_index(op.f('ix_photos_project_id'), table_name='photos')
    op.drop_index('idx_photo_project_status', table_name='photos')
    op.drop_column('photos', 'project_id')
    
    # Drop photo_tasks association table
    op.drop_table('photo_tasks')
    
    # Drop tasks table
    op.drop_index(op.f('ix_tasks_status'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_project_id'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_priority'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_parent_task_id'), table_name='tasks')
    op.drop_index(op.f('ix_tasks_depends_on_task_id'), table_name='tasks')
    op.drop_index('idx_task_project_status', table_name='tasks')
    op.drop_index('idx_task_due_date', table_name='tasks')
    op.drop_index('idx_task_assigned_status', table_name='tasks')
    op.drop_table('tasks')
    
    # Drop projects table
    op.drop_index(op.f('ix_projects_status'), table_name='projects')
    op.drop_index(op.f('ix_projects_name'), table_name='projects')
    op.drop_index('idx_project_status_due', table_name='projects')
    op.drop_index('idx_project_client', table_name='projects')
    op.drop_table('projects')
    
    # Drop enum types
    op.execute('DROP TYPE IF EXISTS taskpriority')
    op.execute('DROP TYPE IF EXISTS taskstatus')
    op.execute('DROP TYPE IF EXISTS projectstatus')