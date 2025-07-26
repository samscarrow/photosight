"""Fix project schema and add recipe associations

Revision ID: 004
Revises: 003
Create Date: 2025-01-22

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade():
    # Create new enums
    project_phase_enum = postgresql.ENUM(
        'Capture', 'Import', 'Cull', 'Edit', 'Review', 'Export', 'Deliver',
        name='projectphase'
    )
    project_phase_enum.create(op.get_bind())
    
    # Update existing enums
    op.execute("ALTER TYPE projectstatus RENAME VALUE 'planning' TO 'Planning'")
    op.execute("ALTER TYPE projectstatus RENAME VALUE 'active' TO 'Active'")
    op.execute("ALTER TYPE projectstatus RENAME VALUE 'on_hold' TO 'On Hold'")
    op.execute("ALTER TYPE projectstatus RENAME VALUE 'completed' TO 'Completed'")
    op.execute("ALTER TYPE projectstatus RENAME VALUE 'archived' TO 'Archived'")
    
    op.execute("ALTER TYPE taskstatus RENAME VALUE 'todo' TO 'To Do'")
    op.execute("ALTER TYPE taskstatus RENAME VALUE 'in_progress' TO 'In Progress'")
    op.execute("ALTER TYPE taskstatus RENAME VALUE 'review' TO 'Code Review'")
    op.execute("ALTER TYPE taskstatus RENAME VALUE 'completed' TO 'Done'")
    op.execute("ALTER TYPE taskstatus RENAME VALUE 'blocked' TO 'Blocked'")
    
    op.execute("ALTER TYPE taskpriority RENAME VALUE 'low' TO 'P3-Low'")
    op.execute("ALTER TYPE taskpriority RENAME VALUE 'medium' TO 'P2-Medium'")
    op.execute("ALTER TYPE taskpriority RENAME VALUE 'high' TO 'P1-High'")
    op.execute("ALTER TYPE taskpriority RENAME VALUE 'urgent' TO 'P0-Critical'")
    op.execute("ALTER TYPE taskpriority ADD VALUE 'P0-Critical'")
    
    # Add new columns to projects table
    op.add_column('projects', sa.Column('phase', 
        sa.Enum('Capture', 'Import', 'Cull', 'Edit', 'Review', 'Export', 'Deliver', 
                name='projectphase'), nullable=True))
    op.add_column('projects', sa.Column('priority', 
        sa.Enum('P0-Critical', 'P1-High', 'P2-Medium', 'P3-Low', 
                name='taskpriority'), nullable=True))
    op.add_column('projects', sa.Column('default_recipe_id', sa.Integer(), nullable=True))
    
    # Add foreign key constraint
    op.create_foreign_key('fk_projects_default_recipe', 'projects', 'processing_recipes', 
                         ['default_recipe_id'], ['id'])
    
    # Add recipe_id to tasks table
    op.add_column('tasks', sa.Column('recipe_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_tasks_recipe', 'tasks', 'processing_recipes', 
                         ['recipe_id'], ['id'])
    
    # Create project_photos association table
    op.create_table('project_photos',
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('photo_id', sa.Integer(), nullable=False),
        sa.Column('added_at', sa.DateTime(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['photo_id'], ['photos.id'], ),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
        sa.PrimaryKeyConstraint('project_id', 'photo_id')
    )
    
    # Migrate existing project_id relationships to association table
    op.execute("""
        INSERT INTO project_photos (project_id, photo_id, added_at)
        SELECT project_id, id, created_at 
        FROM photos 
        WHERE project_id IS NOT NULL
    """)
    
    # Remove project_id column from photos table
    op.drop_constraint('photos_project_id_fkey', 'photos', type_='foreignkey')
    op.drop_column('photos', 'project_id')
    
    # Update processing_recipes table
    op.add_column('processing_recipes', sa.Column('category', sa.String(50), nullable=True))
    op.add_column('processing_recipes', sa.Column('version', sa.String(20), nullable=True))
    op.add_column('processing_recipes', sa.Column('parent_recipe_id', sa.Integer(), nullable=True))
    op.add_column('processing_recipes', sa.Column('is_active', sa.Boolean(), nullable=True))
    op.add_column('processing_recipes', sa.Column('suitable_for', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    
    # Set default values
    op.execute("UPDATE processing_recipes SET version = '1.0' WHERE version IS NULL")
    op.execute("UPDATE processing_recipes SET is_active = true WHERE is_active IS NULL")
    op.execute("UPDATE processing_recipes SET suitable_for = '{}' WHERE suitable_for IS NULL")
    op.execute("UPDATE projects SET phase = 'Capture' WHERE phase IS NULL")
    op.execute("UPDATE projects SET priority = 'P2-Medium' WHERE priority IS NULL")
    
    # Add foreign key for parent recipe
    op.create_foreign_key('fk_recipes_parent', 'processing_recipes', 'processing_recipes',
                         ['parent_recipe_id'], ['id'])
    
    # Create new indexes
    op.create_index('idx_project_phase_priority', 'projects', ['phase', 'priority'])
    op.create_index('idx_recipe_category_active', 'processing_recipes', ['category', 'is_active'])
    op.create_index('idx_recipe_usage', 'processing_recipes', ['times_used', 'last_used'])
    
    # Drop old index
    op.drop_index('idx_photo_project_status', table_name='photos')
    op.create_index('idx_photo_processing_status', 'photos', ['processing_status'])


def downgrade():
    # This is a complex migration - downgrade would be very involved
    # For now, we'll provide a basic structure
    
    # Drop new indexes
    op.drop_index('idx_photo_processing_status', table_name='photos')
    op.drop_index('idx_recipe_usage', table_name='processing_recipes')
    op.drop_index('idx_recipe_category_active', table_name='processing_recipes')
    op.drop_index('idx_project_phase_priority', table_name='projects')
    
    # Add project_id back to photos (would need data migration)
    op.add_column('photos', sa.Column('project_id', sa.Integer(), nullable=True))
    
    # Drop association table
    op.drop_table('project_photos')
    
    # Remove new columns
    op.drop_column('processing_recipes', 'suitable_for')
    op.drop_column('processing_recipes', 'is_active')
    op.drop_column('processing_recipes', 'parent_recipe_id')
    op.drop_column('processing_recipes', 'version')
    op.drop_column('processing_recipes', 'category')
    op.drop_column('tasks', 'recipe_id')
    op.drop_column('projects', 'default_recipe_id')
    op.drop_column('projects', 'priority')
    op.drop_column('projects', 'phase')
    
    # Drop enum
    op.execute("DROP TYPE projectphase")