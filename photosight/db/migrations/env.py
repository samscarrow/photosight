"""
Alembic environment configuration for PhotoSight database migrations.
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Add the project root to the path so we can import photosight modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import PhotoSight modules
from photosight.config import load_config
from photosight.db.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_database_url():
    """Get database URL from PhotoSight configuration."""
    try:
        # Load PhotoSight config
        config_path = project_root / "config.yaml"
        photosight_config = load_config(config_path)
        
        db_config = photosight_config.get('database', {})
        if not db_config.get('enabled', False):
            raise ValueError("Database not enabled in PhotoSight configuration")
            
        return db_config.get('url')
    except Exception as e:
        # Fallback to environment variable
        db_url = os.environ.get('PHOTOSIGHT_DATABASE_URL')
        if not db_url:
            raise ValueError(
                f"Could not get database URL from config: {e}. "
                "Set PHOTOSIGHT_DATABASE_URL environment variable."
            )
        return db_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Override the sqlalchemy.url in alembic config with PhotoSight config
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = get_database_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()