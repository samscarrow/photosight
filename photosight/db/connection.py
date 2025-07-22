"""
Database connection management for PhotoSight.

Handles SQLAlchemy engine creation, session management,
and database initialization.
"""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, Engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from .models import Base

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def configure_database(config: Dict[str, Any]) -> None:
    """
    Configure database connection from PhotoSight config.
    
    Args:
        config: PhotoSight configuration dictionary
    """
    global _engine, _session_factory
    
    db_config = config.get('database', {})
    
    if not db_config.get('enabled', False):
        logger.info("Database integration disabled")
        return
        
    database_url = db_config.get('url')
    if not database_url:
        logger.error("Database URL not configured")
        return
        
    # Create engine with connection pooling
    engine_kwargs = {
        'poolclass': QueuePool,
        'pool_size': db_config.get('pool_size', 10),
        'max_overflow': db_config.get('max_overflow', 20),
        'pool_timeout': db_config.get('pool_timeout', 30),
        'echo': db_config.get('echo_sql', False),
        'future': True  # Use SQLAlchemy 2.0 style
    }
    
    try:
        _engine = create_engine(database_url, **engine_kwargs)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
        
        # Test connection
        with _engine.connect() as conn:
            conn.execute("SELECT 1")
        
        logger.info("Database connection established successfully")
        
        # Auto-initialize if configured
        if db_config.get('auto_init', True):
            init_database()
            
    except Exception as e:
        logger.error(f"Failed to configure database: {e}")
        _engine = None
        _session_factory = None
        raise


def get_engine() -> Optional[Engine]:
    """Get the SQLAlchemy engine instance."""
    return _engine


def get_session_factory() -> Optional[sessionmaker]:
    """Get the session factory."""
    return _session_factory


@contextmanager
def get_session():
    """
    Context manager for database sessions.
    
    Yields:
        Session: SQLAlchemy session
        
    Usage:
        with get_session() as session:
            # Use session for database operations
            session.add(photo)
            session.commit()
    """
    if not _session_factory:
        raise RuntimeError("Database not configured. Call configure_database() first.")
        
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def init_database() -> None:
    """
    Initialize database schema by creating all tables.
    """
    if not _engine:
        raise RuntimeError("Database engine not available")
        
    try:
        # Create all tables
        Base.metadata.create_all(_engine)
        logger.info("Database schema initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
        raise


def drop_database() -> None:
    """
    Drop all database tables. USE WITH CAUTION!
    """
    if not _engine:
        raise RuntimeError("Database engine not available")
        
    try:
        Base.metadata.drop_all(_engine)
        logger.warning("Database schema dropped successfully")
        
    except Exception as e:
        logger.error(f"Failed to drop database schema: {e}")
        raise


def is_database_available() -> bool:
    """
    Check if database is configured and available.
    
    Returns:
        bool: True if database is available
    """
    if not _engine:
        return False
        
    try:
        with _engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


# Connection event listeners for better debugging
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set database-specific optimizations."""
    # This is primarily for PostgreSQL, but can be extended for other databases
    pass


@event.listens_for(Engine, "begin")
def do_begin(conn):
    """Log transaction begin for debugging."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Database transaction started")


@event.listens_for(Engine, "commit")
def do_commit(conn):
    """Log transaction commit for debugging."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Database transaction committed")


@event.listens_for(Engine, "rollback")
def do_rollback(conn):
    """Log transaction rollback for debugging."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Database transaction rolled back")