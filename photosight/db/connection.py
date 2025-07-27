"""
Database connection management for PhotoSight.

Handles dual database architecture with Projects Database (multi-schema) 
and Bay View Database (dedicated). Provides unified access to PhotoSight
schema in Projects Database.
"""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, Engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool
from .models import Base
from .dual_database_manager import get_dual_database_manager

logger = logging.getLogger(__name__)

# Global database manager and legacy compatibility
_dual_manager = None
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def configure_database(config: Dict[str, Any]) -> None:
    """
    Configure database connection using dual database architecture with optimized connection pooling.
    
    Args:
        config: PhotoSight configuration dictionary
    """
    global _dual_manager, _engine, _session_factory
    
    db_config = config.get('database', {})
    
    if not db_config.get('enabled', False):
        logger.info("Database integration disabled")
        return
    
    try:
        # Initialize dual database manager with connection pooling configuration
        pool_config = db_config.get('connection_pool', {})
        _dual_manager = get_dual_database_manager(pool_config=pool_config)
        
        # For backward compatibility, set up legacy connection to PhotoSight schema
        with _dual_manager.get_projects_connection('photosight') as conn:
            # Test connection and verify pooling is working
            conn.execute(text("SELECT 1"))
        
        # Create legacy engine reference for compatibility
        _engine = _dual_manager.projects_db._engines['photosight']
        _session_factory = _dual_manager.projects_db._session_makers['photosight']
        
        logger.info("Dual database architecture configured successfully")
        logger.info("PhotoSight using Projects Database - PHOTOSIGHT schema")
        logger.info(f"Connection pool configuration: {_get_pool_info(_engine)}")
        
        # Auto-initialize if configured
        if db_config.get('auto_init', True):
            init_database()
            
    except Exception as e:
        logger.error(f"Failed to configure dual database architecture: {e}")
        _dual_manager = None
        _engine = None
        _session_factory = None
        raise


def get_engine() -> Optional[Engine]:
    """Get the SQLAlchemy engine instance."""
    return _engine


def get_session_factory() -> Optional[sessionmaker]:
    """Get the session factory."""
    return _session_factory


def get_dual_manager():
    """Get the dual database manager instance."""
    return _dual_manager


def get_projects_session(schema: str = 'photosight'):
    """
    Get a session for Projects Database schema.
    
    Args:
        schema: Schema name ('photosight', 'analytics', 'shared', 'admin')
        
    Returns:
        Context manager for database session
    """
    if not _dual_manager:
        raise RuntimeError("Database not configured. Call configure_database() first.")
    return _dual_manager.get_projects_session(schema)


def get_bayview_session():
    """
    Get a session for Bay View Database.
    
    Returns:
        Context manager for database session
    """
    if not _dual_manager:
        raise RuntimeError("Database not configured. Call configure_database() first.")
    return _dual_manager.get_bayview_session()


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
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_connection_pool_status() -> Dict[str, Any]:
    """
    Get current connection pool status for monitoring.
    
    Returns:
        Dictionary with pool statistics
    """
    if not _engine or not hasattr(_engine.pool, 'status'):
        return {'status': 'unavailable', 'message': 'No engine or pool available'}
    
    pool = _engine.pool
    
    # Different pool types have different attributes
    if hasattr(pool, 'size'):
        # QueuePool attributes
        return {
            'pool_type': pool.__class__.__name__,
            'size': getattr(pool, 'size', lambda: 'unknown')(),
            'checked_in': getattr(pool, 'checkedin', lambda: 'unknown')(),
            'checked_out': getattr(pool, 'checkedout', lambda: 'unknown')(),
            'overflow': getattr(pool, 'overflow', lambda: 'unknown')(),
            'invalid': getattr(pool, 'invalidated', lambda: 'unknown')()
        }
    else:
        return {
            'pool_type': pool.__class__.__name__,
            'status': 'available'
        }


def _get_pool_info(engine: Engine) -> str:
    """
    Get human-readable pool configuration info.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        String describing pool configuration
    """
    if not engine:
        return "No engine available"
    
    pool = engine.pool
    pool_type = pool.__class__.__name__
    
    if hasattr(pool, '_pool_size'):
        return f"{pool_type} (size: {pool._pool_size}, max_overflow: {getattr(pool, '_max_overflow', 'N/A')})"
    else:
        return f"{pool_type}"


# Connection event listeners for better debugging and optimization
@event.listens_for(Engine, "connect")
def set_database_optimizations(dbapi_connection, connection_record):
    """Set database-specific optimizations and security settings."""
    cursor = dbapi_connection.cursor()
    
    # PostgreSQL-specific optimizations
    if hasattr(dbapi_connection, 'server_version'):
        try:
            # Set connection-level optimizations for PostgreSQL
            cursor.execute("SET statement_timeout = '30s'")  # Prevent runaway queries
            cursor.execute("SET lock_timeout = '10s'")       # Prevent deadlocks
            cursor.execute("SET idle_in_transaction_session_timeout = '60s'")  # Clean up idle transactions
        except Exception as e:
            logger.debug(f"Could not set PostgreSQL optimizations: {e}")
    
    cursor.close()


@event.listens_for(Engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log connection checkout for pool monitoring."""
    if logger.isEnabledFor(logging.DEBUG):
        pool = connection_proxy._pool
        logger.debug(f"Connection checked out from pool. Pool size: {getattr(pool, 'size', lambda: 'unknown')()}")


@event.listens_for(Engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log connection checkin for pool monitoring."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Connection checked back into pool")


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