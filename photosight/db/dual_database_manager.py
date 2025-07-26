"""
Dual Database Manager for Oracle Always-Free Architecture
Manages connections to both Projects Database (multi-schema) and Bay View Database (dedicated).
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
import oracledb
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

class ProjectsDatabaseManager:
    """
    Manages multi-schema Projects Database containing PhotoSight and other project applications.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Projects Database manager."""
        self.config = config or self._load_default_config()
        self._engines: Dict[str, Any] = {}
        self._session_makers: Dict[str, sessionmaker] = {}
        self._setup_connections()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for Projects Database."""
        wallet_location = os.getenv('PROJECTS_DB_WALLET_LOCATION', '/Users/sam/Downloads/wallet2')
        os.environ['TNS_ADMIN'] = wallet_location
        
        return {
            'wallet_location': wallet_location,
            'service_name': os.getenv('PROJECTS_DB_SERVICE_NAME', 'photosightdb_high'),
            'schemas': {
                'admin': {
                    'user': 'ADMIN',
                    'password': os.getenv('ORACLE_ADMIN_PASSWORD', os.getenv('ORACLE_PASSWORD', '')),
                    'auth_type': 'password' if os.getenv('ORACLE_ADMIN_PASSWORD') or os.getenv('ORACLE_PASSWORD') else 'wallet',
                    'description': 'Database administration'
                },
                'photosight': {
                    'user': 'PHOTOSIGHT',
                    'password': os.getenv('PHOTOSIGHT_SCHEMA_PASSWORD', 'PhotoApp2024!'),
                    'auth_type': 'password',
                    'description': 'PhotoSight application'
                },
                'analytics': {
                    'user': 'ANALYTICS', 
                    'password': os.getenv('ANALYTICS_SCHEMA_PASSWORD', 'Analytics2024!'),
                    'auth_type': 'password',
                    'description': 'Cross-project analytics'
                },
                'shared': {
                    'user': 'SHARED',
                    'password': os.getenv('SHARED_SCHEMA_PASSWORD', 'Shared2024!'),
                    'auth_type': 'password', 
                    'description': 'Shared utilities and reference data'
                }
            }
        }
    
    def _setup_connections(self):
        """Set up connection pools for each schema."""
        base_config = {
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'poolclass': QueuePool
        }
        
        for schema_name, schema_config in self.config['schemas'].items():
            try:
                # Build connection string based on auth type
                if schema_config['auth_type'] == 'wallet':
                    connection_string = f"oracle+oracledb://@{self.config['service_name']}"
                else:
                    user = schema_config['user']
                    password = schema_config['password']
                    # Use URL encoding for special characters in password
                    from urllib.parse import quote_plus
                    encoded_password = quote_plus(password)
                    connection_string = f"oracle+oracledb://{user}:{encoded_password}@{self.config['service_name']}"
                
                # Create engine
                engine = create_engine(
                    connection_string,
                    **base_config,
                    echo=False,
                    future=True
                )
                
                # Test connection
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT USER FROM DUAL"))
                    current_user = result.scalar()
                    logger.info(f"✅ Projects DB - Connected to {schema_name.upper()} schema as {current_user}")
                
                self._engines[schema_name] = engine
                self._session_makers[schema_name] = sessionmaker(bind=engine)
                
            except Exception as e:
                logger.warning(f"⚠️ Projects DB - Failed to connect to {schema_name} schema: {e}")
                # For MCP server, continue without optional schemas
                if schema_name in ['shared']:  # Optional schemas
                    logger.info(f"Continuing without optional {schema_name} schema")
                    continue
                else:
                    raise
    
    @contextmanager
    def get_session(self, schema: str = 'photosight'):
        """Get a database session for the specified schema."""
        if schema not in self._session_makers:
            raise ValueError(f"Unknown schema: {schema}. Available: {list(self._session_makers.keys())}")
        
        session = self._session_makers[schema]()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()  
            raise
        finally:
            session.close()
    
    @contextmanager  
    def get_connection(self, schema: str = 'photosight'):
        """Get a raw database connection for the specified schema."""
        if schema not in self._engines:
            raise ValueError(f"Unknown schema: {schema}. Available: {list(self._engines.keys())}")
        
        with self._engines[schema].connect() as conn:
            yield conn


class BayViewDatabaseManager:
    """
    Manages dedicated Bay View Database - kept completely separate and untouched.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Bay View Database manager.""" 
        self.config = config or self._load_default_config()
        self._engine = None
        self._session_maker = None
        self._setup_connection()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for Bay View Database."""
        wallet_location = os.getenv('BAYVIEW_DB_WALLET_LOCATION', '/Users/sam/Downloads/bayview_wallet')
        
        return {
            'wallet_location': wallet_location,
            'service_name': os.getenv('BAYVIEW_DB_SERVICE_NAME', 'bayviewdb_high'),
            'user': os.getenv('BAYVIEW_DB_USER', 'ADMIN'),
            'auth_type': 'wallet'
        }
    
    def _setup_connection(self):
        """Set up connection pool for Bay View Database."""
        try:
            # Set TNS_ADMIN for Bay View wallet
            os.environ['TNS_ADMIN'] = self.config['wallet_location']
            
            # Build connection string 
            connection_string = f"oracle+oracledb://@{self.config['service_name']}"
            
            # Create engine with dedicated resources
            self._engine = create_engine(
                connection_string,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=3600,
                poolclass=QueuePool,
                echo=False,
                future=True
            )
            
            # Test connection
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT USER FROM DUAL"))
                current_user = result.scalar()
                logger.info(f"✅ Bay View DB - Connected as {current_user}")
            
            self._session_maker = sessionmaker(bind=self._engine)
            
        except Exception as e:
            logger.error(f"❌ Bay View DB - Connection failed: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get a database session for Bay View Database."""
        session = self._session_maker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_connection(self):
        """Get a raw database connection for Bay View Database."""
        with self._engine.connect() as conn:
            yield conn


class DualDatabaseManager:
    """
    Unified manager for both Projects Database (multi-schema) and Bay View Database (dedicated).
    """
    
    def __init__(self, projects_config: Optional[Dict] = None, bayview_config: Optional[Dict] = None):
        """Initialize dual database manager."""
        self.projects_db = ProjectsDatabaseManager(projects_config)
        
        # Bay View database is optional - only initialize if configuration is available
        self.bayview_db = None
        try:
            self.bayview_db = BayViewDatabaseManager(bayview_config)
        except Exception as e:
            logger.warning(f"Bay View Database not available: {e}")
    
    # Projects Database Methods
    def get_projects_session(self, schema: str = 'photosight'):
        """Get session for Projects Database schema."""
        return self.projects_db.get_session(schema)
    
    def get_projects_connection(self, schema: str = 'photosight'):
        """Get connection for Projects Database schema."""
        return self.projects_db.get_connection(schema)
    
    # Bay View Database Methods  
    def get_bayview_session(self):
        """Get session for Bay View Database."""
        if not self.bayview_db:
            raise RuntimeError("Bay View Database not configured")
        return self.bayview_db.get_session()
    
    def get_bayview_connection(self):
        """Get connection for Bay View Database."""
        if not self.bayview_db:
            raise RuntimeError("Bay View Database not configured")
        return self.bayview_db.get_connection()
    
    # Cross-Database Operations
    def execute_cross_db_analytics(self, projects_query: str, bayview_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute analytics queries across both databases.
        
        Args:
            projects_query: Query to run on Projects Database (analytics schema)
            bayview_query: Optional query to run on Bay View Database
            
        Returns:
            Combined results from both databases
        """
        results = {}
        
        # Execute Projects Database query
        with self.get_projects_connection('analytics') as conn:
            projects_result = conn.execute(text(projects_query))
            results['projects'] = [dict(row._asdict()) for row in projects_result]
        
        # Execute Bay View Database query if provided and available
        if bayview_query and self.bayview_db:
            with self.get_bayview_connection() as conn:
                bayview_result = conn.execute(text(bayview_query))
                results['bayview'] = [dict(row._asdict()) for row in bayview_result]
        
        return results
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get summary information for both databases."""
        summary = {
            'projects_db': {
                'type': 'multi_schema',
                'schemas': {}
            }
        }
        
        # Projects Database summary
        for schema_name in self.projects_db.config['schemas'].keys():
            try:
                with self.get_projects_connection(schema_name) as conn:
                    # Get table count
                    result = conn.execute(text("SELECT COUNT(*) FROM user_tables"))
                    table_count = result.scalar()
                    
                    # Get current user
                    result = conn.execute(text("SELECT USER FROM DUAL"))
                    current_user = result.scalar()
                    
                    summary['projects_db']['schemas'][schema_name] = {
                        'status': 'connected',
                        'user': current_user,
                        'table_count': table_count
                    }
            except Exception as e:
                summary['projects_db']['schemas'][schema_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Bay View Database summary
        if self.bayview_db:
            try:
                with self.get_bayview_connection() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM user_tables"))
                    table_count = result.scalar()
                    
                    result = conn.execute(text("SELECT USER FROM DUAL"))
                    current_user = result.scalar()
                    
                    summary['bayview_db'] = {
                        'type': 'dedicated',
                        'status': 'connected',
                        'user': current_user,
                        'table_count': table_count
                    }
            except Exception as e:
                summary['bayview_db'] = {
                    'type': 'dedicated',
                    'status': 'error',
                    'error': str(e)
                }
        else:
            summary['bayview_db'] = {
                'type': 'dedicated',
                'status': 'not_configured'
            }
        
        return summary
    
    def close_all_connections(self):
        """Close all database connections."""
        # Close Projects Database connections
        for schema_name, engine in self.projects_db._engines.items():
            try:
                engine.dispose()
                logger.info(f"Closed Projects DB {schema_name} connections")
            except Exception as e:
                logger.error(f"Error closing Projects DB {schema_name}: {e}")
        
        # Close Bay View Database connections
        if self.bayview_db and self.bayview_db._engine:
            try:
                self.bayview_db._engine.dispose()
                logger.info("Closed Bay View DB connections")
            except Exception as e:
                logger.error(f"Error closing Bay View DB: {e}")


# Global instance
_dual_manager: Optional[DualDatabaseManager] = None

def get_dual_database_manager() -> DualDatabaseManager:
    """Get the global dual database manager instance."""
    global _dual_manager
    if _dual_manager is None:
        _dual_manager = DualDatabaseManager()
    return _dual_manager

def close_all_connections():
    """Close all database connections."""
    global _dual_manager
    if _dual_manager:
        _dual_manager.close_all_connections()
        _dual_manager = None