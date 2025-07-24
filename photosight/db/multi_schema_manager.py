"""
Multi-Schema Database Connection Manager for Oracle
Provides centralized access to different schemas with proper connection pooling.
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
import oracledb
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

class MultiSchemaManager:
    """
    Manages connections to multiple Oracle schemas with connection pooling.
    
    Schemas:
    - ADMIN: Database administration and metadata
    - PHOTOSIGHT: PhotoSight application data (photos, analysis, etc.)
    - ANALYTICS: Analytics and reporting data
    """
    
    def __init__(self, config_file: str = "config.yaml"):
        """Initialize multi-schema manager with configuration."""
        self.config_file = config_file
        self._engines: Dict[str, Any] = {}
        self._session_makers: Dict[str, sessionmaker] = {}
        self._setup_connection_pools()
    
    def _setup_connection_pools(self):
        """Set up connection pools for each schema."""
        # Set TNS_ADMIN environment variable
        wallet_location = os.getenv('ORACLE_WALLET_LOCATION', '/Users/sam/Downloads/wallet2')
        os.environ['TNS_ADMIN'] = wallet_location
        
        # Base connection parameters
        base_config = {
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'poolclass': QueuePool
        }
        
        # Schema-specific configurations
        schemas = {
            'admin': {
                'user': 'ADMIN',
                'description': 'Database administration schema'
            },
            'photosight': {
                'user': 'PHOTOSIGHT', 
                'password': 'AppUser2024!',
                'description': 'PhotoSight application schema'
            },
            'analytics': {
                'user': 'ANALYTICS',
                'password': 'DataUser2024!', 
                'description': 'Analytics and reporting schema'
            }
        }
        
        # Get connection details from environment
        service_name = os.getenv('ORACLE_SERVICE_NAME', 'photosightdb_high')
        
        for schema_name, schema_config in schemas.items():
            try:
                # Build connection string
                if schema_name == 'admin':
                    # ADMIN uses wallet authentication
                    connection_string = f'oracle+oracledb://@{service_name}?wallet_location={wallet_location}&wallet_password='
                else:
                    # Other schemas use username/password
                    user = schema_config['user']
                    password = schema_config['password']
                    connection_string = f'oracle+oracledb://{user}:{password}@{service_name}?wallet_location={wallet_location}&wallet_password='
                
                # Create engine with connection pooling
                engine = create_engine(
                    connection_string,
                    **base_config,
                    echo=False,
                    future=True
                )
                
                # Test connection
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT USER FROM DUAL"))
                    current_user = result.scalar()
                    logger.info(f"✅ Connected to {schema_name.upper()} schema as {current_user}")
                
                self._engines[schema_name] = engine
                self._session_makers[schema_name] = sessionmaker(bind=engine)
                
            except Exception as e:
                logger.error(f"❌ Failed to connect to {schema_name} schema: {e}")
                raise
    
    @contextmanager
    def get_session(self, schema: str = 'photosight'):
        """
        Get a database session for the specified schema.
        
        Args:
            schema: Schema name ('admin', 'photosight', 'analytics')
            
        Yields:
            SQLAlchemy Session object
        """
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
        """
        Get a raw database connection for the specified schema.
        
        Args:
            schema: Schema name ('admin', 'photosight', 'analytics')
            
        Yields:
            SQLAlchemy Connection object
        """
        if schema not in self._engines:
            raise ValueError(f"Unknown schema: {schema}. Available: {list(self._engines.keys())}")
        
        with self._engines[schema].connect() as conn:
            yield conn
    
    def execute_query(self, query: str, schema: str = 'photosight', params: Optional[Dict] = None) -> Any:
        """
        Execute a query on the specified schema.
        
        Args:
            query: SQL query to execute
            schema: Schema name to execute on
            params: Query parameters
            
        Returns:
            Query result
        """
        with self.get_connection(schema) as conn:
            if params:
                return conn.execute(text(query), params)
            else:
                return conn.execute(text(query))
    
    def get_table_info(self, schema: str = 'photosight') -> Dict[str, Any]:
        """
        Get information about tables in the specified schema.
        
        Args:
            schema: Schema name to inspect
            
        Returns:
            Dictionary with table information
        """
        schema_user = schema.upper()
        query = """
        SELECT 
            table_name,
            num_rows,
            blocks,
            avg_row_len,
            last_analyzed
        FROM all_tables 
        WHERE owner = :schema_name
        ORDER BY table_name
        """
        
        result = self.execute_query(query, 'admin', {'schema_name': schema_user})
        tables = {}
        
        for row in result:
            tables[row.table_name] = {
                'rows': row.num_rows or 0,
                'blocks': row.blocks or 0,
                'avg_row_length': row.avg_row_len or 0,
                'last_analyzed': row.last_analyzed
            }
        
        return tables
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get summary information for all schemas."""
        summary = {}
        
        for schema_name in self._engines.keys():
            try:
                tables = self.get_table_info(schema_name)
                total_rows = sum(t['rows'] for t in tables.values())
                
                summary[schema_name] = {
                    'status': 'connected',
                    'table_count': len(tables),
                    'total_rows': total_rows,
                    'tables': list(tables.keys())
                }
            except Exception as e:
                summary[schema_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return summary
    
    def close_all(self):
        """Close all database connections."""
        for schema_name, engine in self._engines.items():
            try:
                engine.dispose()
                logger.info(f"Closed connections for {schema_name} schema")
            except Exception as e:
                logger.error(f"Error closing {schema_name} connections: {e}")
        
        self._engines.clear()
        self._session_makers.clear()


# Global instance
_manager: Optional[MultiSchemaManager] = None

def get_multi_schema_manager() -> MultiSchemaManager:
    """Get the global multi-schema manager instance."""
    global _manager
    if _manager is None:
        _manager = MultiSchemaManager()
    return _manager

def close_connections():
    """Close all database connections."""
    global _manager
    if _manager:
        _manager.close_all()
        _manager = None