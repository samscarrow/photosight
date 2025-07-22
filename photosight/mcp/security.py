"""
Security manager for PhotoSight MCP Server.

Ensures all database operations are read-only and validates queries
to prevent SQL injection and unauthorized access.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from sqlalchemy.sql import Select

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Manages security for MCP server database operations.
    
    Enforces:
    - Read-only access
    - Query validation
    - SQL injection prevention
    - Schema restrictions
    """
    
    # Dangerous SQL keywords that should never appear in queries
    FORBIDDEN_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE',
        'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'EXEC', 'EXECUTE',
        'CALL', 'MERGE', 'REPLACE', 'LOCK', 'UNLOCK'
    ]
    
    # Allowed tables for queries
    ALLOWED_TABLES = [
        'photos', 'analysis_results', 'processing_recipes', 'batch_sessions',
        'face_detections', 'similarity_groups', 'photo_similarities',
        'composition_analysis'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize security manager with configuration.
        
        Args:
            config: PhotoSight configuration
        """
        self.config = config
        self.max_query_limit = config.get('database', {}).get('mcp_server', {}).get('max_query_limit', 1000)
        self.query_timeout = config.get('database', {}).get('mcp_server', {}).get('query_timeout', 30)
        
    def validate_query(self, query: str) -> bool:
        """
        Validate a SQL query for safety.
        
        Args:
            query: SQL query string
            
        Returns:
            bool: True if query is safe
            
        Raises:
            SecurityError: If query contains forbidden operations
        """
        query_upper = query.upper()
        
        # Check for forbidden keywords
        for keyword in self.FORBIDDEN_KEYWORDS:
            if re.search(r'\b' + keyword + r'\b', query_upper):
                raise SecurityError(f"Forbidden operation: {keyword}")
        
        # Ensure query starts with SELECT
        if not query_upper.strip().startswith('SELECT'):
            raise SecurityError("Only SELECT queries are allowed")
        
        # Check for suspicious patterns
        if ';' in query and not query.strip().endswith(';'):
            raise SecurityError("Multiple statements not allowed")
            
        # Validate table names
        if not self._validate_tables(query):
            raise SecurityError("Query contains unauthorized tables")
            
        return True
    
    def _validate_tables(self, query: str) -> bool:
        """Check if query only accesses allowed tables."""
        # Extract table names using regex (simplified)
        # This is a basic check - production would use SQL parsing
        table_pattern = r'FROM\s+(\w+)|JOIN\s+(\w+)'
        matches = re.findall(table_pattern, query, re.IGNORECASE)
        
        for match in matches:
            table = match[0] or match[1]
            if table.lower() not in self.ALLOWED_TABLES:
                return False
                
        return True
    
    def enforce_query_limits(self, query: str) -> str:
        """
        Enforce query limits to prevent resource exhaustion.
        
        Args:
            query: SQL query string
            
        Returns:
            str: Query with enforced limits
        """
        query_upper = query.upper()
        
        # Check if query already has a LIMIT
        if 'LIMIT' not in query_upper:
            # Add default limit
            query = f"{query.rstrip(';')} LIMIT {self.max_query_limit}"
        else:
            # Ensure limit is not too high
            limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
            if limit_match:
                limit = int(limit_match.group(1))
                if limit > self.max_query_limit:
                    query = re.sub(
                        r'LIMIT\s+\d+', 
                        f'LIMIT {self.max_query_limit}', 
                        query, 
                        flags=re.IGNORECASE
                    )
        
        return query
    
    def create_read_only_session(self, session):
        """
        Configure a database session for read-only access.
        
        Args:
            session: SQLAlchemy session
        """
        # Set session to read-only
        session.execute(text("SET TRANSACTION READ ONLY"))
        
        # Set statement timeout
        session.execute(text(f"SET statement_timeout = '{self.query_timeout}s'"))
    
    def sanitize_user_input(self, value: Any) -> Any:
        """
        Sanitize user input to prevent SQL injection.
        
        Args:
            value: User input value
            
        Returns:
            Sanitized value
        """
        if isinstance(value, str):
            # Escape single quotes
            return value.replace("'", "''")
        return value
    
    def validate_natural_language_query(self, query: str) -> bool:
        """
        Validate natural language query for safety.
        
        Args:
            query: Natural language query
            
        Returns:
            bool: True if query is safe
        """
        query_lower = query.lower()
        
        # Check for attempts to modify data
        modify_patterns = [
            'delete', 'remove', 'update', 'change', 'modify',
            'drop', 'create', 'insert', 'add new'
        ]
        
        for pattern in modify_patterns:
            if pattern in query_lower:
                logger.warning(f"Rejected modification attempt: {pattern}")
                return False
                
        return True
    
    def log_query(self, query: str, user: Optional[str] = None):
        """
        Log queries for audit trail.
        
        Args:
            query: SQL query
            user: User identifier (if available)
        """
        logger.info(f"MCP Query executed - User: {user or 'unknown'}, Query: {query[:200]}")


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass