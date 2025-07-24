"""
PhotoSight MCP Server Implementation

Implements the Model Context Protocol server for PhotoSight, providing
natural language query capabilities and photography insights.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    stdio_server = None
    types = None

from ..config import load_config
from ..db import configure_database, is_database_available
from ..db.operations import PhotoOperations, AnalysisOperations
from .tools import QueryTool, StatisticsTool, InsightsTool
from .project_tool import ProjectTool
from .resources import SchemaResource, MetadataResource
from .security import SecurityManager

logger = logging.getLogger(__name__)


class PhotoSightMCPServer:
    """
    MCP Server for PhotoSight database queries and insights.
    
    Provides a secure, read-only interface for AI assistants to:
    - Query photo metadata using natural language
    - Generate statistics and analytics
    - Provide photography insights and recommendations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the MCP server with PhotoSight configuration.
        
        Args:
            config_path: Path to PhotoSight config.yaml
        """
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP SDK not installed. Install with: pip install mcp")
            
        # Load PhotoSight configuration
        from pathlib import Path
        config_path = Path(config_path) if config_path else None
        self.config = load_config(config_path)
        
        # Initialize database connection
        if not self._init_database():
            raise RuntimeError("Database not available for MCP server")
            
        # Initialize security manager
        self.security = SecurityManager(self.config)
        
        # Initialize MCP server
        self.server = Server(
            name="photosight-mcp",
            version="1.0.0"
        )
        
        # Register tools
        self._register_tools()
        
        # Register resources
        self._register_resources()
        
        logger.info("PhotoSight MCP server initialized")
    
    def _init_database(self) -> bool:
        """Initialize database connection with read-only access."""
        try:
            db_config = self.config.get('database', {})
            if not db_config.get('enabled', False):
                logger.error("Database not enabled in configuration")
                return False
                
            # Configure database with read-only user if specified
            if db_config.get('mcp_server', {}).get('read_only_user'):
                # Override connection string with read-only user
                original_url = db_config['url']
                db_config['url'] = self._create_readonly_url(
                    original_url, 
                    db_config['mcp_server']['read_only_user']
                )
            
            # Configure database without auto-initialization for read-only access
            config_copy = self.config.copy()
            config_copy['database']['auto_init'] = False
            configure_database(config_copy)
            
            if not is_database_available():
                logger.error("Database connection failed")
                return False
                
            logger.info("Database connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def _create_readonly_url(self, url: str, readonly_user: str) -> str:
        """Create a read-only database URL."""
        # Parse and replace username in connection URL
        # postgresql://username:password@host:port/database
        parts = url.split('@')
        if len(parts) == 2:
            auth_part = parts[0].split('//')[-1]
            user_pass = auth_part.split(':')
            if len(user_pass) == 2:
                # Replace username, keep password
                return f"postgresql://{readonly_user}:{user_pass[1]}@{parts[1]}"
        return url
    
    def _register_tools(self):
        """Register MCP tools for photo queries and analysis."""
        
        # Query tool for natural language searches
        self.query_tool = QueryTool(self.security)
        self.server.add_tool(self.query_tool.get_tool_definition())
        
        # Statistics tool for analytics
        self.stats_tool = StatisticsTool(self.security)
        self.server.add_tool(self.stats_tool.get_tool_definition())
        
        # Insights tool for recommendations
        self.insights_tool = InsightsTool(self.security)
        self.server.add_tool(self.insights_tool.get_tool_definition())
        
        # Project tool for project management queries
        self.project_tool = ProjectTool(self.security)
        self.server.add_tool(self.project_tool.get_tool_definition())
        
        logger.info("Registered 4 MCP tools")
    
    def _register_resources(self):
        """Register MCP resources for schema and metadata access."""
        
        # Schema resource for database structure
        self.schema_resource = SchemaResource()
        for resource in self.schema_resource.get_resources():
            self.server.add_resource(resource)
        
        # Metadata resource for EXIF fields
        self.metadata_resource = MetadataResource()
        for resource in self.metadata_resource.get_resources():
            self.server.add_resource(resource)
            
        logger.info("Registered MCP resources")
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Handle incoming tool calls from AI assistants.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            # Route to appropriate tool
            if name == "query_photos":
                return await self.query_tool.execute(arguments)
            elif name == "get_statistics":
                return await self.stats_tool.execute(arguments)
            elif name == "get_insights":
                return await self.insights_tool.execute(arguments)
            elif name == "query_projects":
                return await self.project_tool.execute(arguments)
            else:
                return {"error": f"Unknown tool: {name}"}
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": str(e)}
    
    async def run(self):
        """Run the MCP server using stdio transport."""
        logger.info("Starting PhotoSight MCP server on stdio...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)
    
    def run_http(self, host: str = "localhost", port: int = 8080):
        """
        Run the MCP server using HTTP transport.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        # HTTP transport would be implemented here
        # This is a placeholder for future HTTP support
        raise NotImplementedError("HTTP transport not yet implemented")


def main():
    """Main entry point for MCP server."""
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="PhotoSight MCP Server")
    parser.add_argument('--config', help='Path to config.yaml')
    parser.add_argument('--http', action='store_true', help='Use HTTP transport')
    parser.add_argument('--host', default='localhost', help='HTTP host')
    parser.add_argument('--port', type=int, default=8080, help='HTTP port')
    
    args = parser.parse_args()
    
    try:
        server = PhotoSightMCPServer(args.config)
        
        if args.http:
            server.run_http(args.host, args.port)
        else:
            asyncio.run(server.run())
            
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise


if __name__ == "__main__":
    main()