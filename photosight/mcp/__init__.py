"""
PhotoSight MCP (Model Context Protocol) Server

Provides AI-powered natural language queries and insights for the PhotoSight photo database.
Implements a secure, read-only interface for AI assistants to analyze photo metadata,
generate statistics, and provide photography insights.
"""

from .server import main
from .tools import QueryTool, StatisticsTool, InsightsTool
from .resources import SchemaResource, MetadataResource

__all__ = [
    'main',
    'QueryTool',
    'StatisticsTool', 
    'InsightsTool',
    'SchemaResource',
    'MetadataResource'
]