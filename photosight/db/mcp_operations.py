"""
MCP-specific database operations using direct connection.

This module provides simplified database operations for the MCP server,
using the direct connection to avoid dual database complexity.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import text

from .direct_connection import get_direct_session, execute_direct_query

logger = logging.getLogger(__name__)


class MCPProjectOperations:
    """Simplified project operations for MCP server."""
    
    @staticmethod
    def list_projects(status: Optional[str] = None, client: Optional[str] = None) -> List[Dict[str, Any]]:
        """List projects with optional filtering."""
        query = """
        SELECT 
            id,
            name,
            description,
            status,
            phase,
            start_date,
            end_date,
            created_at,
            updated_at,
            priority,
            owner_id,
            meta_data
        FROM projects
        WHERE 1=1
        """
        
        params = {}
        
        if status:
            query += " AND status = :status"
            params['status'] = status
            
        # Skip client filter since client_name doesn't exist in PHOTOSIGHT schema
            
        query += " ORDER BY created_at DESC"
        
        return execute_direct_query(query, params)
    
    @staticmethod
    def get_project_by_name(name: str) -> Optional[Dict[str, Any]]:
        """Get project by name."""
        query = """
        SELECT 
            id,
            name,
            description,
            status,
            phase,
            start_date,
            end_date,
            created_at,
            updated_at,
            priority,
            owner_id,
            meta_data
        FROM projects
        WHERE name = :name
        """
        
        results = execute_direct_query(query, {'name': name})
        return results[0] if results else None
    
    @staticmethod
    def create_project(**kwargs) -> Optional[Dict[str, Any]]:
        """Create a new project."""
        query = """
        INSERT INTO projects (
            name, description, status, phase,
            start_date, end_date, priority, owner_id,
            created_at, updated_at, meta_data
        ) VALUES (
            :name, :description, :status, :phase,
            :start_date, :end_date, :priority, :owner_id,
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, :meta_data
        )
        """
        
        params = {
            'name': kwargs.get('name'),
            'description': kwargs.get('description'),
            'status': kwargs.get('status', 'Planning'),
            'phase': kwargs.get('phase', 'Capture'),
            'start_date': kwargs.get('start_date'),
            'end_date': kwargs.get('end_date'),
            'priority': kwargs.get('priority', 'P2-Medium'),
            'owner_id': kwargs.get('owner_id'),
            'meta_data': kwargs.get('meta_data')
        }
        
        try:
            execute_direct_query(query, params)
            
            # Get the created project
            return MCPProjectOperations.get_project_by_name(kwargs.get('name'))
                
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return None


class MCPTaskOperations:
    """Simplified task operations for MCP server."""
    
    @staticmethod
    def list_tasks(project_name: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List tasks with optional filtering."""
        query = """
        SELECT 
            t.id,
            t.name,
            t.description,
            t.task_type,
            t.status,
            t.priority,
            t.assigned_to,
            t.estimated_hours,
            t.actual_hours,
            t.created_at,
            t.updated_at,
            p.name as project_name,
            p.id as project_id
        FROM tasks t
        JOIN projects p ON t.project_id = p.id
        WHERE 1=1
        """
        
        params = {}
        
        if project_name:
            query += " AND p.name = :project_name"
            params['project_name'] = project_name
            
        if status:
            query += " AND t.status = :status"
            params['status'] = status
            
        query += " ORDER BY t.priority DESC, t.created_at DESC"
        
        return execute_direct_query(query, params)
    
    @staticmethod
    def create_task(project_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Create a new task for a project."""
        # First get project ID
        project = MCPProjectOperations.get_project_by_name(project_name)
        if not project:
            logger.error(f"Project not found: {project_name}")
            return None
        
        query = """
        INSERT INTO tasks (
            project_id, name, description, task_type,
            status, priority, assigned_to, estimated_hours,
            created_at, updated_at
        ) VALUES (
            :project_id, :name, :description, :task_type,
            :status, :priority, :assigned_to, :estimated_hours,
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
        )
        """
        
        params = {
            'project_id': project['ID'],
            'name': kwargs.get('task_name'),
            'description': kwargs.get('description'),
            'task_type': kwargs.get('task_type'),
            'status': 'To Do',
            'priority': kwargs.get('priority', 'P2-Medium'),
            'assigned_to': kwargs.get('assigned_to'),
            'estimated_hours': kwargs.get('estimated_hours')
        }
        
        try:
            execute_direct_query(query, params)
            
            # Return the created task
            tasks = MCPTaskOperations.list_tasks(project_name=project_name)
            for task in tasks:
                if task['NAME'] == kwargs.get('task_name'):
                    return task
            return None
            
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return None


class MCPPhotoOperations:
    """Simplified photo operations for MCP server."""
    
    @staticmethod
    def get_statistics(metric_type: str = 'overview') -> Dict[str, Any]:
        """Get photo library statistics."""
        if metric_type == 'overview':
            query = """
            SELECT 
                COUNT(*) as total_photos,
                COUNT(DISTINCT camera_make || ' ' || camera_model) as unique_cameras,
                COUNT(DISTINCT lens_model) as unique_lenses,
                SUM(CASE WHEN file_status = 'processed' THEN 1 ELSE 0 END) as processed_photos,
                SUM(CASE WHEN file_status = 'rejected' THEN 1 ELSE 0 END) as rejected_photos
            FROM photos
            """
            
            results = execute_direct_query(query)
            return results[0] if results else {
                'total_photos': 0,
                'unique_cameras': 0,
                'unique_lenses': 0,
                'processed_photos': 0,
                'rejected_photos': 0
            }
        
        elif metric_type == 'technical':
            query = """
            SELECT 
                AVG(iso) as avg_iso,
                AVG(aperture) as avg_aperture,
                AVG(focal_length) as avg_focal_length,
                COUNT(DISTINCT iso) as unique_iso_values,
                MIN(date_taken) as earliest_photo,
                MAX(date_taken) as latest_photo
            FROM photos
            WHERE iso IS NOT NULL
            """
            
            results = execute_direct_query(query)
            return results[0] if results else {}
        
        return {'message': f'Statistics for {metric_type} not yet implemented'}