"""
MCP Tool for project management queries and operations.

Provides natural language access to project information, task management,
and project-based analytics for AI assistants.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import func, and_, or_

from ..db import get_session
from ..db.models import Project, Task, Photo, ProjectStatus, TaskStatus, TaskPriority
from ..db.operations import ProjectOperations, TaskOperations
from .security import SecurityManager

logger = logging.getLogger(__name__)


class ProjectTool:
    """
    Tool for project management queries and analytics.
    
    Allows AI assistants to:
    - Query project status and details
    - List and filter projects
    - Access task information
    - Generate project analytics
    """
    
    def __init__(self, security: SecurityManager):
        self.security = security
        
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition."""
        return {
            "name": "query_projects",
            "description": "Query photography projects, tasks, and project management information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list_projects", "project_status", "list_tasks", "project_analytics"],
                        "description": "Type of project query to perform"
                    },
                    "project_name": {
                        "type": "string",
                        "description": "Specific project name (for status or analytics)"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Additional filters for queries",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["planning", "active", "on_hold", "completed", "archived"]
                            },
                            "due_days": {
                                "type": "integer",
                                "description": "Projects due within N days"
                            },
                            "overdue": {
                                "type": "boolean",
                                "description": "Only show overdue projects"
                            },
                            "assigned_to": {
                                "type": "string",
                                "description": "Filter tasks by assignee"
                            },
                            "task_status": {
                                "type": "string",
                                "enum": ["todo", "in_progress", "review", "completed", "blocked"]
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "urgent"]
                            }
                        }
                    }
                },
                "required": ["action"]
            }
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute project management queries.
        
        Args:
            arguments: Tool arguments with action and optional filters
            
        Returns:
            Query results based on the action
        """
        action = arguments.get('action')
        project_name = arguments.get('project_name')
        filters = arguments.get('filters', {})
        
        try:
            if action == 'list_projects':
                return await self._list_projects(filters)
            elif action == 'project_status':
                if not project_name:
                    return {"error": "project_name required for project_status action"}
                return await self._get_project_status(project_name)
            elif action == 'list_tasks':
                return await self._list_tasks(project_name, filters)
            elif action == 'project_analytics':
                return await self._get_project_analytics(project_name, filters)
            else:
                return {"error": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"Project tool execution failed: {e}")
            return {"error": str(e)}
    
    async def _list_projects(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """List projects with optional filtering."""
        with get_session() as session:
            query = session.query(Project)
            
            # Apply status filter
            if 'status' in filters:
                status = ProjectStatus(filters['status'])
                query = query.filter(Project.status == status)
            
            # Apply due date filters
            if 'due_days' in filters:
                due_date = datetime.now() + timedelta(days=filters['due_days'])
                query = query.filter(Project.due_date <= due_date)
            
            if filters.get('overdue'):
                query = query.filter(
                    and_(
                        Project.due_date < datetime.now(),
                        Project.status != ProjectStatus.COMPLETED
                    )
                )
            
            # Order by due date
            query = query.order_by(Project.due_date.asc().nullsfirst())
            
            projects = query.all()
            
            # Format results
            results = []
            for project in projects:
                # Get photo count
                photo_count = session.query(Photo).filter(
                    Photo.project_id == project.id
                ).count()
                
                # Get task summary
                task_count = session.query(Task).filter(
                    Task.project_id == project.id
                ).count()
                
                completed_tasks = session.query(Task).filter(
                    Task.project_id == project.id,
                    Task.status == TaskStatus.COMPLETED
                ).count()
                
                results.append({
                    "name": project.name,
                    "status": project.status.value,
                    "client": project.client_name,
                    "type": project.project_type,
                    "shoot_date": project.shoot_date.isoformat() if project.shoot_date else None,
                    "due_date": project.due_date.isoformat() if project.due_date else None,
                    "days_until_due": (project.due_date - datetime.now()).days if project.due_date else None,
                    "photo_count": photo_count,
                    "expected_photos": project.expected_photos,
                    "task_progress": f"{completed_tasks}/{task_count}" if task_count > 0 else "No tasks"
                })
            
            return {
                "action": "list_projects",
                "count": len(results),
                "projects": results
            }
    
    async def _get_project_status(self, project_name: str) -> Dict[str, Any]:
        """Get detailed status for a specific project."""
        stats = ProjectOperations.get_project_statistics(project_name)
        
        if not stats:
            return {
                "error": f"Project '{project_name}' not found"
            }
        
        # Get recent tasks
        with get_session() as session:
            recent_tasks = session.query(Task).join(
                Project, Task.project_id == Project.id
            ).filter(
                Project.name == project_name
            ).order_by(Task.updated_at.desc()).limit(5).all()
            
            task_list = []
            for task in recent_tasks:
                task_list.append({
                    "name": task.name,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "assigned_to": task.assigned_to,
                    "due_date": task.due_date.isoformat() if task.due_date else None
                })
        
        return {
            "action": "project_status",
            "project": project_name,
            "status": stats['status'],
            "progress_percentage": stats['progress_percentage'],
            "days_until_due": stats['days_until_due'],
            "photos": {
                "total": stats['total_photos'],
                "accepted": stats['accepted_photos'],
                "rejected": stats['rejected_photos'],
                "expected": stats['expected_photos']
            },
            "tasks": {
                "total": stats['total_tasks'],
                "completed": stats['completed_tasks'],
                "completion_rate": stats['task_completion_rate'],
                "recent": task_list
            }
        }
    
    async def _list_tasks(self, project_name: Optional[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """List tasks with optional filtering."""
        with get_session() as session:
            query = session.query(Task)
            
            # Filter by project if specified
            if project_name:
                query = query.join(
                    Project, Task.project_id == Project.id
                ).filter(Project.name == project_name)
            
            # Apply status filter
            if 'task_status' in filters:
                status = TaskStatus(filters['task_status'])
                query = query.filter(Task.status == status)
            
            # Apply priority filter
            if 'priority' in filters:
                priority = TaskPriority(filters['priority'])
                query = query.filter(Task.priority == priority)
            
            # Apply assignee filter
            if 'assigned_to' in filters:
                query = query.filter(Task.assigned_to == filters['assigned_to'])
            
            # Order by priority and due date
            query = query.order_by(
                Task.priority.desc(),
                Task.due_date.asc().nullsfirst()
            )
            
            tasks = query.limit(50).all()
            
            # Format results
            results = []
            for task in tasks:
                # Get project name if not filtered by project
                project = session.query(Project).filter(
                    Project.id == task.project_id
                ).first()
                
                results.append({
                    "id": task.id,
                    "name": task.name,
                    "project": project.name if project else "Unknown",
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "assigned_to": task.assigned_to,
                    "type": task.task_type,
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                    "estimated_hours": task.estimated_hours,
                    "actual_hours": task.actual_hours
                })
            
            return {
                "action": "list_tasks",
                "project": project_name,
                "count": len(results),
                "tasks": results
            }
    
    async def _get_project_analytics(self, project_name: Optional[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Get analytics for projects."""
        with get_session() as session:
            if project_name:
                # Single project analytics
                project = session.query(Project).filter(
                    Project.name == project_name
                ).first()
                
                if not project:
                    return {"error": f"Project '{project_name}' not found"}
                
                # Get photo quality distribution
                quality_stats = session.query(
                    Photo.processing_status,
                    func.count(Photo.id).label('count')
                ).filter(
                    Photo.project_id == project.id
                ).group_by(Photo.processing_status).all()
                
                # Get rejection reasons
                rejection_stats = session.query(
                    Photo.rejection_reason,
                    func.count(Photo.id).label('count')
                ).filter(
                    Photo.project_id == project.id,
                    Photo.rejection_reason.isnot(None)
                ).group_by(Photo.rejection_reason).all()
                
                # Task analytics
                task_stats = session.query(
                    Task.status,
                    func.count(Task.id).label('count')
                ).filter(
                    Task.project_id == project.id
                ).group_by(Task.status).all()
                
                return {
                    "action": "project_analytics",
                    "project": project_name,
                    "photo_quality": {
                        status: count for status, count in quality_stats
                    },
                    "rejection_reasons": {
                        reason: count for reason, count in rejection_stats
                    },
                    "task_distribution": {
                        status.value: count for status, count in task_stats
                    },
                    "efficiency": {
                        "acceptance_rate": self._calculate_acceptance_rate(quality_stats),
                        "delivery_progress": (project.delivered_photos / project.expected_photos * 100) 
                                           if project.expected_photos else 0
                    }
                }
            else:
                # Overall project analytics
                active_projects = session.query(Project).filter(
                    Project.status == ProjectStatus.ACTIVE
                ).count()
                
                overdue_projects = session.query(Project).filter(
                    and_(
                        Project.due_date < datetime.now(),
                        Project.status != ProjectStatus.COMPLETED
                    )
                ).count()
                
                # Photos by project type
                type_stats = session.query(
                    Project.project_type,
                    func.count(Photo.id).label('photo_count')
                ).join(
                    Photo, Project.id == Photo.project_id
                ).group_by(Project.project_type).all()
                
                return {
                    "action": "project_analytics",
                    "overview": {
                        "active_projects": active_projects,
                        "overdue_projects": overdue_projects,
                        "photos_by_type": {
                            ptype: count for ptype, count in type_stats if ptype
                        }
                    }
                }
    
    def _calculate_acceptance_rate(self, quality_stats: List) -> float:
        """Calculate photo acceptance rate from quality statistics."""
        total = sum(count for _, count in quality_stats)
        accepted = sum(count for status, count in quality_stats if status == 'processed')
        return round((accepted / total * 100) if total > 0 else 0, 1)