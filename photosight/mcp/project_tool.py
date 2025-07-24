"""
MCP Tool for project management queries and operations.

Provides natural language access to project information, task management,
and project-based analytics for AI assistants.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..db.connection import get_session
from ..db.models import Project, Task, Photo, ProjectPhoto, ProcessingRecipe, project_photos
from sqlalchemy import and_, or_, func, distinct, case
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
                query = query.filter(Project.status == filters['status'])
            
            # Apply due date filters
            if 'due_days' in filters:
                due_date = datetime.now() + timedelta(days=filters['due_days'])
                query = query.filter(Project.due_date <= due_date)
            
            if filters.get('overdue'):
                query = query.filter(
                    and_(
                        Project.due_date < datetime.now(),
                        Project.status != 'Completed'
                    )
                )
            
            # Order by created date (due_date not in model)
            query = query.order_by(Project.created_at.desc())
            
            projects = query.all()
            
            # Format results
            results = []
            for project in projects:
                # Get photo count
                photo_count = session.query(func.count(project_photos.c.photo_id)).filter(
                    project_photos.c.project_id == project.id
                ).scalar() or 0
                
                # Get task summary
                task_count = session.query(Task).filter(
                    Task.project_id == project.id
                ).count()
                
                completed_tasks = session.query(Task).filter(
                    Task.project_id == project.id,
                    Task.status == 'Done'
                ).count()
                
                results.append({
                    "name": project.name,
                    "status": project.status,
                    "phase": project.phase,
                    "priority": project.priority,
                    "description": project.description,
                    "created_at": project.created_at.isoformat() if project.created_at else None,
                    "updated_at": project.updated_at.isoformat() if project.updated_at else None,
                    "photo_count": photo_count,
                    "task_count": task_count,
                    "completed_tasks": completed_tasks,
                    "task_progress": f"{completed_tasks}/{task_count}" if task_count > 0 else "No tasks"
                })
            
            return {
                "action": "list_projects",
                "count": len(results),
                "projects": results
            }
    
    async def _get_project_status(self, project_name: str) -> Dict[str, Any]:
        """Get detailed status for a specific project."""
        with get_session() as session:
            # Get project
            project = session.query(Project).filter(
                Project.name == project_name
            ).first()
            
            if not project:
                return {
                    "error": f"Project '{project_name}' not found"
                }
            
            # Get statistics
            photo_count = session.query(func.count(project_photos.c.photo_id)).filter(
                project_photos.c.project_id == project.id
            ).scalar() or 0
            
            # Get photos with ratings (using Photo.rating if it exists, otherwise use analysis_results)
            rated_photos = session.query(
                func.coalesce(Photo.rating, 0).label('rating'),
                func.count(Photo.id).label('count')
            ).select_from(Photo).join(
                project_photos, Photo.id == project_photos.c.photo_id
            ).filter(
                project_photos.c.project_id == project.id
            ).group_by(func.coalesce(Photo.rating, 0)).all()
            
            accepted = sum(count for rating, count in rated_photos if rating and rating >= 3)
            rejected = sum(count for rating, count in rated_photos if rating and rating < 3)
            
            # Get task counts
            total_tasks = session.query(Task).filter(
                Task.project_id == project.id
            ).count()
            
            completed_tasks = session.query(Task).filter(
                Task.project_id == project.id,
                Task.status == 'Done'
            ).count()
            
            # Get recent tasks
            recent_tasks = session.query(Task).filter(
                Task.project_id == project.id
            ).order_by(Task.updated_at.desc()).limit(5).all()
            
            task_list = []
            for task in recent_tasks:
                task_list.append({
                    "name": task.name,
                    "status": task.status,
                    "priority": task.priority,
                    "assignee": task.assignee,
                    "due_date": task.due_date.isoformat() if task.due_date else None
                })
        
            return {
                "action": "project_status",
                "project": project_name,
                "status": project.status,
                "phase": project.phase,
                "priority": project.priority,
                "created_at": project.created_at.isoformat() if project.created_at else None,
                "updated_at": project.updated_at.isoformat() if project.updated_at else None,
                "completed_at": project.completed_at.isoformat() if project.completed_at else None,
                "photos": {
                    "total": photo_count,
                    "accepted": accepted,
                    "rejected": rejected,
                    "rating_distribution": [
                        {"rating": rating, "count": count}
                        for rating, count in rated_photos
                    ]
                },
                "tasks": {
                    "total": total_tasks,
                    "completed": completed_tasks,
                    "completion_rate": round(completed_tasks / total_tasks * 100, 1) if total_tasks else 0,
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
                # Map from enum values to actual status strings
                status_map = {
                    'todo': 'To Do',
                    'in_progress': 'In Progress', 
                    'review': 'Code Review',
                    'completed': 'Done',
                    'blocked': 'Blocked'
                }
                status = status_map.get(filters['task_status'], filters['task_status'])
                query = query.filter(Task.status == status)
            
            # Apply priority filter  
            if 'priority' in filters:
                # Map from enum values to actual priority strings
                priority_map = {
                    'low': 'P3-Low',
                    'medium': 'P2-Medium',
                    'high': 'P1-High',
                    'urgent': 'P0-Critical'
                }
                priority = priority_map.get(filters['priority'], filters['priority'])
                query = query.filter(Task.priority == priority)
            
            # Apply assignee filter
            if 'assigned_to' in filters:
                query = query.filter(Task.assigned_to == filters['assigned_to'])
            
            # Order by priority and due date
            # Create custom ordering for priority
            priority_order = case(
                (Task.priority == 'P0-Critical', 0),
                (Task.priority == 'P1-High', 1),
                (Task.priority == 'P2-Medium', 2),
                (Task.priority == 'P3-Low', 3),
                else_=4
            )
            
            query = query.order_by(
                priority_order,
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
                    "status": task.status,
                    "priority": task.priority,
                    "assignee": task.assigned_to,
                    "description": task.description,
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None
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
                
                # Get photo quality distribution by rating (using analysis results since Photo.rating might not exist)
                quality_stats = session.query(
                    func.coalesce(Photo.rating, 0).label('rating'),
                    func.count(Photo.id).label('count')
                ).select_from(Photo).join(
                    project_photos, Photo.id == project_photos.c.photo_id
                ).filter(
                    project_photos.c.project_id == project.id
                ).group_by(func.coalesce(Photo.rating, 0)).all()
                
                # Get photos with issues from metadata
                photos_with_issues = session.query(Photo).select_from(Photo).join(
                    project_photos, Photo.id == project_photos.c.photo_id
                ).filter(
                    project_photos.c.project_id == project.id,
                    func.coalesce(Photo.rating, 0) < 3
                ).all()
                
                issue_counts = {}
                for photo in photos_with_issues:
                    if photo.meta_data and 'quality_issues' in photo.meta_data:
                        for issue in photo.meta_data.get('quality_issues', []):
                            issue_counts[issue] = issue_counts.get(issue, 0) + 1
                
                rejection_stats = list(issue_counts.items())
                
                # Task analytics
                task_stats = session.query(
                    Task.status,
                    func.count(Task.id).label('count')  
                ).filter(
                    Task.project_id == project.id
                ).group_by(Task.status).all()
                
                # Count processed photos (with recipes) - using ProcessingRecipe table  
                processed_count = session.query(
                    func.count(distinct(ProcessingRecipe.photo_id))
                ).select_from(ProcessingRecipe).join(
                    Photo, ProcessingRecipe.photo_id == Photo.id
                ).join(
                    project_photos, Photo.id == project_photos.c.photo_id
                ).filter(
                    project_photos.c.project_id == project.id
                ).scalar() or 0
                
                total_photos = sum(count for _, count in quality_stats)
                accepted_photos = sum(count for rating, count in quality_stats if rating and rating >= 3)
                
                return {
                    "action": "project_analytics",
                    "project": project_name,
                    "photo_quality": {
                        f"rating_{rating or 0}": count 
                        for rating, count in quality_stats
                    },
                    "quality_issues": dict(rejection_stats),
                    "task_distribution": {
                        status: count for status, count in task_stats
                    },
                    "efficiency": {
                        "total_photos": total_photos,
                        "accepted_photos": accepted_photos,
                        "processed_photos": processed_count,
                        "acceptance_rate": round(accepted_photos / total_photos * 100, 1) if total_photos else 0,
                        "processing_rate": round(processed_count / total_photos * 100, 1) if total_photos else 0
                    }
                }
            else:
                # Overall project analytics
                active_projects = session.query(Project).filter(
                    Project.status == 'Active'
                ).count()
                
                completed_projects = session.query(Project).filter(
                    Project.status == 'Completed'
                ).count()
                
                # Photos by project phase
                phase_stats = session.query(
                    Project.phase,
                    func.count(Photo.id).label('photo_count')
                ).select_from(Project).join(
                    project_photos, Project.id == project_photos.c.project_id
                ).join(
                    Photo, project_photos.c.photo_id == Photo.id
                ).group_by(Project.phase).all()
                
                return {
                    "action": "project_analytics",
                    "overview": {
                        "active_projects": active_projects,
                        "completed_projects": completed_projects,
                        "total_projects": session.query(Project).count(),
                        "photos_by_phase": {
                            phase: count for phase, count in phase_stats if phase
                        }
                    }
                }
    
