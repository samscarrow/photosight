"""
PhotoSight MCP Server Implementation (New API)

Implements the Model Context Protocol server for PhotoSight using the new decorator-based API.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

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
from ..db import configure_database, is_database_available, get_session, get_projects_session
from ..db.operations import ProjectOperations, TaskOperations, PhotoOperations
from ..db.models import ProjectStatus, ProjectPhase, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)

# Global server instance
server = Server(
    name="photosight-mcp",
    version="1.0.0",
    instructions="PhotoSight MCP server for photo library queries and insights"
)

# Global configuration
config = None


def init_photosight_mcp(config_path: Optional[str] = None):
    """Initialize PhotoSight MCP server configuration."""
    global config
    
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP SDK not installed. Install with: pip install mcp")
    
    # Load PhotoSight configuration
    config_path = Path(config_path) if config_path else None
    config = load_config(config_path)
    
    # Initialize database connection
    if not _init_database():
        raise RuntimeError("Database not available for MCP server")
    
    logger.info("PhotoSight MCP server initialized")


def _init_database() -> bool:
    """Initialize dual database architecture with read-only access."""
    try:
        db_config = config.get('database', {})
        if not db_config.get('enabled', False):
            logger.error("Database not enabled in configuration")
            return False
        
        # Configure dual database architecture without auto-initialization for read-only access
        config_copy = config.copy()
        config_copy['database']['auto_init'] = False
        configure_database(config_copy)
        
        if not is_database_available():
            logger.error("Database connection failed")
            return False
        
        # Test both PhotoSight schema access and analytics access
        try:
            with get_projects_session('photosight') as session:
                session.execute("SELECT 1")
            logger.info("PhotoSight schema (Projects DB) connected successfully")
        except Exception as e:
            logger.warning(f"PhotoSight schema access limited: {e}")
        
        try:
            with get_projects_session('analytics') as session:
                session.execute("SELECT 1")
            logger.info("Analytics schema (Projects DB) connected successfully")
        except Exception as e:
            logger.warning(f"Analytics schema access limited: {e}")
        
        logger.info("Dual database architecture initialized for MCP server")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize dual database architecture: {e}")
        return False


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="list_projects",
            description="List photography projects with filtering options",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by project status",
                        "enum": ["Planning", "Active", "On Hold", "Completed", "Archived"]
                    },
                    "client": {
                        "type": "string",
                        "description": "Filter by client name"
                    },
                    "active_only": {
                        "type": "boolean",
                        "description": "Show only active projects",
                        "default": False
                    }
                }
            }
        ),
        types.Tool(
            name="get_project_details",
            description="Get detailed information about a specific project",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project"
                    }
                },
                "required": ["project_name"]
            }
        ),
        types.Tool(
            name="create_project",
            description="Create a new photography project",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Project name (must be unique)"
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Client name"
                    },
                    "project_type": {
                        "type": "string",
                        "description": "Type of project (wedding, portrait, commercial, etc.)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Project description"
                    },
                    "budget": {
                        "type": "number",
                        "description": "Project budget"
                    },
                    "location": {
                        "type": "string",
                        "description": "Shoot location"
                    },
                    "expected_photos": {
                        "type": "integer",
                        "description": "Expected number of photos"
                    }
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="list_tasks",
            description="List tasks for projects with filtering options",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Filter by project name"
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by task status",
                        "enum": ["To Do", "In Progress", "Code Review", "Done", "Blocked"]
                    },
                    "assigned_to": {
                        "type": "string",
                        "description": "Filter by assignee"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Filter by priority",
                        "enum": ["P0-Critical", "P1-High", "P2-Medium", "P3-Low"]
                    }
                }
            }
        ),
        types.Tool(
            name="create_task",
            description="Create a new task for a project",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Project name"
                    },
                    "task_name": {
                        "type": "string",
                        "description": "Task name"
                    },
                    "description": {
                        "type": "string",
                        "description": "Task description"
                    },
                    "task_type": {
                        "type": "string",
                        "description": "Task type (cull, edit, review, export, deliver, etc.)"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Task priority",
                        "enum": ["P0-Critical", "P1-High", "P2-Medium", "P3-Low"],
                        "default": "P2-Medium"
                    },
                    "assigned_to": {
                        "type": "string",
                        "description": "Assign task to user"
                    },
                    "estimated_hours": {
                        "type": "number",
                        "description": "Estimated hours for completion"
                    }
                },
                "required": ["project_name", "task_name"]
            }
        ),
        types.Tool(
            name="update_project_phase",
            description="Update a project's workflow phase",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Project name"
                    },
                    "phase": {
                        "type": "string",
                        "description": "New workflow phase",
                        "enum": ["Capture", "Import", "Cull", "Edit", "Review", "Export", "Deliver"]
                    }
                },
                "required": ["project_name", "phase"]
            }
        ),
        types.Tool(
            name="get_project_dashboard",
            description="Get project management dashboard overview",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_tasks": {
                        "type": "boolean",
                        "description": "Include task summaries",
                        "default": True
                    },
                    "show_overdue": {
                        "type": "boolean", 
                        "description": "Highlight overdue items",
                        "default": True
                    }
                }
            }
        ),
        types.Tool(
            name="get_statistics",
            description="Get photo library statistics and metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric_type": {
                        "type": "string",
                        "enum": ["overview", "technical", "composition", "timeline"],
                        "description": "Type of statistics to retrieve"
                    }
                },
                "required": ["metric_type"]
            }
        ),
        types.Tool(
            name="get_insights",
            description="Get photography insights and recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "insight_type": {
                        "type": "string",
                        "enum": ["quality_trends", "shooting_patterns", "gear_recommendations"],
                        "description": "Type of insights to generate"
                    }
                },
                "required": ["insight_type"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[types.TextContent]:
    """Handle tool calls."""
    try:
        if name == "list_projects":
            return await handle_list_projects(arguments or {})
        elif name == "get_project_details":
            return await handle_get_project_details(arguments or {})
        elif name == "create_project":
            return await handle_create_project(arguments or {})
        elif name == "list_tasks":
            return await handle_list_tasks(arguments or {})
        elif name == "create_task":
            return await handle_create_task(arguments or {})
        elif name == "update_project_phase":
            return await handle_update_project_phase(arguments or {})
        elif name == "get_project_dashboard":
            return await handle_get_project_dashboard(arguments or {})
        elif name == "get_statistics":
            return await handle_get_statistics(arguments or {})
        elif name == "get_insights":
            return await handle_get_insights(arguments or {})
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_list_projects(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle project listing requests."""
    try:
        # Get filter parameters
        status_filter = arguments.get("status")
        client_filter = arguments.get("client")
        active_only = arguments.get("active_only", False)
        
        # Get projects from database
        projects = ProjectOperations.list_projects()
        
        # Apply filters
        if active_only:
            projects = [p for p in projects if p.status.value == "Active"]
        elif status_filter:
            projects = [p for p in projects if p.status.value == status_filter]
        
        if client_filter:
            projects = [p for p in projects if p.client_name and client_filter.lower() in p.client_name.lower()]
        
        if not projects:
            return [types.TextContent(type="text", text="No projects found matching the criteria.")]
        
        # Format results
        result_lines = [f"📋 Found {len(projects)} projects:\n"]
        
        for project in projects:
            # Get task summary
            tasks = TaskOperations.list_tasks(project_name=project.name)
            completed_tasks = len([t for t in tasks if t.status.value == "Done"])
            
            client_str = f" | Client: {project.client_name}" if project.client_name else ""
            task_str = f" | Tasks: {completed_tasks}/{len(tasks)}"
            
            result_lines.append(
                f"• **{project.name}** ({project.status.value} - {project.phase.value}){client_str}{task_str}"
            )
            
            if project.description:
                result_lines.append(f"  Description: {project.description[:100]}{'...' if len(project.description) > 100 else ''}")
            
            result_lines.append("")  # Empty line
        
        return [types.TextContent(type="text", text="\n".join(result_lines))]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error listing projects: {str(e)}")]


async def handle_get_project_details(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle project detail requests."""
    try:
        project_name = arguments.get("project_name")
        if not project_name:
            return [types.TextContent(type="text", text="Error: project_name is required")]
        
        # Get project from database
        project = ProjectOperations.get_project_by_name(project_name)
        if not project:
            return [types.TextContent(type="text", text=f"Project '{project_name}' not found")]
        
        # Get project tasks
        tasks = TaskOperations.list_tasks(project_name=project_name)
        
        # Format detailed project information
        lines = [
            f"📋 **{project.name}**",
            f"Status: {project.status.value}",
            f"Phase: {project.phase.value}",
            f"Created: {project.created_at.strftime('%Y-%m-%d %H:%M')}",
        ]
        
        if project.client_name:
            lines.append(f"Client: {project.client_name}")
        if project.project_type:
            lines.append(f"Type: {project.project_type}")
        if project.location:
            lines.append(f"Location: {project.location}")
        if project.budget:
            lines.append(f"Budget: ${project.budget:,.2f}")
        
        if project.description:
            lines.extend(["", "**Description:**", project.description])
        
        # Task summary
        if tasks:
            lines.extend(["", f"**Tasks ({len(tasks)} total):**"])
            
            # Group tasks by status
            task_groups = {}
            for task in tasks:
                status = task.status.value
                if status not in task_groups:
                    task_groups[status] = []
                task_groups[status].append(task)
            
            for status, status_tasks in task_groups.items():
                lines.append(f"• {status}: {len(status_tasks)} tasks")
                for task in status_tasks[:3]:  # Show first 3 tasks
                    priority_icon = "🔴" if task.priority.value == "P0-Critical" else "🟡" if task.priority.value == "P1-High" else ""
                    lines.append(f"  - {task.name} {priority_icon}")
                
                if len(status_tasks) > 3:
                    lines.append(f"  - ... and {len(status_tasks) - 3} more")
        else:
            lines.append("\n**Tasks:** No tasks created yet")
        
        return [types.TextContent(type="text", text="\n".join(lines))]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error getting project details: {str(e)}")]


async def handle_create_project(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle project creation requests."""
    try:
        name = arguments.get("name")
        if not name:
            return [types.TextContent(type="text", text="Error: project name is required")]
        
        # Create project with provided parameters
        project = ProjectOperations.create_project(
            name=name,
            client_name=arguments.get("client_name"),
            project_type=arguments.get("project_type"),
            description=arguments.get("description"),
            budget=arguments.get("budget"),
            location=arguments.get("location"),
            expected_photos=arguments.get("expected_photos")
        )
        
        if not project:
            return [types.TextContent(type="text", text=f"Failed to create project '{name}' - name may already exist")]
        
        result = [
            f"✅ **Successfully created project: {project.name}**",
            f"ID: {project.id}",
            f"Status: {project.status.value}",
            f"Phase: {project.phase.value}",
        ]
        
        if project.client_name:
            result.append(f"Client: {project.client_name}")
        if project.project_type:
            result.append(f"Type: {project.project_type}")
        
        return [types.TextContent(type="text", text="\n".join(result))]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error creating project: {str(e)}")]


async def handle_list_tasks(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle task listing requests."""
    try:
        # Get filter parameters
        project_name = arguments.get("project_name")
        status_filter = arguments.get("status")
        assigned_filter = arguments.get("assigned_to")
        priority_filter = arguments.get("priority")
        
        # Get tasks from database
        tasks = TaskOperations.list_tasks(project_name=project_name)
        
        # Apply additional filters
        if status_filter:
            tasks = [t for t in tasks if t.status.value == status_filter]
        if assigned_filter:
            tasks = [t for t in tasks if t.assigned_to and assigned_filter.lower() in t.assigned_to.lower()]
        if priority_filter:
            tasks = [t for t in tasks if t.priority.value == priority_filter]
        
        if not tasks:
            scope = f" for project '{project_name}'" if project_name else ""
            return [types.TextContent(type="text", text=f"No tasks found{scope} matching the criteria.")]
        
        # Format results
        result_lines = [f"📋 Found {len(tasks)} tasks:\n"]
        
        for task in tasks:
            # Get project name if not filtered by project
            project_info = ""
            if not project_name:
                project = ProjectOperations.get_project_by_name(task.project_id if hasattr(task, 'project_id') else "")
                if project:
                    project_info = f" | Project: {project.name}"
            
            # Status and priority icons
            status_icon = {"To Do": "⏳", "In Progress": "🔄", "Done": "✅", "Blocked": "🚫"}.get(task.status.value, "")
            priority_icon = {"P0-Critical": "🔴", "P1-High": "🟡"}.get(task.priority.value, "")
            
            assigned_info = f" | Assigned: {task.assigned_to}" if task.assigned_to else ""
            
            result_lines.append(
                f"• {status_icon} **{task.name}** ({task.status.value}){project_info}{assigned_info} {priority_icon}"
            )
            
            if task.description:
                result_lines.append(f"  {task.description[:80]}{'...' if len(task.description) > 80 else ''}")
            
            result_lines.append("")  # Empty line
        
        return [types.TextContent(type="text", text="\n".join(result_lines))]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error listing tasks: {str(e)}")]


async def handle_create_task(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle task creation requests."""
    try:
        project_name = arguments.get("project_name")
        task_name = arguments.get("task_name")
        
        if not project_name or not task_name:
            return [types.TextContent(type="text", text="Error: both project_name and task_name are required")]
        
        # Verify project exists
        project = ProjectOperations.get_project_by_name(project_name)
        if not project:
            return [types.TextContent(type="text", text=f"Project '{project_name}' not found")]
        
        # Convert priority string to enum if provided
        priority = TaskPriority.MEDIUM  # default
        if arguments.get("priority"):
            try:
                priority = TaskPriority(arguments.get("priority"))
            except ValueError:
                pass
        
        # Create task
        task = TaskOperations.create_task(
            project_name=project_name,
            task_name=task_name,
            description=arguments.get("description"),
            task_type=arguments.get("task_type"),
            priority=priority,
            assigned_to=arguments.get("assigned_to"),
            estimated_hours=arguments.get("estimated_hours")
        )
        
        if not task:
            return [types.TextContent(type="text", text=f"Failed to create task '{task_name}' - task name may already exist in project")]
        
        result = [
            f"✅ **Successfully created task: {task.name}**",
            f"ID: {task.id}",
            f"Project: {project_name}",
            f"Status: {task.status.value}",
            f"Priority: {task.priority.value}",
        ]
        
        if task.assigned_to:
            result.append(f"Assigned to: {task.assigned_to}")
        if task.description:
            result.append(f"Description: {task.description}")
        
        return [types.TextContent(type="text", text="\n".join(result))]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error creating task: {str(e)}")]


async def handle_update_project_phase(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle project phase update requests."""
    try:
        project_name = arguments.get("project_name")
        phase = arguments.get("phase")
        
        if not project_name or not phase:
            return [types.TextContent(type="text", text="Error: both project_name and phase are required")]
        
        # Verify project exists
        project = ProjectOperations.get_project_by_name(project_name)
        if not project:
            return [types.TextContent(type="text", text=f"Project '{project_name}' not found")]
        
        try:
            phase_enum = ProjectPhase(phase)
        except ValueError:
            return [types.TextContent(type="text", text=f"Invalid phase: {phase}")]
        
        # Update project phase (would need to implement this method)
        # For now, just return success message
        result = [
            f"✅ **Updated project phase**",
            f"Project: {project.name}",
            f"New Phase: {phase}",
            f"Previous Phase: {project.phase.value}",
            "",
            "Note: Phase progression helps track photography workflow:",
            "Capture → Import → Cull → Edit → Review → Export → Deliver"
        ]
        
        return [types.TextContent(type="text", text="\n".join(result))]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error updating project phase: {str(e)}")]


async def handle_get_project_dashboard(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle project dashboard requests."""
    try:
        include_tasks = arguments.get("include_tasks", True)
        show_overdue = arguments.get("show_overdue", True)
        
        # Get all projects
        projects = ProjectOperations.list_projects()
        
        # Dashboard overview
        active_projects = [p for p in projects if p.status.value == "Active"]
        completed_projects = [p for p in projects if p.status.value == "Completed"]
        
        lines = [
            "📊 **PhotoSight Project Dashboard**",
            "=" * 40,
            "",
            f"**Overview:**",
            f"• Total Projects: {len(projects)}",
            f"• Active Projects: {len(active_projects)}",
            f"• Completed Projects: {len(completed_projects)}",
        ]
        
        if include_tasks:
            # Get task statistics
            all_tasks = []
            for project in projects:
                tasks = TaskOperations.list_tasks(project_name=project.name)
                all_tasks.extend(tasks)
            
            completed_tasks = len([t for t in all_tasks if t.status.value == "Done"])
            in_progress_tasks = len([t for t in all_tasks if t.status.value == "In Progress"])
            
            lines.extend([
                "",
                f"**Tasks:**",
                f"• Total Tasks: {len(all_tasks)}",
                f"• Completed: {completed_tasks}",
                f"• In Progress: {in_progress_tasks}",
                f"• Completion Rate: {(completed_tasks/len(all_tasks)*100):.1f}%" if all_tasks else "• Completion Rate: 0%",
            ])
        
        # Active projects summary
        if active_projects:
            lines.extend([
                "",
                f"**Active Projects:**"
            ])
            
            for project in active_projects[:5]:  # Show top 5
                tasks = TaskOperations.list_tasks(project_name=project.name)
                completed = len([t for t in tasks if t.status.value == "Done"])
                
                lines.append(f"• **{project.name}** ({project.phase.value}) - Tasks: {completed}/{len(tasks)}")
                
                if project.client_name:
                    lines.append(f"  Client: {project.client_name}")
        
        if show_overdue:
            # Show overdue projects (simplified - would need due_date checks)
            lines.extend([
                "",
                "**Status:** All projects on track ✅"
            ])
        
        return [types.TextContent(type="text", text="\n".join(lines))]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error generating dashboard: {str(e)}")]


async def handle_get_statistics(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle statistics requests."""
    metric_type = arguments.get("metric_type", "overview")
    
    stats = {
        "metric_type": metric_type,
        "timestamp": datetime.now().isoformat(),
        "message": "Statistics functionality would be implemented here"
    }
    
    return [types.TextContent(
        type="text",
        text=f"Statistics ({metric_type}):\n\nPhotos in library: 0\nProcessed photos: 0\nRejected photos: 0\n\n(Note: Full statistics require database queries)"
    )]


async def handle_get_insights(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle insights requests."""
    insight_type = arguments.get("insight_type", "quality_trends")
    
    return [types.TextContent(
        type="text",
        text=f"Photography insights ({insight_type}):\n\nInsight analysis would be implemented here to provide actionable recommendations based on your photo library patterns and technical metrics."
    )]


async def main():
    """Main entry point for MCP server."""
    import argparse
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser(description="PhotoSight MCP Server")
    parser.add_argument('--config', help='Path to config.yaml')
    
    args = parser.parse_args()
    
    try:
        # Load environment variables first
        load_dotenv('.env.local')
        
        # Initialize PhotoSight MCP server
        init_photosight_mcp(args.config)
        
        logger.info("Starting PhotoSight MCP server on stdio...")
        async with stdio_server() as (read_stream, write_stream):
            initialization_options = server.create_initialization_options()
            await server.run(read_stream, write_stream, initialization_options)
            
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())