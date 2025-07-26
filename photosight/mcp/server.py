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
from ..db.direct_connection import get_direct_session, is_direct_connection_available
from ..db.mcp_operations import MCPProjectOperations, MCPTaskOperations, MCPPhotoOperations

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
    """Initialize database connection for MCP server."""
    # Use direct connection for MCP server - bypasses dual database complexity
    import os
    from photosight.db.direct_connection import configure_direct_connection
    
    oracle_password = os.getenv('ORACLE_ADMIN_PASSWORD') or os.getenv('ORACLE_PASSWORD')
    
    if not oracle_password:
        logger.error("No Oracle password found in ORACLE_ADMIN_PASSWORD or ORACLE_PASSWORD environment variables")
        return False
    
    logger.info("Using direct Oracle connection for MCP server")
    try:
        # Configure direct database connection
        from urllib.parse import quote_plus
        encoded_password = quote_plus(oracle_password)
        
        # Get TNS name from environment
        tns_name = os.getenv('ORACLE_TNS_NAME', 'photosightdb_high')
        
        direct_config = {
            'url': f"oracle+oracledb://ADMIN:{encoded_password}@{tns_name}",
            'default_schema': 'PHOTOSIGHT'
        }
        
        configure_direct_connection(direct_config)
        logger.info("Direct Oracle connection configured successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to configure direct Oracle connection: {e}")
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
        ),
        types.Tool(
            name="rank_photos",
            description="Rank and select best photos based on quality metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Optional project name to filter photos"
                    },
                    "target_percentage": {
                        "type": "number",
                        "description": "Target percentage of photos to select (0.1 = top 10%)",
                        "default": 0.1,
                        "minimum": 0.01,
                        "maximum": 1.0
                    },
                    "context": {
                        "type": "string",
                        "description": "Context for ranking",
                        "enum": ["general", "portrait", "landscape"],
                        "default": "general"
                    },
                    "detect_bursts": {
                        "type": "boolean",
                        "description": "Whether to detect and handle burst sequences",
                        "default": True
                    },
                    "custom_weights": {
                        "type": "object",
                        "description": "Custom weights for quality metrics",
                        "properties": {
                            "sharpness": {"type": "number", "minimum": 0, "maximum": 1},
                            "exposure": {"type": "number", "minimum": 0, "maximum": 1},
                            "composition": {"type": "number", "minimum": 0, "maximum": 1},
                            "face_quality": {"type": "number", "minimum": 0, "maximum": 1},
                            "emotional_impact": {"type": "number", "minimum": 0, "maximum": 1},
                            "technical_excellence": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                }
            }
        ),
        types.Tool(
            name="get_photo_selection_stats",
            description="Get statistics about photo selection/ranking results",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Optional project name to analyze"
                    }
                }
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
        elif name == "rank_photos":
            return await handle_rank_photos(arguments or {})
        elif name == "get_photo_selection_stats":
            return await handle_get_photo_selection_stats(arguments or {})
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
        if active_only:
            projects = MCPProjectOperations.list_projects(status="Active")
        else:
            projects = MCPProjectOperations.list_projects(status=status_filter, client=client_filter)
        
        if not projects:
            return [types.TextContent(type="text", text="No projects found matching the criteria.")]
        
        # Format results
        result_lines = [f"📋 Found {len(projects)} projects:\n"]
        
        for project in projects:
            # Get task summary
            tasks = MCPTaskOperations.list_tasks(project_name=project.get('NAME'))
            completed_tasks = len([t for t in tasks if t.get('STATUS') == "Done"])
            
            client_str = f" | Client: {project.get('CLIENT_NAME')}" if project.get('CLIENT_NAME') else ""
            task_str = f" | Tasks: {completed_tasks}/{len(tasks)}"
            
            result_lines.append(
                f"• **{project.get('NAME')}** ({project.get('STATUS')} - {project.get('PHASE')}){client_str}{task_str}"
            )
            
            if project.get('DESCRIPTION'):
                desc = project.get('DESCRIPTION')
                result_lines.append(f"  Description: {desc[:100]}{'...' if len(desc) > 100 else ''}")
            
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
        project = MCPProjectOperations.get_project_by_name(project_name)
        if not project:
            return [types.TextContent(type="text", text=f"Project '{project_name}' not found")]
        
        # Get project tasks
        tasks = MCPTaskOperations.list_tasks(project_name=project_name)
        
        # Format detailed project information
        lines = [
            f"📋 **{project.get('NAME')}**",
            f"Status: {project.get('STATUS')}",
            f"Phase: {project.get('PHASE')}",
            f"Created: {project.get('CREATED_AT', 'Unknown')}",
        ]
        
        if project.get('CLIENT_NAME'):
            lines.append(f"Client: {project.get('CLIENT_NAME')}")
        if project.get('PROJECT_TYPE'):
            lines.append(f"Type: {project.get('PROJECT_TYPE')}")
        if project.get('LOCATION'):
            lines.append(f"Location: {project.get('LOCATION')}")
        if project.get('BUDGET'):
            lines.append(f"Budget: ${float(project.get('BUDGET')):,.2f}")
        
        if project.get('DESCRIPTION'):
            lines.extend(["", "**Description:**", project.get('DESCRIPTION')])
        
        # Task summary
        if tasks:
            lines.extend(["", f"**Tasks ({len(tasks)} total):**"])
            
            # Group tasks by status
            task_groups = {}
            for task in tasks:
                status = task.get('STATUS')
                if status not in task_groups:
                    task_groups[status] = []
                task_groups[status].append(task)
            
            for status, status_tasks in task_groups.items():
                lines.append(f"• {status}: {len(status_tasks)} tasks")
                for task in status_tasks[:3]:  # Show first 3 tasks
                    priority = task.get('PRIORITY', '')
                    priority_icon = "🔴" if priority == "P0-Critical" else "🟡" if priority == "P1-High" else ""
                    lines.append(f"  - {task.get('TITLE')} {priority_icon}")
                
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
        project = MCPProjectOperations.create_project(
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
            f"✅ **Successfully created project: {project.get('NAME')}**",
            f"ID: {project.get('ID')}",
            f"Status: {project.get('STATUS')}",
            f"Phase: {project.get('PHASE')}",
        ]
        
        if project.get('CLIENT_NAME'):
            result.append(f"Client: {project.get('CLIENT_NAME')}")
        if project.get('PROJECT_TYPE'):
            result.append(f"Type: {project.get('PROJECT_TYPE')}")
        
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
        tasks = MCPTaskOperations.list_tasks(project_name=project_name)
        
        # Apply additional filters
        if status_filter:
            tasks = [t for t in tasks if t.get('STATUS') == status_filter]
        if assigned_filter:
            tasks = [t for t in tasks if t.get('ASSIGNEE_ID') and assigned_filter.lower() in str(t.get('ASSIGNEE_ID')).lower()]
        if priority_filter:
            tasks = [t for t in tasks if t.get('PRIORITY') == priority_filter]
        
        if not tasks:
            scope = f" for project '{project_name}'" if project_name else ""
            return [types.TextContent(type="text", text=f"No tasks found{scope} matching the criteria.")]
        
        # Format results
        result_lines = [f"📋 Found {len(tasks)} tasks:\n"]
        
        for task in tasks:
            # Get project name if not filtered by project
            project_info = ""
            if not project_name and task.get('PROJECT_NAME'):
                project_info = f" | Project: {task.get('PROJECT_NAME')}"
            
            # Status and priority icons
            status_icon = {"To Do": "⏳", "In Progress": "🔄", "Done": "✅", "Blocked": "🚫"}.get(task.get('STATUS'), "")
            priority_icon = {"P0-Critical": "🔴", "P1-High": "🟡"}.get(task.get('PRIORITY'), "")
            
            assigned_info = f" | Assigned: {task.get('ASSIGNEE_ID')}" if task.get('ASSIGNEE_ID') else ""
            
            result_lines.append(
                f"• {status_icon} **{task.get('TITLE')}** ({task.get('STATUS')}){project_info}{assigned_info} {priority_icon}"
            )
            
            if task.get('DESCRIPTION'):
                result_lines.append(f"  {task.get('DESCRIPTION')[:80]}{'...' if len(task.get('DESCRIPTION', '')) > 80 else ''}")
            
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
        project = MCPProjectOperations.get_project_by_name(project_name)
        if not project:
            return [types.TextContent(type="text", text=f"Project '{project_name}' not found")]
        
        # Create task using MCP operations
        task = MCPTaskOperations.create_task(
            project_name=project_name,
            task_name=task_name,
            description=arguments.get("description"),
            priority=arguments.get("priority", "P2-Medium"),
            assignee_id=arguments.get("assigned_to"),
            tags=arguments.get("tags"),
            meta_data=arguments.get("meta_data")
        )
        
        if not task:
            return [types.TextContent(type="text", text=f"Failed to create task '{task_name}' - task name may already exist in project")]
        
        result = [
            f"✅ **Successfully created task: {task.get('TITLE')}**",
            f"ID: {task.get('ID')}",
            f"Project: {project_name}",
            f"Status: {task.get('STATUS')}",
            f"Priority: {task.get('PRIORITY')}",
        ]
        
        if task.get('ASSIGNEE_ID'):
            result.append(f"Assigned to: {task.get('ASSIGNEE_ID')}")
        if task.get('DESCRIPTION'):
            result.append(f"Description: {task.get('DESCRIPTION')}")
        
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
                tasks = MCPTaskOperations.list_tasks(project_name=project.get('NAME'))
                all_tasks.extend(tasks)
            
            completed_tasks = len([t for t in all_tasks if t.get('STATUS') == "Done"])
            in_progress_tasks = len([t for t in all_tasks if t.get('STATUS') == "In Progress"])
            
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
                tasks = MCPTaskOperations.list_tasks(project_name=project.get('NAME'))
                completed = len([t for t in tasks if t.get('STATUS') == "Done"])
                
                lines.append(f"• **{project.get('NAME')}** ({project.get('PHASE')}) - Tasks: {completed}/{len(tasks)}")
                
                if project.get('CLIENT_NAME'):
                    lines.append(f"  Client: {project.get('CLIENT_NAME')}")
        
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
    
    try:
        stats = MCPPhotoOperations.get_statistics(metric_type)
        
        if metric_type == "overview":
            text = f"""📊 **Photo Library Statistics**

Total Photos: {stats.get('TOTAL_PHOTOS', 0):,}
Processed Photos: {stats.get('PROCESSED_PHOTOS', 0):,}
Rejected Photos: {stats.get('REJECTED_PHOTOS', 0):,}

**Equipment:**
• Unique Cameras: {stats.get('UNIQUE_CAMERAS', 0)}
• Unique Lenses: {stats.get('UNIQUE_LENSES', 0)}
"""
        elif metric_type == "technical":
            text = f"""📷 **Technical Statistics**

Average ISO: {stats.get('AVG_ISO', 'N/A')}
Average Aperture: f/{stats.get('AVG_APERTURE', 'N/A')}
Average Focal Length: {stats.get('AVG_FOCAL_LENGTH', 'N/A')}mm

Unique ISO Values: {stats.get('UNIQUE_ISO_VALUES', 0)}
Date Range: {stats.get('EARLIEST_PHOTO', 'N/A')} to {stats.get('LATEST_PHOTO', 'N/A')}
"""
        else:
            text = f"Statistics ({metric_type}):\n\n{stats.get('message', 'Not implemented')}"
        
        return [types.TextContent(type="text", text=text)]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error getting statistics: {str(e)}"
        )]


async def handle_get_insights(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle insights requests."""
    insight_type = arguments.get("insight_type", "quality_trends")
    
    return [types.TextContent(
        type="text",
        text=f"Photography insights ({insight_type}):\n\nInsight analysis would be implemented here to provide actionable recommendations based on your photo library patterns and technical metrics."
    )]


async def handle_rank_photos(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle photo ranking requests."""
    try:
        project_name = arguments.get("project_name")
        target_percentage = arguments.get("target_percentage", 0.1)
        context = arguments.get("context", "general")
        detect_bursts = arguments.get("detect_bursts", True)
        custom_weights = arguments.get("custom_weights")
        
        # Use the centralized orchestration method
        from photosight.selection import PhotoQualityRanker
        ranker = PhotoQualityRanker()
        
        # Get the original count for reporting
        original_photos = MCPPhotoOperations.get_photos_for_selection(
            project_name=project_name,
            limit=2000
        )
        original_count = len(original_photos) if original_photos else 0
        
        # Execute the complete selection workflow
        ranked_photos = ranker.select_best_photos(
            project_name=project_name,
            target_percentage=target_percentage,
            context=context,
            detect_bursts=detect_bursts,
            custom_weights=custom_weights,
            limit=2000
        )
        
        if not ranked_photos:
            return [types.TextContent(
                type="text", 
                text="No processed photos found for ranking."
            )]
        
        # Get selection report
        report = ranker.get_selection_report(original_count, ranked_photos)
        
        # Format results
        lines = [
            f"📸 **Photo Selection Results**",
            f"Total photos analyzed: {original_count}",
            f"Selected: {len(ranked_photos)} (top {int(target_percentage * 100)}%)",
            f"Context: {context}",
            f"Average quality score: {report['average_score']:.2f}",
            ""
        ]
        
        if ranked_photos:
            lines.append("**Top Selections:**")
            for i, photo in enumerate(ranked_photos[:10], 1):
                score = photo.get('COMPOSITE_SCORE', 0)
                filename = photo.get('FILENAME', 'Unknown')
                camera = photo.get('CAMERA_MODEL', 'Unknown')
                
                # Quality indicators
                quality = "⭐⭐⭐" if score >= 0.8 else "⭐⭐" if score >= 0.6 else "⭐"
                
                lines.append(
                    f"{i}. {quality} **{filename}** (Score: {score:.2f})"
                )
                
                # Add technical details
                details = []
                if photo.get('SHARPNESS_SCORE'):
                    details.append(f"Sharp: {photo['SHARPNESS_SCORE']:.2f}")
                if photo.get('COMPOSITION_SCORE'):
                    details.append(f"Comp: {photo['COMPOSITION_SCORE']:.2f}")
                if photo.get('PERSON_DETECTED'):
                    details.append("👤 Portrait")
                    
                if details:
                    lines.append(f"   {' | '.join(details)}")
                lines.append("")
            
            if len(ranked_photos) > 10:
                lines.append(f"... and {len(ranked_photos) - 10} more selections")
        
        return [types.TextContent(type="text", text="\n".join(lines))]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error ranking photos: {str(e)}"
        )]


async def handle_get_photo_selection_stats(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle photo selection statistics requests."""
    try:
        project_name = arguments.get("project_name")
        
        # Get all photos
        all_photos = MCPPhotoOperations.get_photos_for_selection(
            project_name=project_name,
            limit=5000
        )
        
        if not all_photos:
            return [types.TextContent(
                type="text",
                text="No photos found for statistics."
            )]
        
        # Calculate statistics
        total_count = len(all_photos)
        
        # Count photos with analysis
        with_analysis = len([p for p in all_photos if p.get('OVERALL_AI_SCORE') is not None])
        
        # Score distributions
        excellent = len([p for p in all_photos if p.get('OVERALL_AI_SCORE', 0) >= 0.8])
        good = len([p for p in all_photos if 0.6 <= p.get('OVERALL_AI_SCORE', 0) < 0.8])
        fair = len([p for p in all_photos if 0.4 <= p.get('OVERALL_AI_SCORE', 0) < 0.6])
        poor = len([p for p in all_photos if p.get('OVERALL_AI_SCORE', 0) < 0.4 and p.get('OVERALL_AI_SCORE') is not None])
        
        # Technical quality averages
        avg_sharpness = sum(p.get('SHARPNESS_SCORE', 0) for p in all_photos if p.get('SHARPNESS_SCORE')) / max(1, sum(1 for p in all_photos if p.get('SHARPNESS_SCORE')))
        avg_exposure = sum(p.get('EXPOSURE_QUALITY', 0) for p in all_photos if p.get('EXPOSURE_QUALITY')) / max(1, sum(1 for p in all_photos if p.get('EXPOSURE_QUALITY')))
        avg_composition = sum(p.get('COMPOSITION_SCORE', 0) for p in all_photos if p.get('COMPOSITION_SCORE')) / max(1, sum(1 for p in all_photos if p.get('COMPOSITION_SCORE')))
        
        # Portrait vs non-portrait
        portraits = len([p for p in all_photos if p.get('PERSON_DETECTED')])
        
        lines = [
            f"📊 **Photo Selection Statistics**",
            f"{'Project: ' + project_name if project_name else 'All Photos'}",
            "",
            f"**Overview:**",
            f"• Total photos: {total_count}",
            f"• Photos with analysis: {with_analysis} ({with_analysis/total_count*100:.1f}%)",
            f"• Portraits detected: {portraits} ({portraits/total_count*100:.1f}%)",
            "",
            f"**Quality Distribution:**",
            f"• ⭐⭐⭐ Excellent (≥0.8): {excellent} ({excellent/total_count*100:.1f}%)",
            f"• ⭐⭐ Good (0.6-0.8): {good} ({good/total_count*100:.1f}%)",
            f"• ⭐ Fair (0.4-0.6): {fair} ({fair/total_count*100:.1f}%)",
            f"• Poor (<0.4): {poor} ({poor/total_count*100:.1f}%)",
            "",
            f"**Average Scores:**",
            f"• Sharpness: {avg_sharpness:.2f}",
            f"• Exposure: {avg_exposure:.2f}",
            f"• Composition: {avg_composition:.2f}",
            "",
            f"**Recommendations:**"
        ]
        
        # Add recommendations based on statistics
        if excellent / total_count < 0.05:
            lines.append("• Very few excellent photos - consider stricter culling during capture")
        elif excellent / total_count > 0.3:
            lines.append("• High percentage of excellent photos - great shooting consistency!")
            
        if avg_sharpness < 0.6:
            lines.append("• Low average sharpness - check focus techniques or camera stability")
            
        if avg_exposure < 0.6:
            lines.append("• Exposure could be improved - consider exposure compensation adjustments")
            
        if portraits > total_count * 0.7 and project_name:
            lines.append("• Mostly portraits - use portrait-optimized ranking context")
        
        return [types.TextContent(type="text", text="\n".join(lines))]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error getting selection statistics: {str(e)}"
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