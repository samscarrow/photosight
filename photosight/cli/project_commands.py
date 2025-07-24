"""
PhotoSight project management CLI commands.
Implements photography workflow management with project/task tracking.
"""

import click
from datetime import datetime, timedelta
from typing import Optional, List
from tabulate import tabulate
import json

from photosight.config import load_config
from photosight.db.connection import configure_database, is_database_available
from photosight.db.operations import ProjectOperations, TaskOperations
from photosight.db.models import ProjectStatus, ProjectPhase, TaskStatus, TaskPriority


@click.group(name='project')
@click.pass_context
def project_group(ctx):
    """Photography project management commands."""
    if ctx.obj is None:
        ctx.obj = {}
    
    # Initialize database connection
    config = load_config()
    db_config = config.get('database', {})
    if not db_config.get('enabled', False):
        click.echo("❌ Database not enabled. Enable database in config.yaml", err=True)
        ctx.exit(1)
    
    configure_database(config)
    if not is_database_available():
        click.echo("❌ Database connection failed", err=True)
        ctx.exit(1)
    
    ctx.obj['project_ops'] = ProjectOperations
    ctx.obj['task_ops'] = TaskOperations


@project_group.command('create')
@click.argument('name')
@click.option('--client', '-c', help='Client name')
@click.option('--type', '-t', 'project_type', help='Project type (wedding, portrait, commercial, etc.)')
@click.option('--description', '-d', help='Project description')
@click.option('--shoot-date', type=click.DateTime(['%Y-%m-%d']), help='Shoot date (YYYY-MM-DD)')
@click.option('--due-date', type=click.DateTime(['%Y-%m-%d']), help='Due date (YYYY-MM-DD)')
@click.option('--budget', type=float, help='Project budget')
@click.option('--location', '-l', help='Shoot location')
@click.pass_context
def create_project(ctx, name: str, client: Optional[str] = None, project_type: Optional[str] = None, 
                  description: Optional[str] = None, shoot_date: Optional[datetime] = None,
                  due_date: Optional[datetime] = None, budget: Optional[float] = None,
                  location: Optional[str] = None):
    """Create a new photography project."""
    
    project_ops = ctx.obj['project_ops']
    
    try:
        project_data = {
            'name': name,
            'client_name': client,
            'project_type': project_type or 'general',
            'description': description,
            'shoot_date': shoot_date,
            'due_date': due_date,
            'budget': budget,
            'location': location,
            'status': ProjectStatus.PLANNING,
            'phase': ProjectPhase.CAPTURE
        }
        
        project = project_ops.create_project(**project_data)
        
        click.echo(f"✅ Created project '{name}' (ID: {project.id})")
        if client:
            click.echo(f"   Client: {client}")
        if shoot_date:
            click.echo(f"   Shoot Date: {shoot_date.strftime('%Y-%m-%d')}")
        if due_date:
            click.echo(f"   Due Date: {due_date.strftime('%Y-%m-%d')}")
            
    except Exception as e:
        click.echo(f"❌ Error creating project: {e}", err=True)


@project_group.command('list')
@click.option('--status', type=click.Choice([s.value for s in ProjectStatus]), help='Filter by status')
@click.option('--client', '-c', help='Filter by client name')
@click.option('--active', '-a', is_flag=True, help='Show only active projects')
@click.option('--overdue', is_flag=True, help='Show overdue projects')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def list_projects(ctx, status: Optional[str] = None, client: Optional[str] = None,
                 active: bool = False, overdue: bool = False, output_json: bool = False):
    """List photography projects."""
    
    project_ops = ctx.obj['project_ops']
    
    try:
        # Build filters
        filters = {}
        if status:
            filters['status'] = ProjectStatus(status)
        if client:
            filters['client_name'] = client
        if active:
            filters['status'] = ProjectStatus.ACTIVE
            
        projects = project_ops.get_projects(**filters)
        
        # Filter overdue if requested
        if overdue:
            now = datetime.now()
            projects = [p for p in projects if p.due_date and p.due_date < now and p.status != ProjectStatus.COMPLETED]
        
        if not projects:
            click.echo("No projects found.")
            return
            
        if output_json:
            # JSON output for programmatic use
            project_data = []
            for project in projects:
                project_data.append({
                    'id': project.id,
                    'name': project.name,
                    'client': project.client_name,
                    'status': project.status.value,
                    'phase': project.phase.value,
                    'due_date': project.due_date.isoformat() if project.due_date else None,
                    'created': project.created_at.isoformat()
                })
            click.echo(json.dumps(project_data, indent=2))
        else:
            # Table output for human viewing
            headers = ['ID', 'Name', 'Client', 'Status', 'Phase', 'Due Date', 'Tasks']
            rows = []
            
            for project in projects:
                # Get task count
                tasks = project_ops.get_project_tasks(project.id)
                completed_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
                total_tasks = len(tasks)
                
                due_str = ""
                if project.due_date:
                    due_str = project.due_date.strftime('%Y-%m-%d')
                    if project.due_date < datetime.now() and project.status != ProjectStatus.COMPLETED:
                        due_str = f"⚠️  {due_str}"
                
                task_str = f"{completed_tasks}/{total_tasks}" if total_tasks > 0 else "0"
                
                rows.append([
                    project.id,
                    project.name[:30] + ('...' if len(project.name) > 30 else ''),
                    project.client_name or '-',
                    project.status.value,
                    project.phase.value,
                    due_str,
                    task_str
                ])
            
            click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
            click.echo(f"\nTotal: {len(projects)} project(s)")
            
    except Exception as e:
        click.echo(f"❌ Error listing projects: {e}", err=True)


@project_group.command('show')
@click.argument('project_id', type=int)
@click.option('--tasks', '-t', is_flag=True, help='Show project tasks')
@click.option('--photos', '-p', is_flag=True, help='Show project photos')
@click.pass_context
def show_project(ctx, project_id: int, tasks: bool = False, photos: bool = False):
    """Show detailed project information."""
    
    project_ops = ctx.obj['project_ops']
    
    try:
        project = project_ops.get_project(project_id)
        if not project:
            click.echo(f"❌ Project {project_id} not found", err=True)
            return
        
        # Project header
        click.echo(f"\n📋 Project: {project.name}")
        click.echo("=" * (len(project.name) + 11))
        
        # Basic info
        click.echo(f"ID: {project.id}")
        click.echo(f"Status: {project.status.value}")
        click.echo(f"Phase: {project.phase.value}")
        if project.client_name:
            click.echo(f"Client: {project.client_name}")
        if project.project_type:
            click.echo(f"Type: {project.project_type}")
        if project.location:
            click.echo(f"Location: {project.location}")
        
        # Dates
        click.echo(f"\n📅 Timeline:")
        click.echo(f"Created: {project.created_at.strftime('%Y-%m-%d %H:%M')}")
        if project.shoot_date:
            click.echo(f"Shoot Date: {project.shoot_date.strftime('%Y-%m-%d')}")
        if project.due_date:
            due_str = project.due_date.strftime('%Y-%m-%d')
            if project.due_date < datetime.now() and project.status != ProjectStatus.COMPLETED:
                due_str += " ⚠️ OVERDUE"
            click.echo(f"Due Date: {due_str}")
        if project.completed_at:
            click.echo(f"Completed: {project.completed_at.strftime('%Y-%m-%d %H:%M')}")
        
        # Description and notes
        if project.description:
            click.echo(f"\n📝 Description:")
            click.echo(project.description)
        
        if project.notes:
            click.echo(f"\n🗒️  Notes:")
            click.echo(project.notes)
        
        # Budget
        if project.budget:
            click.echo(f"\n💰 Budget: ${project.budget:,.2f}")
        
        # Photo counts
        if project.expected_photos or project.delivered_photos:
            click.echo(f"\n📸 Photos:")
            if project.expected_photos:
                click.echo(f"Expected: {project.expected_photos}")
            click.echo(f"Delivered: {project.delivered_photos or 0}")
        
        # Tasks summary
        project_tasks = project_ops.get_project_tasks(project_id)
        if project_tasks:
            click.echo(f"\n✅ Tasks: {len(project_tasks)} total")
            status_counts = {}
            for task in project_tasks:
                status_counts[task.status] = status_counts.get(task.status, 0) + 1
            
            for status, count in status_counts.items():
                click.echo(f"  {status.value}: {count}")
        
        # Detailed task list if requested
        if tasks and project_tasks:
            click.echo(f"\n📋 Task Details:")
            click.echo("-" * 50)
            
            for task in project_tasks:
                status_icon = "✅" if task.status == TaskStatus.COMPLETED else "⏳"
                priority_icon = "🔴" if task.priority == TaskPriority.CRITICAL else "🟡" if task.priority == TaskPriority.HIGH else ""
                
                click.echo(f"{status_icon} [{task.id}] {task.name} {priority_icon}")
                if task.description:
                    click.echo(f"    {task.description}")
                if task.due_date:
                    due_str = task.due_date.strftime('%Y-%m-%d')
                    if task.due_date < datetime.now() and task.status != TaskStatus.COMPLETED:
                        due_str += " ⚠️"
                    click.echo(f"    Due: {due_str}")
                click.echo(f"    Status: {task.status.value} | Priority: {task.priority.value}")
                click.echo()
        
        # Photo list if requested
        if photos:
            # This would require implementing photo-project association queries
            click.echo(f"\n📸 Photos: Feature coming soon")
            
    except Exception as e:
        click.echo(f"❌ Error showing project: {e}", err=True)


@project_group.command('update')
@click.argument('project_id', type=int)
@click.option('--name', help='Update project name')
@click.option('--status', type=click.Choice([s.value for s in ProjectStatus]), help='Update status')
@click.option('--phase', type=click.Choice([p.value for p in ProjectPhase]), help='Update phase')
@click.option('--client', help='Update client name')
@click.option('--due-date', type=click.DateTime(['%Y-%m-%d']), help='Update due date')
@click.option('--notes', help='Update notes')
@click.pass_context
def update_project(ctx, project_id: int, **kwargs):
    """Update project information."""
    
    project_ops = ctx.obj['project_ops']
    
    try:
        # Remove None values
        updates = {k: v for k, v in kwargs.items() if v is not None}
        
        # Convert enum strings to enum values
        if 'status' in updates:
            updates['status'] = ProjectStatus(updates['status'])
        if 'phase' in updates:
            updates['phase'] = ProjectPhase(updates['phase'])
        
        if not updates:
            click.echo("❌ No updates specified", err=True)
            return
        
        success = project_ops.update_project(project_id, **updates)
        
        if success:
            click.echo(f"✅ Updated project {project_id}")
            for key, value in updates.items():
                click.echo(f"   {key}: {value}")
        else:
            click.echo(f"❌ Project {project_id} not found", err=True)
            
    except Exception as e:
        click.echo(f"❌ Error updating project: {e}", err=True)


@project_group.command('delete')
@click.argument('project_id', type=int)
@click.option('--force', is_flag=True, help='Delete without confirmation')
@click.pass_context
def delete_project(ctx, project_id: int, force: bool = False):
    """Delete a project and all its tasks."""
    
    project_ops = ctx.obj['project_ops']
    
    try:
        project = project_ops.get_project(project_id)
        if not project:
            click.echo(f"❌ Project {project_id} not found", err=True)
            return
        
        # Confirmation
        if not force:
            click.confirm(
                f"Delete project '{project.name}' and all its tasks? This cannot be undone.",
                abort=True
            )
        
        success = project_ops.delete_project(project_id)
        
        if success:
            click.echo(f"✅ Deleted project '{project.name}' (ID: {project_id})")
        else:
            click.echo(f"❌ Failed to delete project {project_id}", err=True)
            
    except Exception as e:
        click.echo(f"❌ Error deleting project: {e}", err=True)


# Task management commands
@project_group.group('task')
@click.pass_context
def task_group(ctx):
    """Task management commands."""
    pass


@task_group.command('create')
@click.argument('project_id', type=int)
@click.argument('name')
@click.option('--description', '-d', help='Task description')
@click.option('--type', '-t', 'task_type', help='Task type (cull, edit, review, export, deliver)')
@click.option('--priority', type=click.Choice([p.value for p in TaskPriority]), 
              default='P2-Medium', help='Task priority')
@click.option('--assigned-to', help='Assign task to user')
@click.option('--due-date', type=click.DateTime(['%Y-%m-%d']), help='Due date (YYYY-MM-DD)')
@click.option('--estimated-hours', type=float, help='Estimated hours')
@click.option('--depends-on', type=int, help='Task ID this depends on')
@click.pass_context
def create_task(ctx, project_id: int, name: str, **kwargs):
    """Create a new task within a project."""
    
    task_ops = ctx.obj['task_ops']
    project_ops = ctx.obj['project_ops']
    
    try:
        # Verify project exists
        project = project_ops.get_project(project_id)
        if not project:
            click.echo(f"❌ Project {project_id} not found", err=True)
            return
        
        # Prepare task data
        task_data = {k.replace('_', '_'): v for k, v in kwargs.items() if v is not None}
        task_data['project_id'] = project_id
        task_data['name'] = name
        
        # Convert priority string to enum
        if 'priority' in task_data:
            task_data['priority'] = TaskPriority(task_data['priority'])
        else:
            task_data['priority'] = TaskPriority.MEDIUM
        
        # Set default task type based on project phase
        if not task_data.get('task_type'):
            phase_to_task = {
                ProjectPhase.CAPTURE: 'capture',
                ProjectPhase.IMPORT: 'import',
                ProjectPhase.CULL: 'cull',
                ProjectPhase.EDIT: 'edit',
                ProjectPhase.REVIEW: 'review',
                ProjectPhase.EXPORT: 'export',
                ProjectPhase.DELIVER: 'deliver'
            }
            task_data['task_type'] = phase_to_task.get(project.phase, 'general')
        
        task = task_ops.create_task(**task_data)
        
        click.echo(f"✅ Created task '{name}' (ID: {task.id}) in project '{project.name}'")
        click.echo(f"   Priority: {task.priority.value}")
        if task.due_date:
            click.echo(f"   Due: {task.due_date.strftime('%Y-%m-%d')}")
        if task.assigned_to:
            click.echo(f"   Assigned to: {task.assigned_to}")
            
    except Exception as e:
        click.echo(f"❌ Error creating task: {e}", err=True)


@task_group.command('list')
@click.argument('project_id', type=int, required=False)
@click.option('--status', type=click.Choice([s.value for s in TaskStatus]), help='Filter by status')
@click.option('--assigned-to', help='Filter by assignee')
@click.option('--overdue', is_flag=True, help='Show overdue tasks')
@click.option('--my-tasks', is_flag=True, help='Show tasks assigned to you')
@click.pass_context
def list_tasks(ctx, project_id: Optional[int] = None, status: Optional[str] = None,
               assigned_to: Optional[str] = None, overdue: bool = False, my_tasks: bool = False):
    """List tasks (for a project or all projects)."""
    
    task_ops = ctx.obj['task_ops']
    project_ops = ctx.obj['project_ops']
    
    try:
        # Build filters
        filters = {}
        if project_id:
            filters['project_id'] = project_id
        if status:
            filters['status'] = TaskStatus(status)
        if assigned_to:
            filters['assigned_to'] = assigned_to
        if my_tasks:
            # Get current user (could be from config or env)
            import getpass
            filters['assigned_to'] = getpass.getuser()
        
        tasks = task_ops.get_tasks(**filters)
        
        # Filter overdue if requested
        if overdue:
            now = datetime.now()
            tasks = [t for t in tasks if t.due_date and t.due_date < now and t.status != TaskStatus.COMPLETED]
        
        if not tasks:
            click.echo("No tasks found.")
            return
        
        # Table output
        headers = ['ID', 'Project', 'Task', 'Status', 'Priority', 'Assigned', 'Due Date']
        rows = []
        
        for task in tasks:
            # Get project name
            project = project_ops.get_project(task.project_id)
            project_name = project.name[:15] + ('...' if len(project.name) > 15 else '') if project else 'Unknown'
            
            due_str = ""
            if task.due_date:
                due_str = task.due_date.strftime('%m-%d')
                if task.due_date < datetime.now() and task.status != TaskStatus.COMPLETED:
                    due_str = f"⚠️ {due_str}"
            
            status_icon = {
                TaskStatus.TODO: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.REVIEW: "👀", 
                TaskStatus.COMPLETED: "✅",
                TaskStatus.BLOCKED: "🚫"
            }.get(task.status, "")
            
            priority_icon = {
                TaskPriority.CRITICAL: "🔴",
                TaskPriority.HIGH: "🟡",
                TaskPriority.MEDIUM: "",
                TaskPriority.LOW: "🟢"
            }.get(task.priority, "")
            
            rows.append([
                task.id,
                project_name,
                task.name[:25] + ('...' if len(task.name) > 25 else ''),
                f"{status_icon} {task.status.value}",
                f"{priority_icon} {task.priority.value}",
                task.assigned_to or '-',
                due_str
            ])
        
        click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
        click.echo(f"\nTotal: {len(tasks)} task(s)")
        
    except Exception as e:
        click.echo(f"❌ Error listing tasks: {e}", err=True)


@task_group.command('update')
@click.argument('task_id', type=int)
@click.option('--status', type=click.Choice([s.value for s in TaskStatus]), help='Update status')
@click.option('--priority', type=click.Choice([p.value for p in TaskPriority]), help='Update priority')
@click.option('--assigned-to', help='Update assignee')
@click.option('--notes', help='Update notes')
@click.option('--actual-hours', type=float, help='Actual hours spent')
@click.pass_context
def update_task(ctx, task_id: int, **kwargs):
    """Update task information."""
    
    task_ops = ctx.obj['task_ops']
    
    try:
        # Remove None values and convert enums
        updates = {k.replace('_', '_'): v for k, v in kwargs.items() if v is not None}
        
        if 'status' in updates:
            updates['status'] = TaskStatus(updates['status'])
            # Auto-set timestamps based on status
            if updates['status'] == TaskStatus.IN_PROGRESS:
                updates['started_at'] = datetime.now()
            elif updates['status'] == TaskStatus.COMPLETED:
                updates['completed_at'] = datetime.now()
                
        if 'priority' in updates:
            updates['priority'] = TaskPriority(updates['priority'])
        
        if not updates:
            click.echo("❌ No updates specified", err=True)
            return
        
        success = task_ops.update_task(task_id, **updates)
        
        if success:
            click.echo(f"✅ Updated task {task_id}")
            for key, value in updates.items():
                if key not in ['started_at', 'completed_at']:  # Don't show auto-set timestamps
                    click.echo(f"   {key}: {value}")
        else:
            click.echo(f"❌ Task {task_id} not found", err=True)
            
    except Exception as e:
        click.echo(f"❌ Error updating task: {e}", err=True)


# Photography workflow shortcuts
@project_group.command('workflow')
@click.argument('project_id', type=int)
@click.option('--phase', type=click.Choice([p.value for p in ProjectPhase]), help='Set project phase')
@click.option('--create-tasks', is_flag=True, help='Create standard tasks for the phase')
@click.pass_context
def workflow(ctx, project_id: int, phase: Optional[str] = None, create_tasks: bool = False):
    """Manage photography project workflow phases."""
    
    project_ops = ctx.obj['project_ops']
    task_ops = ctx.obj['task_ops']
    
    try:
        project = project_ops.get_project(project_id)
        if not project:
            click.echo(f"❌ Project {project_id} not found", err=True)
            return
        
        if phase:
            # Update project phase
            phase_enum = ProjectPhase(phase)
            project_ops.update_project(project_id, phase=phase_enum)
            click.echo(f"✅ Updated project '{project.name}' to phase: {phase}")
            
            if create_tasks:
                # Create standard tasks for the phase
                phase_tasks = {
                    ProjectPhase.CAPTURE: [
                        ("Equipment Check", "Verify all camera gear, batteries, memory cards"),
                        ("Location Scout", "Scout and prepare shooting locations"),
                        ("Client Briefing", "Review shot list and expectations with client")
                    ],
                    ProjectPhase.IMPORT: [
                        ("Import Photos", "Import all photos from memory cards"),
                        ("Backup Creation", "Create secure backups of original files"),
                        ("Initial Sort", "Basic file organization and naming")
                    ],
                    ProjectPhase.CULL: [
                        ("Technical Cull", "Remove duplicates, blurry, and technically poor shots"),
                        ("Creative Cull", "Select best shots for client review"),
                        ("Client Preview", "Prepare preview gallery for client selection")
                    ],
                    ProjectPhase.EDIT: [
                        ("Basic Corrections", "Exposure, color, and lens corrections"),
                        ("Creative Editing", "Artistic adjustments and style application"),
                        ("Final Review", "Quality check all edited images")
                    ],
                    ProjectPhase.REVIEW: [
                        ("Client Review", "Present edited images to client"),
                        ("Revision Requests", "Handle any requested changes"),
                        ("Final Approval", "Get client sign-off on final images")
                    ],
                    ProjectPhase.EXPORT: [
                        ("High-Res Export", "Export full resolution final images"),
                        ("Web Gallery", "Create online gallery for client"),
                        ("Print Prep", "Prepare images for print if needed")
                    ],
                    ProjectPhase.DELIVER: [
                        ("Package Delivery", "Deliver final images to client"),
                        ("Invoice", "Send final invoice"),
                        ("Project Archive", "Archive project files and close project")
                    ]
                }
                
                tasks_for_phase = phase_tasks.get(phase_enum, [])
                created_count = 0
                
                for task_name, task_desc in tasks_for_phase:
                    try:
                        task_ops.create_task(
                            project_id=project_id,
                            name=task_name,
                            description=task_desc,
                            task_type=phase.lower(),
                            status=TaskStatus.TODO,
                            priority=TaskPriority.MEDIUM
                        )
                        created_count += 1
                    except:
                        # Task might already exist, skip
                        pass
                
                if created_count > 0:
                    click.echo(f"✅ Created {created_count} standard tasks for {phase} phase")
        else:
            # Show current workflow status
            click.echo(f"\n📋 Project Workflow: {project.name}")
            click.echo("=" * (len(project.name) + 20))
            click.echo(f"Current Phase: {project.phase.value}")
            click.echo(f"Status: {project.status.value}")
            
            # Show phase progression
            phases = list(ProjectPhase)
            current_index = phases.index(project.phase)
            
            click.echo("\n📊 Phase Progress:")
            for i, p in enumerate(phases):
                if i < current_index:
                    click.echo(f"  ✅ {p.value}")
                elif i == current_index:
                    click.echo(f"  🔄 {p.value} (CURRENT)")
                else:
                    click.echo(f"  ⏳ {p.value}")
            
            # Show tasks for current phase
            tasks = task_ops.get_tasks(project_id=project_id)
            phase_tasks = [t for t in tasks if t.task_type == project.phase.value.lower()]
            
            if phase_tasks:
                click.echo(f"\n📋 {project.phase.value} Tasks:")
                for task in phase_tasks:
                    status_icon = "✅" if task.status == TaskStatus.COMPLETED else "⏳"
                    click.echo(f"  {status_icon} {task.name}")
            
    except Exception as e:
        click.echo(f"❌ Error managing workflow: {e}", err=True)


# Add this to photosight CLI main
def add_project_commands(cli):
    """Add project management commands to the main CLI."""
    cli.add_command(project_group)