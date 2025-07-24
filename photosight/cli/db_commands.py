"""
Database CLI commands for PhotoSight

Provides command-line interface for database operations.
"""

import click
import logging
from pathlib import Path
from datetime import datetime
import json
import shutil

from ..db.models import Photo, Project, ProcessingRecipe, FileStatus
from ..db.operations import PhotoOperations
from ..db.connection import get_session as get_db_session
from ..config import load_config

logger = logging.getLogger(__name__)


@click.group()
def db():
    """Database management commands"""
    pass


@db.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--project', '-p', help='Project name to associate photos with')
@click.option('--recursive/--no-recursive', default=True, help='Recurse into subdirectories')
@click.option('--batch-size', '-b', default=50, help='Number of photos to process per batch')
@click.option('--skip-existing/--update-existing', default=True, help='Skip photos already in database')
def import_photos(directory, project, recursive, batch_size, skip_existing):
    """Import photo metadata and thumbnails into database"""
    click.echo("Photo import functionality not available in this version")
    return
    
    # # Initialize importer
    # importer = PhotoMetadataImporter(skip_existing=skip_existing)
    # 
    # # Import directory
    # start_time = datetime.now()
    # stats = importer.import_directory(
    #     Path(directory),
    #     project_name=project,
    #     recursive=recursive,
    #     batch_size=batch_size
    # )
    # 
    # # Display results
    # duration = datetime.now() - start_time
    # click.echo("\nImport Complete!")
    # click.echo(f"  Total files found: {stats['total_files']}")
    # click.echo(f"  Successfully imported: {stats['imported']}")
    # click.echo(f"  Skipped (existing): {stats['skipped']}")
    # click.echo(f"  Errors: {stats['errors']}")
    # click.echo(f"  Thumbnails generated: {stats['thumbnails_generated']}")
    # click.echo(f"  Processing time: {duration}")


@db.command()
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), default='json', 
              help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--project', '-p', help='Filter by project name')
@click.option('--include-metadata/--no-metadata', default=True, 
              help='Include extended metadata')
def export(format, output, project, include_metadata):
    """Export photo database to JSON or CSV"""
    with get_db_session() as session:
        # Build query
        query = session.query(Photo)
        
        # Filter by project if specified
        if project:
            query = query.join(Photo.projects).filter(Project.name == project)
        
        photos = query.all()
        
        if not photos:
            click.echo("No photos found to export")
            return
        
        # Prepare data
        if format == 'json':
            data = []
            for photo in photos:
                photo_dict = {
                    'id': photo.id,
                    'file_path': photo.file_path,
                    'file_name': photo.file_name,
                    'capture_date': photo.capture_date.isoformat() if photo.capture_date else None,
                    'camera_make': photo.camera_make,
                    'camera_model': photo.camera_model,
                    'lens_model': photo.lens_model,
                    'iso': photo.iso,
                    'aperture': photo.aperture,
                    'shutter_speed': photo.shutter_speed,
                    'focal_length': photo.focal_length,
                    'width': photo.width,
                    'height': photo.height,
                    'rating': photo.rating,
                    'color_label': photo.color_label,
                    'tags': photo.tags,
                    'projects': [p.name for p in photo.projects]
                }
                
                if include_metadata:
                    photo_dict['metadata'] = photo.meta_data
                
                data.append(photo_dict)
            
            # Write JSON
            if output:
                with open(output, 'w') as f:
                    json.dump(data, f, indent=2)
                click.echo(f"Exported {len(photos)} photos to {output}")
            else:
                click.echo(json.dumps(data, indent=2))
        
        else:  # CSV format
            import csv
            
            fieldnames = [
                'id', 'file_path', 'file_name', 'capture_date',
                'camera_make', 'camera_model', 'lens_model',
                'iso', 'aperture', 'shutter_speed', 'focal_length',
                'width', 'height', 'rating', 'color_label', 'projects'
            ]
            
            if output:
                with open(output, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for photo in photos:
                        writer.writerow({
                            'id': photo.id,
                            'file_path': photo.file_path,
                            'file_name': photo.file_name,
                            'capture_date': photo.capture_date.isoformat() if photo.capture_date else '',
                            'camera_make': photo.camera_make or '',
                            'camera_model': photo.camera_model or '',
                            'lens_model': photo.lens_model or '',
                            'iso': photo.iso or '',
                            'aperture': photo.aperture or '',
                            'shutter_speed': photo.shutter_speed or '',
                            'focal_length': photo.focal_length or '',
                            'width': photo.width or '',
                            'height': photo.height or '',
                            'rating': photo.rating or 0,
                            'color_label': photo.color_label or '',
                            'projects': ';'.join([p.name for p in photo.projects])
                        })
                
                click.echo(f"Exported {len(photos)} photos to {output}")
            else:
                click.echo("Please specify --output for CSV export")


@db.command()
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Backup file path (.tar.gz)')
@click.option('--include-thumbnails/--no-thumbnails', default=True,
              help='Include thumbnail files in backup')
def backup(output, include_thumbnails):
    """Backup database and thumbnails"""
    import tarfile
    import tempfile
    
    config = load_config()
    backup_path = Path(output)
    
    click.echo("Creating database backup...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Export database to JSON
        db_export_path = temp_path / "photosight_db_export.json"
        with get_db_session() as session:
            # Export all tables
            backup_data = {
                'export_date': datetime.now().isoformat(),
                'version': '1.0',
                'photos': [],
                'projects': [],
                'recipes': []
            }
            
            # Export photos
            for photo in session.query(Photo).all():
                photo_data = {
                    'id': photo.id,
                    'file_path': photo.file_path,
                    'file_name': photo.file_name,
                    'file_hash': photo.file_hash,
                    'file_size': photo.file_size,
                    'file_type': photo.file_type,
                    'capture_date': photo.capture_date.isoformat() if photo.capture_date else None,
                    'import_date': photo.import_date.isoformat() if photo.import_date else None,
                    'camera_make': photo.camera_make,
                    'camera_model': photo.camera_model,
                    'lens_model': photo.lens_model,
                    'iso': photo.iso,
                    'aperture': photo.aperture,
                    'shutter_speed': photo.shutter_speed,
                    'focal_length': photo.focal_length,
                    'width': photo.width,
                    'height': photo.height,
                    'gps_latitude': photo.gps_latitude,
                    'gps_longitude': photo.gps_longitude,
                    'gps_altitude': photo.gps_altitude,
                    'rating': photo.rating,
                    'color_label': photo.color_label,
                    'tags': photo.tags,
                    'thumbnail_path': photo.thumbnail_path,
                    'preview_path': photo.preview_path,
                    'metadata': photo.meta_data,
                    'project_ids': [p.id for p in photo.projects]
                }
                backup_data['photos'].append(photo_data)
            
            # Export projects
            for project in session.query(Project).all():
                project_data = {
                    'id': project.id,
                    'name': project.name,
                    'description': project.description,
                    'status': project.status,
                    'phase': project.phase,
                    'priority': project.priority,
                    'created_at': project.created_at.isoformat() if project.created_at else None,
                    'updated_at': project.updated_at.isoformat() if project.updated_at else None,
                    'completed_at': project.completed_at.isoformat() if project.completed_at else None,
                    'metadata': project.meta_data
                }
                backup_data['projects'].append(project_data)
            
            # Export recipes
            for recipe in session.query(ProcessingRecipe).all():
                recipe_data = {
                    'id': recipe.id,
                    'photo_id': recipe.photo_id,
                    'name': recipe.name,
                    'recipe_data': recipe.recipe_data,
                    'created_at': recipe.created_at.isoformat() if recipe.created_at else None,
                    'applied_at': recipe.applied_at.isoformat() if recipe.applied_at else None,
                    'is_favorite': recipe.is_favorite
                }
                backup_data['recipes'].append(recipe_data)
        
        # Write backup data
        with open(db_export_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        # Create tar archive
        with tarfile.open(backup_path, 'w:gz') as tar:
            # Add database export
            tar.add(db_export_path, arcname='database/photosight_db_export.json')
            
            # Add thumbnails if requested
            if include_thumbnails:
                thumbnails_dir = Path(config.get('thumbnails_dir', 'thumbnails'))
                if thumbnails_dir.exists():
                    click.echo("Including thumbnail files...")
                    tar.add(thumbnails_dir, arcname='thumbnails')
        
        click.echo(f"Backup created: {backup_path}")
        click.echo(f"  Photos: {len(backup_data['photos'])}")
        click.echo(f"  Projects: {len(backup_data['projects'])}")
        click.echo(f"  Recipes: {len(backup_data['recipes'])}")


@db.command()
def migrate():
    """Run database migrations"""
    click.echo("Running database migrations...")
    
    try:
        from alembic import command
        from alembic.config import Config
        
        # Load Alembic configuration
        alembic_cfg = Config("alembic.ini")
        
        # Run migrations
        command.upgrade(alembic_cfg, "head")
        
        click.echo("Migrations completed successfully")
        
    except Exception as e:
        click.echo(f"Migration failed: {e}", err=True)
        raise


@db.command()
@click.option('--remove-orphans/--no-remove-orphans', default=False,
              help='Remove photos with missing files')
@click.option('--rebuild-thumbnails/--no-rebuild-thumbnails', default=False,
              help='Rebuild missing thumbnails')
@click.option('--vacuum/--no-vacuum', default=False,
              help='Run VACUUM to reclaim space')
def clean(remove_orphans, rebuild_thumbnails, vacuum):
    """Clean and optimize database"""
    click.echo("Cleaning database...")
    
    stats = {
        'orphans_removed': 0,
        'thumbnails_rebuilt': 0,
        'errors': 0
    }
    
    with get_db_session() as session:
        # Check for orphaned photos
        if remove_orphans:
            click.echo("Checking for orphaned photos...")
            photos = session.query(Photo).all()
            
            for photo in photos:
                if not Path(photo.file_path).exists():
                    click.echo(f"  Removing orphan: {photo.file_name}")
                    session.delete(photo)
                    stats['orphans_removed'] += 1
            
            session.commit()
        
        # Rebuild missing thumbnails
        if rebuild_thumbnails:
            click.echo("Checking thumbnails...")
            photos = session.query(Photo).all()
            importer = PhotoMetadataImporter()
            
            for photo in photos:
                try:
                    if photo.thumbnail_path and not Path(photo.thumbnail_path).exists():
                        click.echo(f"  Rebuilding thumbnail: {photo.file_name}")
                        new_thumb = importer._generate_thumbnail(Path(photo.file_path))
                        if new_thumb:
                            photo.thumbnail_path = str(new_thumb)
                            stats['thumbnails_rebuilt'] += 1
                except Exception as e:
                    logger.error(f"Error rebuilding thumbnail for {photo.file_name}: {e}")
                    stats['errors'] += 1
            
            session.commit()
    
    # Run VACUUM
    if vacuum:
        click.echo("Running VACUUM...")
        from sqlalchemy import text
        
        with get_db_session() as session:
            # PostgreSQL automatically runs VACUUM, but we can suggest ANALYZE
            session.execute(text("ANALYZE"))
            session.commit()
    
    click.echo("\nCleanup complete:")
    click.echo(f"  Orphans removed: {stats['orphans_removed']}")
    click.echo(f"  Thumbnails rebuilt: {stats['thumbnails_rebuilt']}")
    click.echo(f"  Errors: {stats['errors']}")


@db.command()
def stats():
    """Show database statistics"""
    with get_db_session() as session:
        # Count photos
        total_photos = session.query(Photo).count()
        
        # Count by camera
        from sqlalchemy import func
        camera_stats = session.query(
            Photo.camera_model,
            func.count(Photo.id).label('count')
        ).group_by(Photo.camera_model).all()
        
        # Count by project
        project_stats = session.query(
            Project.name,
            func.count(Photo.id).label('count')
        ).join(Photo.projects).group_by(Project.name).all()
        
        # Storage stats
        total_size = session.query(func.sum(Photo.file_size)).scalar() or 0
        
        # Date range
        earliest = session.query(func.min(Photo.capture_date)).scalar()
        latest = session.query(func.max(Photo.capture_date)).scalar()
        
        click.echo("Database Statistics")
        click.echo("==================")
        click.echo(f"Total photos: {total_photos}")
        click.echo(f"Total size: {total_size / (1024**3):.2f} GB")
        
        if earliest and latest:
            click.echo(f"Date range: {earliest.date()} to {latest.date()}")
        
        click.echo("\nPhotos by Camera:")
        for camera, count in sorted(camera_stats, key=lambda x: x[1], reverse=True):
            if camera:
                click.echo(f"  {camera}: {count}")
        
        click.echo("\nPhotos by Project:")
        for project, count in sorted(project_stats, key=lambda x: x[1], reverse=True):
            click.echo(f"  {project}: {count}")


@db.command()
@click.option('--status', '-s', type=click.Choice(['local_only', 'cloud_only', 'synced', 'local_modified', 'cloud_modified', 'missing', 'conflict']), 
              help='Filter by file status')
@click.option('--machine', '-m', help='Filter by machine ID')
@click.option('--limit', '-l', default=50, help='Maximum number of results')
def sync_status(status, machine, limit):
    """Show file sync status for photos"""
    from ..db.connection import configure_database, is_database_available
    
    try:
        # Configure database
        config = load_config()
        configure_database(config)
        
        if not is_database_available():
            click.echo("Database not available", err=True)
            return
        
        # Get sync statistics
        stats = PhotoOperations.get_sync_statistics(machine_id=machine)
        
        click.echo("\n📊 Sync Statistics:")
        click.echo(f"Total photos: {stats['total_photos']}")
        click.echo(f"Photos with checksums: {stats['photos_with_checksums']} ({stats['checksum_percentage']:.1f}%)")
        
        click.echo("\nFile Status Breakdown:")
        for file_status, count in stats['status_counts'].items():
            if count > 0:
                icon = {
                    'local_only': '💻',
                    'cloud_only': '☁️',
                    'synced': '✅',
                    'local_modified': '📝',
                    'cloud_modified': '🔄',
                    'missing': '❌',
                    'conflict': '⚠️'
                }.get(file_status, '')
                click.echo(f"  {icon} {file_status}: {count}")
        
        # Show filtered photos if status specified
        if status:
            click.echo(f"\n📋 Photos with status '{status}':")
            photos = PhotoOperations.get_photos_by_sync_status(
                FileStatus(status), 
                machine_id=machine, 
                limit=limit
            )
            
            for photo in photos[:limit]:
                sync_info = f"[{photo.machine_id or 'unknown'}]" if photo.machine_id else ""
                last_sync = photo.last_sync_at.strftime('%Y-%m-%d %H:%M') if photo.last_sync_at else "never"
                click.echo(f"  • {photo.filename} {sync_info} - Last sync: {last_sync}")
                if photo.storage_path:
                    click.echo(f"    Cloud path: {photo.storage_path}")
        
        # Show recent sync activity
        if stats.get('recent_syncs'):
            click.echo("\n🔄 Recent Sync Activity:")
            for sync in stats['recent_syncs'][:5]:
                click.echo(f"  • {sync['filename']} ({sync['status']}) - {sync['last_sync'] or 'never'}")
                
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.error(f"Sync status failed: {e}")


@db.command()
@click.argument('photo_path', type=click.Path(exists=True))
@click.option('--status', '-s', required=True, 
              type=click.Choice(['local_only', 'cloud_only', 'synced', 'local_modified', 'cloud_modified', 'missing', 'conflict']),
              help='New file status')
@click.option('--machine', '-m', help='Machine ID')
@click.option('--storage-path', '-p', help='Cloud storage path')
def update_sync(photo_path, status, machine, storage_path):
    """Update sync status for a photo"""
    from ..db.connection import configure_database, is_database_available
    
    try:
        # Configure database
        config = load_config()
        configure_database(config)
        
        if not is_database_available():
            click.echo("Database not available", err=True)
            return
        
        # Find photo by path
        photo = PhotoOperations.get_photo_by_path(photo_path)
        if not photo:
            click.echo(f"Photo not found: {photo_path}", err=True)
            return
        
        # Update sync status
        success = PhotoOperations.update_file_status(
            photo.id,
            FileStatus(status),
            machine_id=machine,
            storage_path=storage_path
        )
        
        if success:
            click.echo(f"✅ Updated {photo.filename} to status '{status}'")
            if machine:
                click.echo(f"   Machine: {machine}")
            if storage_path:
                click.echo(f"   Storage: {storage_path}")
        else:
            click.echo("Failed to update sync status", err=True)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.error(f"Update sync failed: {e}")


@db.command()
@click.option('--machine', '-m', required=True, help='Machine ID to check')
def sync_conflicts(machine):
    """Show sync conflicts for a machine"""
    from ..db.connection import configure_database, is_database_available
    
    try:
        # Configure database
        config = load_config()
        configure_database(config)
        
        if not is_database_available():
            click.echo("Database not available", err=True)
            return
        
        # Get conflicts
        conflicts = PhotoOperations.get_sync_conflicts(machine)
        
        if not conflicts:
            click.echo(f"✅ No sync conflicts found for machine '{machine}'")
            return
        
        click.echo(f"\n⚠️ Found {len(conflicts)} sync conflicts for machine '{machine}':")
        
        for photo in conflicts:
            status_icon = '⚠️' if photo.file_status == FileStatus.CONFLICT else '❌'
            click.echo(f"\n{status_icon} {photo.filename}")
            click.echo(f"   Status: {photo.file_status.value}")
            click.echo(f"   Path: {photo.file_path}")
            if photo.last_sync_at:
                click.echo(f"   Last sync: {photo.last_sync_at.strftime('%Y-%m-%d %H:%M')}")
            if photo.file_modified_at:
                click.echo(f"   Modified: {photo.file_modified_at.strftime('%Y-%m-%d %H:%M')}")
                
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.error(f"Sync conflicts check failed: {e}")


@db.command()
def init():
    """Initialize database tables"""
    click.echo("Initializing database...")
    
    try:
        from ..db.connection import init_database
        init_database()
        click.echo("Database initialized successfully")
    except Exception as e:
        click.echo(f"Database initialization failed: {e}", err=True)
        raise


# # Import recipe commands
# from .recipe_commands import recipe

# Create main CLI group
@click.group()
def cli():
    """PhotoSight database management"""
    pass


cli.add_command(db)
# cli.add_command(recipe)


if __name__ == '__main__':
    cli()