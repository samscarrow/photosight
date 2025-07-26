#!/usr/bin/env python3
"""
PhotoSight Command Line Interface

Main CLI entry point for PhotoSight photo management and ranking system.
Provides comprehensive commands for photo analysis, ranking, selection, and automation.
"""

import sys
import click
import logging
from pathlib import Path
from typing import Optional

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from photosight.config import load_config
from photosight.cli.db_commands import db
from photosight.cli.project_commands import project_group


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.pass_context
def main(ctx, config: Optional[str] = None, verbose: bool = False, quiet: bool = False):
    """
    PhotoSight - Intelligent photo management and ranking system
    
    A comprehensive CLI for managing photography workflows, from import and analysis
    to ranking, selection, and organization. Designed for professional photographers
    and serious enthusiasts who need efficient photo management tools.
    """
    
    # Ensure context object exists
    if ctx.obj is None:
        ctx.obj = {}
    
    # Configure logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Load configuration
    try:
        if config:
            ctx.obj['config'] = load_config(config)
        else:
            ctx.obj['config'] = load_config()
    except Exception as e:
        if not quiet:
            click.echo(f"Warning: Could not load config: {e}", err=True)
        ctx.obj['config'] = {}
    
    # Store CLI options in context
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet


@main.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--project', '-p', help='Project name to associate photos with')
@click.option('--top-n', '-n', default=50, help='Number of top photos to select')
@click.option('--threshold', '-t', type=float, default=0.7, 
              help='Quality threshold (0.0-1.0)')
@click.option('--output-dir', '-o', type=click.Path(file_okay=False, dir_okay=True),
              help='Output directory for selected photos')
@click.option('--copy/--move', default=True, help='Copy or move selected photos')
@click.option('--dry-run', is_flag=True, help='Show what would be done without doing it')
@click.option('--recursive/--no-recursive', default=True, help='Process subdirectories')
@click.pass_context
def rank(ctx, directory: str, project: Optional[str] = None, top_n: int = 50,
         threshold: float = 0.7, output_dir: Optional[str] = None, 
         copy: bool = True, dry_run: bool = False, recursive: bool = True):
    """
    Rank photos in a directory and select the best ones.
    
    Analyzes photos using AI-powered quality assessment, composition analysis,
    and technical metrics to automatically rank and select the best photos
    from a directory.
    
    DIRECTORY: Path to directory containing photos to rank
    """
    
    from photosight.ranking.quality_ranker import QualityRanker
    from photosight.utils.file_organizer import PhotoOrganizer
    
    config = ctx.obj.get('config', {})
    verbose = ctx.obj.get('verbose', False)
    quiet = ctx.obj.get('quiet', False)
    
    if not quiet:
        click.echo(f"🔍 Analyzing photos in: {directory}")
        if project:
            click.echo(f"📁 Project: {project}")
        click.echo(f"🎯 Selecting top {top_n} photos with threshold {threshold}")
    
    try:
        # Initialize ranker
        ranker = QualityRanker(config)
        
        # Process directory
        directory_path = Path(directory)
        photo_files = []
        
        # Find photo files
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.raw', '.cr2', '.nef', '.arw'}
        
        if recursive:
            for ext in extensions:
                photo_files.extend(directory_path.rglob(f"*{ext}"))
                photo_files.extend(directory_path.rglob(f"*{ext.upper()}"))
        else:
            for ext in extensions:
                photo_files.extend(directory_path.glob(f"*{ext}"))
                photo_files.extend(directory_path.glob(f"*{ext.upper()}"))
        
        if not photo_files:
            click.echo("❌ No photos found in directory", err=True)
            return
        
        if not quiet:
            click.echo(f"📸 Found {len(photo_files)} photos")
        
        # Rank photos with progress bar
        with click.progressbar(photo_files, label="Ranking photos") as bar:
            rankings = []
            for photo_path in bar:
                try:
                    score = ranker.rank_photo(photo_path)
                    rankings.append((photo_path, score))
                    if verbose:
                        click.echo(f"  {photo_path.name}: {score:.3f}")
                except Exception as e:
                    if verbose:
                        click.echo(f"  Error processing {photo_path.name}: {e}")
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and top-n
        selected = []
        for path, score in rankings:
            if score >= threshold and len(selected) < top_n:
                selected.append((path, score))
        
        if not selected:
            click.echo("❌ No photos meet the quality threshold", err=True)
            return
        
        if not quiet:
            click.echo(f"\n✅ Selected {len(selected)} photos:")
            for i, (path, score) in enumerate(selected[:10], 1):  # Show top 10
                click.echo(f"  {i:2d}. {path.name} (score: {score:.3f})")
            if len(selected) > 10:
                click.echo(f"     ... and {len(selected) - 10} more")
        
        # Organize selected photos
        if output_dir:
            output_path = Path(output_dir)
            action = "copy" if copy else "move"
            
            if not dry_run:
                output_path.mkdir(parents=True, exist_ok=True)
                
                if not quiet:
                    click.echo(f"\n📂 {action.title()}ing selected photos to: {output_path}")
                
                # Create selects subdirectory
                selects_dir = output_path / "selects"
                selects_dir.mkdir(exist_ok=True)
                
                success_count = 0
                for path, score in selected:
                    try:
                        dest_path = selects_dir / f"{score:.3f}_{path.name}"
                        
                        if copy:
                            import shutil
                            shutil.copy2(path, dest_path)
                        else:
                            path.rename(dest_path)
                        
                        success_count += 1
                        if verbose:
                            click.echo(f"  ✅ {action}: {path.name} -> {dest_path.name}")
                            
                    except Exception as e:
                        click.echo(f"  ❌ Error {action}ing {path.name}: {e}", err=True)
                
                if not quiet:
                    click.echo(f"\n🎉 Successfully {action}ed {success_count}/{len(selected)} photos")
            else:
                if not quiet:
                    click.echo(f"\n🔍 DRY RUN - Would {action} {len(selected)} photos to: {output_path}/selects/")
                    for path, score in selected:
                        dest_name = f"{score:.3f}_{path.name}"
                        click.echo(f"  {action}: {path.name} -> selects/{dest_name}")
        
        # Save ranking results to file
        if not dry_run:
            results_file = directory_path / "photosight_rankings.json"
            import json
            
            results = {
                'timestamp': click.datetime.now().isoformat(),
                'directory': str(directory_path),
                'project': project,
                'total_photos': len(photo_files),
                'threshold': threshold,
                'top_n': top_n,
                'selected_count': len(selected),
                'rankings': [
                    {
                        'file': str(path),
                        'score': score,
                        'rank': i + 1
                    }
                    for i, (path, score) in enumerate(rankings)
                ]
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            if not quiet:
                click.echo(f"💾 Ranking results saved to: {results_file}")
        
    except Exception as e:
        click.echo(f"❌ Error during ranking: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--criteria', '-c', multiple=True, 
              type=click.Choice(['blur', 'exposure', 'composition', 'faces', 'technical']),
              help='Selection criteria (can be used multiple times)')
@click.option('--faces-min', type=int, default=1, help='Minimum number of faces required')
@click.option('--blur-threshold', type=float, default=100.0, help='Minimum sharpness score')
@click.option('--exposure-tolerance', type=float, default=0.3, help='Exposure tolerance (0.0-1.0)')
@click.option('--output-format', type=click.Choice(['table', 'json', 'csv']), default='table',
              help='Output format for results')
@click.pass_context
def select(ctx, directory: str, criteria: tuple, faces_min: int, blur_threshold: float,
           exposure_tolerance: float, output_format: str):
    """
    Advanced photo selection with multiple criteria.
    
    Provides granular control over photo selection using specific technical
    and artistic criteria. Useful for curating photos based on specific
    requirements like portraits, landscapes, or technical specifications.
    
    DIRECTORY: Path to directory containing photos to analyze
    """
    
    from photosight.selection.advanced_selector import AdvancedSelector
    from photosight.analysis.technical_analyzer import TechnicalAnalyzer
    
    config = ctx.obj.get('config', {})
    verbose = ctx.obj.get('verbose', False)
    quiet = ctx.obj.get('quiet', False)
    
    if not criteria:
        criteria = ['blur', 'exposure', 'composition']  # Default criteria
    
    if not quiet:
        click.echo(f"🔍 Analyzing photos in: {directory}")
        click.echo(f"📋 Selection criteria: {', '.join(criteria)}")
    
    try:
        # Initialize components
        selector = AdvancedSelector(config)
        analyzer = TechnicalAnalyzer(config)
        
        directory_path = Path(directory)
        
        # Find photos
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.raw', '.cr2', '.nef', '.arw'}
        photo_files = []
        
        for ext in extensions:
            photo_files.extend(directory_path.rglob(f"*{ext}"))
            photo_files.extend(directory_path.rglob(f"*{ext.upper()}"))
        
        if not photo_files:
            click.echo("❌ No photos found in directory", err=True)
            return
        
        if not quiet:
            click.echo(f"📸 Found {len(photo_files)} photos")
        
        # Analyze photos
        results = []
        with click.progressbar(photo_files, label="Analyzing photos") as bar:
            for photo_path in bar:
                try:
                    analysis = analyzer.analyze_photo(photo_path)
                    
                    # Apply selection criteria
                    selected = True
                    reasons = []
                    
                    if 'blur' in criteria:
                        if analysis.get('sharpness', 0) < blur_threshold:
                            selected = False
                            reasons.append(f"Too blurry ({analysis.get('sharpness', 0):.1f} < {blur_threshold})")
                    
                    if 'exposure' in criteria:
                        exposure_score = analysis.get('exposure_quality', 0.5)
                        if abs(exposure_score - 0.5) > exposure_tolerance:
                            selected = False
                            reasons.append(f"Poor exposure ({exposure_score:.2f})")
                    
                    if 'faces' in criteria:
                        face_count = analysis.get('face_count', 0)
                        if face_count < faces_min:
                            selected = False
                            reasons.append(f"Not enough faces ({face_count} < {faces_min})")
                    
                    if 'composition' in criteria:
                        composition_score = analysis.get('composition_score', 0)
                        if composition_score < 0.6:  # Configurable threshold
                            selected = False
                            reasons.append(f"Poor composition ({composition_score:.2f})")
                    
                    if 'technical' in criteria:
                        technical_score = analysis.get('technical_quality', 0)
                        if technical_score < 0.7:  # Configurable threshold
                            selected = False
                            reasons.append(f"Technical issues ({technical_score:.2f})")
                    
                    results.append({
                        'file': photo_path.name,
                        'path': str(photo_path),
                        'selected': selected,
                        'reasons': reasons if not selected else ['Meets all criteria'],
                        'analysis': analysis
                    })
                    
                except Exception as e:
                    if verbose:
                        click.echo(f"  Error analyzing {photo_path.name}: {e}")
        
        # Output results
        selected_photos = [r for r in results if r['selected']]
        rejected_photos = [r for r in results if not r['selected']]
        
        if not quiet:
            click.echo(f"\n📊 Analysis Results:")
            click.echo(f"  Selected: {len(selected_photos)}")
            click.echo(f"  Rejected: {len(rejected_photos)}")
            click.echo(f"  Total: {len(results)}")
        
        # Format output
        if output_format == 'json':
            import json
            click.echo(json.dumps(results, indent=2))
        
        elif output_format == 'csv':
            click.echo("file,selected,reasons,sharpness,exposure_quality,face_count,composition_score")
            for result in results:
                analysis = result['analysis']
                reasons = '; '.join(result['reasons'])
                click.echo(f"{result['file']},{result['selected']},\"{reasons}\","
                          f"{analysis.get('sharpness', 0):.1f},"
                          f"{analysis.get('exposure_quality', 0):.2f},"
                          f"{analysis.get('face_count', 0)},"
                          f"{analysis.get('composition_score', 0):.2f}")
        
        else:  # table format
            from tabulate import tabulate
            
            if selected_photos:
                click.echo("\n✅ Selected Photos:")
                table_data = []
                for result in selected_photos:
                    analysis = result['analysis']
                    table_data.append([
                        result['file'][:30] + ('...' if len(result['file']) > 30 else ''),
                        f"{analysis.get('sharpness', 0):.1f}",
                        f"{analysis.get('exposure_quality', 0):.2f}",
                        analysis.get('face_count', 0),
                        f"{analysis.get('composition_score', 0):.2f}"
                    ])
                
                headers = ['File', 'Sharpness', 'Exposure', 'Faces', 'Composition']
                click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
            
            if rejected_photos and verbose:
                click.echo("\n❌ Rejected Photos:")
                table_data = []
                for result in rejected_photos[:10]:  # Show first 10
                    table_data.append([
                        result['file'][:30] + ('...' if len(result['file']) > 30 else ''),
                        result['reasons'][0][:40] + ('...' if len(result['reasons'][0]) > 40 else '')
                    ])
                
                headers = ['File', 'Reason']
                click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
                
                if len(rejected_photos) > 10:
                    click.echo(f"... and {len(rejected_photos) - 10} more rejected photos")
        
    except Exception as e:
        click.echo(f"❌ Error during selection: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('source_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('dest_dir', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--structure', type=click.Choice(['date', 'camera', 'project', 'rating']), 
              default='date', help='Organization structure')
@click.option('--copy/--move', default=True, help='Copy or move files')
@click.option('--dry-run', is_flag=True, help='Show what would be done without doing it')
@click.option('--create-selects/--no-selects', default=True, 
              help='Create separate selects folder for top-rated photos')
@click.option('--selects-threshold', type=float, default=0.8, 
              help='Rating threshold for selects folder')
@click.pass_context
def organize(ctx, source_dir: str, dest_dir: str, structure: str, copy: bool, 
             dry_run: bool, create_selects: bool, selects_threshold: float):
    """
    Organize photos into structured directories.
    
    Automatically organizes photos based on various criteria like date,
    camera model, project, or rating. Creates a clean directory structure
    and optionally separates the best photos into a "selects" folder.
    
    SOURCE_DIR: Source directory containing photos to organize
    DEST_DIR: Destination directory for organized photos
    """
    
    from photosight.utils.file_organizer import PhotoOrganizer
    
    config = ctx.obj.get('config', {})
    verbose = ctx.obj.get('verbose', False)
    quiet = ctx.obj.get('quiet', False)
    
    if not quiet:
        click.echo(f"📁 Organizing photos from: {source_dir}")
        click.echo(f"📂 Destination: {dest_dir}")
        click.echo(f"🏗️  Structure: {structure}")
        click.echo(f"⚙️  Action: {'copy' if copy else 'move'}")
        if create_selects:
            click.echo(f"⭐ Creating selects folder (threshold: {selects_threshold})")
    
    try:
        organizer = PhotoOrganizer(config)
        
        source_path = Path(source_dir)
        dest_path = Path(dest_dir)
        
        # Find photos
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.raw', '.cr2', '.nef', '.arw'}
        photo_files = []
        
        for ext in extensions:
            photo_files.extend(source_path.rglob(f"*{ext}"))
            photo_files.extend(source_path.rglob(f"*{ext.upper()}"))
        
        if not photo_files:
            click.echo("❌ No photos found in source directory", err=True)
            return
        
        if not quiet:
            click.echo(f"📸 Found {len(photo_files)} photos")
        
        # Organize photos
        results = organizer.organize_photos(
            photo_files=photo_files,
            dest_dir=dest_path,
            structure=structure,
            copy=copy,
            dry_run=dry_run,
            create_selects=create_selects,
            selects_threshold=selects_threshold
        )
        
        if not quiet:
            click.echo(f"\n📊 Organization Results:")
            click.echo(f"  Processed: {results['processed']}")
            click.echo(f"  Organized: {results['organized']}")
            click.echo(f"  Errors: {results['errors']}")
            if create_selects:
                click.echo(f"  Selects: {results.get('selects', 0)}")
            
            if results['errors'] > 0 and verbose:
                click.echo("\n❌ Errors:")
                for error in results.get('error_details', []):
                    click.echo(f"  {error}")
        
    except Exception as e:
        click.echo(f"❌ Error during organization: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--format', type=click.Choice(['table', 'json', 'summary']), default='summary',
              help='Output format')
@click.option('--include-exif/--no-exif', default=False, help='Include detailed EXIF data')
@click.option('--include-analysis/--no-analysis', default=True, help='Include quality analysis')
@click.option('--group-by', type=click.Choice(['camera', 'lens', 'date', 'project']),
              help='Group statistics by field')
@click.pass_context
def stats(ctx, directory: str, format: str, include_exif: bool, include_analysis: bool, group_by: Optional[str]):
    """
    Generate comprehensive statistics for photo collections.
    
    Analyzes photo collections to provide insights about shooting patterns,
    equipment usage, technical settings, and quality metrics. Useful for
    understanding your photography habits and improving workflow.
    
    DIRECTORY: Path to directory containing photos to analyze
    """
    
    from photosight.analysis.stats_generator import StatsGenerator
    
    config = ctx.obj.get('config', {})
    verbose = ctx.obj.get('verbose', False)
    quiet = ctx.obj.get('quiet', False)
    
    if not quiet:
        click.echo(f"📊 Generating statistics for: {directory}")
    
    try:
        stats_gen = StatsGenerator(config)
        directory_path = Path(directory)
        
        # Find photo files
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.raw', '.cr2', '.nef', '.arw'}
        photo_files = []
        
        for ext in extensions:
            photo_files.extend(directory_path.rglob(f"*{ext}"))
            photo_files.extend(directory_path.rglob(f"*{ext.upper()}"))
        
        if not photo_files:
            click.echo("❌ No photos found in directory", err=True)
            return
        
        if not quiet:
            click.echo(f"📸 Found {len(photo_files)} photos")
        
        # Generate statistics
        with click.progressbar(photo_files, label="Analyzing photos") as bar:
            stats_data = stats_gen.generate_stats(
                photo_files, 
                include_exif=include_exif,
                include_analysis=include_analysis,
                group_by=group_by,
                progress_callback=lambda current, total: bar.update(1)
            )
        
        # Output results
        if format == 'json':
            import json
            click.echo(json.dumps(stats_data, indent=2, default=str))
        
        elif format == 'table':
            from tabulate import tabulate
            # Convert stats to table format and display
            stats_gen.display_table_stats(stats_data)
        
        else:  # summary format
            if not quiet:
                click.echo(f"\n📸 Photo Collection Summary")
                click.echo("=" * 30)
                
                collection_info = stats_data.get('collection_info', {})
                click.echo(f"Total Photos: {collection_info.get('total_photos', 0)}")
                
                file_stats = stats_data.get('file_stats', {})
                if file_stats.get('date_range'):
                    date_range = file_stats['date_range']
                    click.echo(f"Date Range: {date_range['earliest'][:10]} to {date_range['latest'][:10]}")
                
                total_mb = collection_info.get('total_size_mb', 0)
                click.echo(f"Total Size: {total_mb / 1024:.2f} GB")
                
                exif_stats = stats_data.get('exif_stats', {})
                if exif_stats.get('top_cameras'):
                    click.echo(f"\n📷 Top Cameras:")
                    for camera, count in list(exif_stats['top_cameras'].items())[:5]:
                        click.echo(f"  {camera}: {count} photos")
                
                if exif_stats.get('top_lenses'):
                    click.echo(f"\n🔍 Top Lenses:")
                    for lens, count in list(exif_stats['top_lenses'].items())[:5]:
                        click.echo(f"  {lens}: {count} photos")
                
                if exif_stats.get('iso_stats'):
                    iso = exif_stats['iso_stats']
                    click.echo(f"\n⚙️  Technical Averages:")
                    click.echo(f"  ISO: {iso['avg']:.0f}")
                    
                if exif_stats.get('aperture_stats'):
                    aperture = exif_stats['aperture_stats']
                    click.echo(f"  Aperture: f/{aperture['avg']:.1f}")
                    
                if exif_stats.get('focal_length_stats'):
                    focal = exif_stats['focal_length_stats']
                    click.echo(f"  Focal Length: {focal['avg']:.0f}mm")
                
                quality_stats = stats_data.get('quality_stats', {})
                if quality_stats and not quality_stats.get('error'):
                    click.echo(f"\n⭐ Quality Metrics:")
                    click.echo(f"  Average Quality: {quality_stats.get('average_quality', 0):.3f}")
                    distribution = quality_stats.get('quality_distribution', {})
                    excellent = distribution.get('Excellent', 0)
                    very_good = distribution.get('Very Good', 0)
                    click.echo(f"  High Quality (Excellent + Very Good): {excellent + very_good} photos")
        
    except Exception as e:
        click.echo(f"❌ Error generating statistics: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Add existing command groups
main.add_command(db)
main.add_command(project_group)


@main.command()
def version():
    """Show PhotoSight version information."""
    click.echo("PhotoSight v0.1.0")
    click.echo("Intelligent photo management and ranking system")


if __name__ == '__main__':
    main()