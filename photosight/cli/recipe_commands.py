"""
Recipe management CLI commands for PhotoSight

Provides command-line interface for processing recipe operations.
"""

import click
import logging
from pathlib import Path
from datetime import datetime
import json

from ..processing.raw_processor import RawPostProcessor, ProcessingRecipe
from ..db.recipe_manager import RecipeManager
from ..db.database import get_db_session
from ..db.models import Photo, ProcessingRecipe as DBRecipe

logger = logging.getLogger(__name__)


@click.group()
def recipe():
    """Processing recipe management commands"""
    pass


@recipe.command()
@click.argument('raw_file', type=click.Path(exists=True, path_type=Path))
@click.option('--name', '-n', required=True, help='Name for the recipe')
@click.option('--description', '-d', help='Recipe description')
@click.option('--analyze/--no-analyze', default=True, help='Auto-analyze the image')
@click.option('--save-preview', '-p', type=click.Path(), help='Save preview to file')
def create(raw_file, name, description, analyze, save_preview):
    """Create a new processing recipe from a RAW file"""
    click.echo(f"Creating recipe from {raw_file}")
    
    # Initialize processor
    processor = RawPostProcessor(auto_analyze=analyze)
    
    # Create recipe
    recipe = processor.create_default_recipe(raw_file)
    
    # Save to database
    if processor.save_recipe_to_db(recipe, name, description):
        click.echo(f"✓ Recipe '{name}' saved to database")
    else:
        click.echo("✗ Failed to save recipe to database", err=True)
        return
    
    # Generate preview if requested
    if save_preview:
        click.echo("Generating preview...")
        preview = processor.generate_preview(raw_file, recipe)
        import cv2
        cv2.imwrite(str(save_preview), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        click.echo(f"Preview saved to {save_preview}")
    
    # Display recipe summary
    click.echo("\nRecipe Summary:")
    click.echo(f"  Exposure: {recipe.exposure_adjustment:+.2f} EV")
    click.echo(f"  Shadows: {recipe.shadows:+.0f}")
    click.echo(f"  Highlights: {recipe.highlights:+.0f}")
    if recipe.wb_analysis:
        click.echo(f"  White Balance: {recipe.wb_analysis.get('estimated_temp_kelvin', 'Unknown')}K")
    if recipe.color_grading_preset:
        click.echo(f"  Color Grading: {recipe.color_grading_preset}")


@recipe.command()
@click.option('--limit', '-l', default=20, help='Number of recipes to list')
@click.option('--popular', is_flag=True, help='Sort by popularity')
def list(limit, popular):
    """List available processing recipes"""
    manager = RecipeManager()
    
    if popular:
        recipes = manager.get_popular_recipes(limit)
        click.echo("Popular Processing Recipes:")
        click.echo("=" * 80)
        
        for r in recipes:
            click.echo(f"\n{r['name']} (ID: {r['id']})")
            if r['description']:
                click.echo(f"  Description: {r['description']}")
            click.echo(f"  Used: {r['times_used']} times")
            if r['last_used']:
                click.echo(f"  Last used: {r['last_used']}")
            click.echo(f"  Settings: {r['parameters_summary']}")
    else:
        recipes = manager.list_recipes(limit)
        click.echo("Available Processing Recipes:")
        click.echo("=" * 80)
        
        for r in recipes:
            click.echo(f"\n{r['name']} (ID: {r['id']})")
            if r['description']:
                click.echo(f"  {r['description']}")
            click.echo(f"  Created: {r['created_at']}")
            if r['created_by']:
                click.echo(f"  By: {r['created_by']}")
            click.echo(f"  Used: {r['times_used']} times")


@recipe.command()
@click.argument('raw_file', type=click.Path(exists=True, path_type=Path))
@click.argument('recipe_name')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output file path')
@click.option('--format', '-f', type=click.Choice(['jpeg', 'png', 'tiff']), default='jpeg')
@click.option('--quality', '-q', type=click.IntRange(1, 100), default=95)
@click.option('--preview-only', is_flag=True, help='Generate preview instead of full resolution')
def apply(raw_file, recipe_name, output, format, quality, preview_only):
    """Apply a recipe to a RAW file"""
    click.echo(f"Applying recipe '{recipe_name}' to {raw_file}")
    
    # Initialize processor
    processor = RawPostProcessor()
    
    # Load recipe
    recipe = processor.load_recipe_from_db(recipe_name, raw_file)
    if not recipe:
        click.echo(f"Recipe '{recipe_name}' not found", err=True)
        return
    
    output_path = Path(output)
    
    if preview_only:
        # Generate preview
        click.echo("Generating preview...")
        preview = processor.generate_preview(raw_file, recipe)
        
        import cv2
        if format == 'jpeg':
            cv2.imwrite(str(output_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(str(output_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    else:
        # Export full resolution
        click.echo("Processing full resolution image...")
        processor.export_full_size(raw_file, recipe, output_path, format, quality)
    
    click.echo(f"✓ Output saved to {output_path}")


@recipe.command()
@click.argument('photo_id', type=int)
@click.option('--apply', is_flag=True, help='Mark recipe as applied')
def save_for_photo(photo_id, apply):
    """Save the current recipe for a photo in the database"""
    with get_db_session() as session:
        photo = session.query(Photo).filter(Photo.id == photo_id).first()
        if not photo:
            click.echo(f"Photo with ID {photo_id} not found", err=True)
            return
        
        # Initialize processor and create recipe
        processor = RawPostProcessor()
        recipe = processor.create_default_recipe(Path(photo.file_path))
        
        # Save to database
        if processor.save_photo_recipe(photo_id, recipe, is_applied=apply):
            click.echo(f"✓ Recipe saved for photo {photo.file_name}")
            if apply:
                click.echo("  (marked as applied)")
        else:
            click.echo("✗ Failed to save recipe", err=True)


@recipe.command()
@click.argument('recipe_file', type=click.Path(exists=True, path_type=Path))
@click.option('--name', '-n', required=True, help='Name for imported recipe')
@click.option('--description', '-d', help='Recipe description')
def import_recipe(recipe_file, name, description):
    """Import a recipe from a JSON file"""
    try:
        # Load recipe from file
        recipe = ProcessingRecipe.load(recipe_file)
        
        # Save to database
        manager = RecipeManager()
        db_recipe = manager.save_recipe(recipe, name, description)
        
        click.echo(f"✓ Recipe '{name}' imported successfully (ID: {db_recipe.id})")
        
    except Exception as e:
        click.echo(f"✗ Failed to import recipe: {e}", err=True)


@recipe.command()
@click.argument('recipe_name')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def export(recipe_name, output):
    """Export a recipe to a JSON file"""
    manager = RecipeManager()
    recipe = manager.load_recipe(recipe_name)
    
    if not recipe:
        click.echo(f"Recipe '{recipe_name}' not found", err=True)
        return
    
    if output:
        recipe.save(Path(output))
        click.echo(f"✓ Recipe exported to {output}")
    else:
        # Print to stdout
        click.echo(recipe.to_json())


@recipe.command()
@click.argument('raw_file', type=click.Path(exists=True, path_type=Path))
@click.option('--threshold', '-t', type=float, default=0.9, help='Similarity threshold')
def find_similar(raw_file, threshold):
    """Find similar recipes for a RAW file"""
    click.echo(f"Analyzing {raw_file} and searching for similar recipes...")
    
    # Create recipe from file
    processor = RawPostProcessor()
    recipe = processor.create_default_recipe(raw_file)
    
    # Find similar recipes
    similar = processor.find_similar_recipes(recipe, threshold)
    
    if similar:
        click.echo(f"\nFound {len(similar)} similar recipes:")
        click.echo("=" * 60)
        
        for r in similar:
            click.echo(f"\n{r['name']} (ID: {r['id']})")
            click.echo(f"  Similarity: {r['similarity']:.1%}")
            if r['description']:
                click.echo(f"  Description: {r['description']}")
            click.echo(f"  Used: {r['times_used']} times")
    else:
        click.echo("No similar recipes found")


# Add to main CLI
@click.group()
def cli():
    """PhotoSight recipe management"""
    pass


cli.add_command(recipe)


if __name__ == '__main__':
    cli()