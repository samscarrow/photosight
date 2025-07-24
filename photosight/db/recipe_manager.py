"""
Recipe Manager for PhotoSight - Database integration for processing recipes

Bridges the gap between the RAW processor's ProcessingRecipe dataclass
and the database ProcessingRecipe model, enabling persistent storage
and retrieval of processing settings.
"""

import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from .connection import get_session
from .models import ProcessingRecipe as DBRecipe, Photo, PhotoRecipe, Project, Task, project_photos
from ..processing.raw_processor import ProcessingRecipe as ProcessingRecipeData

logger = logging.getLogger(__name__)


class RecipeManager:
    """
    Manages processing recipes in the database.
    
    Handles conversion between the RAW processor's ProcessingRecipe dataclass
    and the database model, enabling persistent storage and batch processing.
    """
    
    def save_recipe(self, recipe_data: ProcessingRecipeData, 
                   name: str, description: Optional[str] = None,
                   created_by: Optional[str] = None) -> DBRecipe:
        """
        Save a processing recipe to the database.
        
        Args:
            recipe_data: ProcessingRecipe dataclass from RAW processor
            name: Unique name for the recipe
            description: Optional description
            created_by: Username/identifier of creator
            
        Returns:
            Database ProcessingRecipe model instance
        """
        logger.info(f"Saving recipe '{name}' to database")
        
        with get_session() as session:
            # Check if recipe with this name exists
            existing = session.query(DBRecipe).filter(
                DBRecipe.name == name
            ).first()
            
            # Convert dataclass to dict, excluding file-specific fields
            recipe_dict = {
                k: v for k, v in recipe_data.__dict__.items()
                if k not in ['source_path', 'file_hash', 'created_at']
            }
            
            if existing:
                # Update existing recipe
                logger.info(f"Updating existing recipe '{name}'")
                existing.parameters = recipe_dict
                existing.description = description or existing.description
                existing.updated_at = datetime.now()
                existing.times_used += 1
                existing.last_used = datetime.now()
                db_recipe = existing
            else:
                # Create new recipe
                logger.info(f"Creating new recipe '{name}'")
                db_recipe = DBRecipe(
                    name=name,
                    description=description,
                    parameters=recipe_dict,
                    created_by=created_by,
                    times_used=1,
                    last_used=datetime.now()
                )
                session.add(db_recipe)
            
            session.commit()
            logger.info(f"Recipe '{name}' saved successfully (ID: {db_recipe.id})")
            return db_recipe
    
    def load_recipe(self, recipe_name: str, 
                   source_path: Optional[str] = None) -> Optional[ProcessingRecipeData]:
        """
        Load a recipe from the database by name.
        
        Args:
            recipe_name: Name of the recipe to load
            source_path: Optional source path for the recipe
            
        Returns:
            ProcessingRecipe dataclass or None if not found
        """
        logger.info(f"Loading recipe '{recipe_name}' from database")
        
        with get_session() as session:
            db_recipe = session.query(DBRecipe).filter(
                DBRecipe.name == recipe_name
            ).first()
            
            if not db_recipe:
                logger.warning(f"Recipe '{recipe_name}' not found")
                return None
            
            # Update usage tracking
            db_recipe.times_used += 1
            db_recipe.last_used = datetime.now()
            session.commit()
            
            # Convert database parameters to dataclass
            params = db_recipe.parameters.copy()
            params['source_path'] = source_path or ''
            params['created_at'] = db_recipe.created_at.isoformat()
            
            logger.info(f"Recipe '{recipe_name}' loaded successfully")
            return ProcessingRecipeData(**params)
    
    def load_recipe_by_id(self, recipe_id: int, 
                         source_path: Optional[str] = None) -> Optional[ProcessingRecipeData]:
        """
        Load a recipe from the database by ID.
        
        Args:
            recipe_id: Database ID of the recipe
            source_path: Optional source path for the recipe
            
        Returns:
            ProcessingRecipe dataclass or None if not found
        """
        with get_session() as session:
            db_recipe = session.query(DBRecipe).filter(
                DBRecipe.id == recipe_id
            ).first()
            
            if not db_recipe:
                logger.warning(f"Recipe with ID {recipe_id} not found")
                return None
            
            # Update usage tracking
            db_recipe.times_used += 1
            db_recipe.last_used = datetime.now()
            session.commit()
            
            # Convert database parameters to dataclass
            params = db_recipe.parameters.copy()
            params['source_path'] = source_path or ''
            params['created_at'] = db_recipe.created_at.isoformat()
            
            return ProcessingRecipeData(**params)
    
    def list_recipes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all available recipes with metadata.
        
        Args:
            limit: Maximum number of recipes to return
            
        Returns:
            List of recipe summaries
        """
        with get_session() as session:
            recipes = session.query(DBRecipe).order_by(
                DBRecipe.last_used.desc().nullsfirst(),
                DBRecipe.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    'id': r.id,
                    'name': r.name,
                    'description': r.description,
                    'times_used': r.times_used,
                    'last_used': r.last_used.isoformat() if r.last_used else None,
                    'created_at': r.created_at.isoformat() if r.created_at else None,
                    'created_by': r.created_by
                }
                for r in recipes
            ]
    
    def save_photo_recipe(self, photo_id: int, recipe_data: ProcessingRecipeData,
                         is_applied: bool = False) -> PhotoRecipe:
        """
        Save a recipe associated with a specific photo.
        
        Args:
            photo_id: Database ID of the photo
            recipe_data: ProcessingRecipe dataclass
            is_applied: Whether this recipe has been applied to generate output
            
        Returns:
            PhotoRecipe association record
        """
        logger.info(f"Saving recipe for photo ID {photo_id}")
        
        with get_session() as session:
            # Check if photo exists
            photo = session.query(Photo).filter(Photo.id == photo_id).first()
            if not photo:
                raise ValueError(f"Photo with ID {photo_id} not found")
            
            # Convert recipe to JSON
            recipe_json = recipe_data.to_json()
            
            # Check for existing photo recipe
            existing = session.query(PhotoRecipe).filter(
                PhotoRecipe.photo_id == photo_id
            ).first()
            
            if existing:
                # Update existing
                existing.recipe_data = json.loads(recipe_json)
                existing.updated_at = datetime.now()
                if is_applied and not existing.applied_at:
                    existing.applied_at = datetime.now()
                photo_recipe = existing
            else:
                # Create new
                photo_recipe = PhotoRecipe(
                    photo_id=photo_id,
                    recipe_data=json.loads(recipe_json),
                    applied_at=datetime.now() if is_applied else None
                )
                session.add(photo_recipe)
            
            session.commit()
            logger.info(f"Recipe saved for photo ID {photo_id}")
            return photo_recipe
    
    def load_photo_recipe(self, photo_id: int) -> Optional[ProcessingRecipeData]:
        """
        Load the recipe associated with a specific photo.
        
        Args:
            photo_id: Database ID of the photo
            
        Returns:
            ProcessingRecipe dataclass or None if not found
        """
        logger.info(f"Loading recipe for photo ID {photo_id}")
        
        with get_session() as session:
            photo_recipe = session.query(PhotoRecipe).filter(
                PhotoRecipe.photo_id == photo_id
            ).first()
            
            if not photo_recipe:
                logger.warning(f"No recipe found for photo ID {photo_id}")
                return None
            
            # Get photo for source path
            photo = session.query(Photo).filter(Photo.id == photo_id).first()
            if not photo:
                logger.error(f"Photo with ID {photo_id} not found")
                return None
            
            # Convert JSON back to dataclass
            recipe_data = photo_recipe.recipe_data.copy()
            recipe_data['source_path'] = photo.file_path
            
            logger.info(f"Recipe loaded for photo ID {photo_id}")
            return ProcessingRecipeData(**recipe_data)
    
    def find_similar_recipes(self, recipe_data: ProcessingRecipeData, 
                           threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        Find recipes with similar parameters.
        
        Args:
            recipe_data: Recipe to compare against
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar recipes with similarity scores
        """
        logger.info("Searching for similar recipes")
        
        # Key parameters for comparison
        key_params = [
            'exposure_adjustment', 'shadows', 'highlights', 'contrast',
            'wb_method', 'temperature_adjustment', 'tint_adjustment',
            'color_grading_preset', 'vibrance', 'saturation'
        ]
        
        with get_session() as session:
            all_recipes = session.query(DBRecipe).all()
            
            similar = []
            for db_recipe in all_recipes:
                # Calculate similarity based on key parameters
                matching_params = 0
                total_params = len(key_params)
                
                for param in key_params:
                    recipe_val = getattr(recipe_data, param, None)
                    db_val = db_recipe.parameters.get(param)
                    
                    if recipe_val == db_val:
                        matching_params += 1
                    elif isinstance(recipe_val, (int, float)) and isinstance(db_val, (int, float)):
                        # For numeric values, consider close values as matching
                        if abs(recipe_val - db_val) < 0.1:
                            matching_params += 0.8
                
                similarity = matching_params / total_params
                
                if similarity >= threshold:
                    similar.append({
                        'id': db_recipe.id,
                        'name': db_recipe.name,
                        'description': db_recipe.description,
                        'similarity': similarity,
                        'times_used': db_recipe.times_used
                    })
            
            # Sort by similarity and usage
            similar.sort(key=lambda x: (x['similarity'], x['times_used']), reverse=True)
            
            logger.info(f"Found {len(similar)} similar recipes")
            return similar[:10]  # Return top 10
    
    def get_popular_recipes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most popular recipes by usage.
        
        Args:
            limit: Number of recipes to return
            
        Returns:
            List of popular recipes
        """
        with get_session() as session:
            recipes = session.query(DBRecipe).order_by(
                DBRecipe.times_used.desc()
            ).limit(limit).all()
            
            return [
                {
                    'id': r.id,
                    'name': r.name,
                    'description': r.description,
                    'times_used': r.times_used,
                    'last_used': r.last_used.isoformat() if r.last_used else None,
                    'parameters_summary': self._summarize_parameters(r.parameters)
                }
                for r in recipes
            ]
    
    def _summarize_parameters(self, params: Dict[str, Any]) -> str:
        """Create a brief summary of recipe parameters."""
        summary_parts = []
        
        if params.get('exposure_adjustment', 0) != 0:
            summary_parts.append(f"Exposure: {params['exposure_adjustment']:+.1f}EV")
        
        if params.get('color_grading_preset'):
            summary_parts.append(f"Style: {params['color_grading_preset']}")
        
        if params.get('wb_method'):
            summary_parts.append(f"WB: {params['wb_method']}")
        
        return ", ".join(summary_parts) if summary_parts else "Default settings"
    
    def get_effective_recipe(self, photo_id: int, task_id: Optional[int] = None, 
                           project_id: Optional[int] = None) -> Optional[ProcessingRecipeData]:
        """
        Get the effective processing recipe for a photo using inheritance hierarchy.
        
        Recipe hierarchy (highest to lowest priority):
        1. Photo-specific recipe
        2. Task-specific recipe 
        3. Project default recipe
        4. System default recipe
        
        Args:
            photo_id: Photo ID to get recipe for
            task_id: Optional task context
            project_id: Optional project context (auto-detected if not provided)
            
        Returns:
            ProcessingRecipe dataclass or None
        """
        with get_session() as session:
            # 1. Check for photo-specific recipe (highest priority)
            photo_recipe = self.load_photo_recipe(photo_id)
            if photo_recipe:
                logger.debug(f"Using photo-specific recipe for photo {photo_id}")
                return photo_recipe
            
            # Get photo for source path and project detection
            photo = session.query(Photo).filter(Photo.id == photo_id).first()
            if not photo:
                logger.warning(f"Photo {photo_id} not found")
                return None
            
            # Auto-detect project if not provided
            if not project_id:
                project_association = session.query(project_photos).filter(
                    project_photos.c.photo_id == photo_id
                ).first()
                if project_association:
                    project_id = project_association.project_id
            
            # 2. Check for task-specific recipe
            if task_id:
                task = session.query(Task).filter(Task.id == task_id).first()
                if task and task.recipe_id:
                    recipe = self.load_recipe_by_id(task.recipe_id, photo.file_path)
                    if recipe:
                        logger.debug(f"Using task recipe for photo {photo_id}")
                        return recipe
            
            # 3. Check for project default recipe
            if project_id:
                project = session.query(Project).filter(Project.id == project_id).first()
                if project and project.default_recipe_id:
                    recipe = self.load_recipe_by_id(project.default_recipe_id, photo.file_path)
                    if recipe:
                        logger.debug(f"Using project default recipe for photo {photo_id}")
                        return recipe
            
            # 4. No recipe found
            logger.debug(f"No recipe found for photo {photo_id}")
            return None
    
    def apply_recipe_to_project_photos(self, project_id: int, recipe_id: int) -> int:
        """
        Apply a recipe to all photos in a project.
        
        Args:
            project_id: Project ID
            recipe_id: ProcessingRecipe ID to apply
            
        Returns:
            Number of photos updated
        """
        with get_session() as session:
            # Get all photos in the project
            photo_ids = session.query(project_photos.c.photo_id).filter(
                project_photos.c.project_id == project_id
            ).all()
            
            photo_id_list = [pid[0] for pid in photo_ids]
            
            logger.info(f"Applying recipe {recipe_id} to {len(photo_id_list)} photos in project {project_id}")
            
            # Apply recipe to each photo
            count = 0
            recipe = self.load_recipe_by_id(recipe_id)
            if not recipe:
                raise ValueError(f"Recipe {recipe_id} not found")
            
            for photo_id in photo_id_list:
                try:
                    self.save_photo_recipe(photo_id, recipe)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to apply recipe to photo {photo_id}: {e}")
            
            return count
    
    def set_project_default_recipe(self, project_id: int, recipe_id: int):
        """
        Set the default recipe for a project.
        
        Args:
            project_id: Project ID
            recipe_id: ProcessingRecipe ID to set as default
        """
        with get_session() as session:
            project = session.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            recipe = session.query(DBRecipe).filter(DBRecipe.id == recipe_id).first()
            if not recipe:
                raise ValueError(f"Recipe {recipe_id} not found")
            
            project.default_recipe_id = recipe_id
            session.commit()
            
            logger.info(f"Set recipe '{recipe.name}' as default for project '{project.name}'")
    
    def get_recipe_suggestions_for_photo(self, photo_id: int) -> List[Dict[str, Any]]:
        """
        Get recipe suggestions for a photo based on project context and similar photos.
        
        Args:
            photo_id: Photo ID to get suggestions for
            
        Returns:
            List of recipe suggestions with reasoning
        """
        suggestions = []
        
        with get_session() as session:
            # Get photo and project context
            photo = session.query(Photo).filter(Photo.id == photo_id).first()
            if not photo:
                return suggestions
            
            # Find project association
            project_association = session.query(project_photos).filter(
                project_photos.c.photo_id == photo_id
            ).first()
            
            if project_association:
                project = session.query(Project).filter(
                    Project.id == project_association.project_id
                ).first()
                
                if project:
                    # Project default recipe
                    if project.default_recipe_id:
                        default_recipe = session.query(DBRecipe).filter(
                            DBRecipe.id == project.default_recipe_id
                        ).first()
                        if default_recipe:
                            suggestions.append({
                                "recipe_id": default_recipe.id,
                                "recipe_name": default_recipe.name,
                                "reason": f"Default for project '{project.name}'",
                                "confidence": 0.9
                            })
                    
                    # Recipes suitable for project type
                    if project.project_type:
                        suitable_recipes = session.query(DBRecipe).filter(
                            DBRecipe.category == project.project_type,
                            DBRecipe.is_active == True
                        ).order_by(DBRecipe.times_used.desc()).limit(3).all()
                        
                        for recipe in suitable_recipes:
                            suggestions.append({
                                "recipe_id": recipe.id,
                                "recipe_name": recipe.name,
                                "reason": f"Suitable for {project.project_type} projects",
                                "confidence": 0.7
                            })
            
            # Popular recipes
            popular_recipes = session.query(DBRecipe).filter(
                DBRecipe.is_active == True
            ).order_by(DBRecipe.times_used.desc()).limit(5).all()
            
            for recipe in popular_recipes:
                if not any(s["recipe_id"] == recipe.id for s in suggestions):
                    suggestions.append({
                        "recipe_id": recipe.id,
                        "recipe_name": recipe.name,
                        "reason": f"Popular recipe (used {recipe.times_used or 0} times)",
                        "confidence": 0.5
                    })
            
            # Sort by confidence
            suggestions.sort(key=lambda x: x["confidence"], reverse=True)
            return suggestions[:10]