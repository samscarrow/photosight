"""
PhotoSight Database Module

Provides database models, operations, and connection management.
"""

from .connection import configure_database, get_session, get_session_factory, get_engine
from .models import *
from .operations import PhotoOperations, ProjectOperations, TaskOperations, BatchSession
from .recipe_manager import RecipeManager

# Check if database is available
def is_database_available() -> bool:
    """Check if database is configured and available."""
    try:
        session_factory = get_session_factory()
        return session_factory is not None
    except Exception:
        return False

__all__ = [
    'configure_database',
    'get_session', 
    'get_session_factory',
    'get_engine',
    'is_database_available',
    'PhotoOperations',
    'ProjectOperations', 
    'TaskOperations',
    'BatchSession',
    'RecipeManager',
    # Models
    'Project',
    'Task', 
    'Photo',
    'ProcessingRecipe',
    'PhotoRecipe',
    'AnalysisResult',
    'FaceDetection',
    'CompositionAnalysis',
    'SimilarityGroup',
    'PhotoSimilarity',
    'ProjectPhoto',
    'ProjectStatus',
    'ProjectPhase',
    'TaskStatus',
    'TaskPriority',
    'ProcessingStatus',
    'project_photos',
    'photo_tasks'
]