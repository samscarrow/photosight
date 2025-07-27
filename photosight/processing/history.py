"""
History stack management for PhotoSight.

Implements undo/redo functionality with efficient state management
and change tracking for non-destructive editing workflows.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import copy

from .raw_processor import ProcessingRecipe


logger = logging.getLogger(__name__)


@dataclass
class HistoryAction:
    """Represents a single action in the history stack."""
    action_id: str
    timestamp: str
    action_type: str  # "recipe_change", "crop", "local_adjustment", etc.
    description: str
    recipe_before: Optional[Dict[str, Any]] = None
    recipe_after: Optional[Dict[str, Any]] = None
    affected_layers: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class HistorySnapshot:
    """Represents a complete state snapshot for fast restoration."""
    snapshot_id: str
    timestamp: str
    description: str
    recipe: Dict[str, Any]
    preview_cache_key: Optional[str] = None
    action_count: int = 0


class HistoryStack:
    """
    Manages undo/redo history for PhotoSight editing sessions.
    
    Features:
    - Efficient delta-based storage
    - Automatic snapshot creation at key points
    - Memory-conscious with configurable limits
    - Fast state restoration
    """
    
    def __init__(self, max_actions: int = 100, max_snapshots: int = 10):
        """
        Initialize history stack.
        
        Args:
            max_actions: Maximum number of actions to retain
            max_snapshots: Maximum number of snapshots to retain
        """
        self.max_actions = max_actions
        self.max_snapshots = max_snapshots
        
        # History storage
        self.actions: List[HistoryAction] = []
        self.snapshots: List[HistorySnapshot] = []
        
        # Current position
        self.current_position = -1  # -1 means at the latest state
        
        # Session info
        self.session_id: Optional[str] = None
        self.original_recipe: Optional[Dict[str, Any]] = None
        
        logger.debug(f"Initialized history stack: max_actions={max_actions}, max_snapshots={max_snapshots}")
    
    def initialize_session(self, session_id: str, initial_recipe: ProcessingRecipe) -> None:
        """
        Initialize history for a new editing session.
        
        Args:
            session_id: Unique session identifier
            initial_recipe: Starting recipe state
        """
        self.session_id = session_id
        self.original_recipe = asdict(initial_recipe)
        self.current_position = -1
        
        # Create initial snapshot
        self.create_snapshot(
            "Session Start",
            self.original_recipe,
            action_count=0
        )
        
        logger.info(f"Initialized history for session {session_id}")
    
    def add_action(self, action_type: str, description: str, 
                   recipe_before: ProcessingRecipe, recipe_after: ProcessingRecipe,
                   affected_layers: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new action to the history stack.
        
        Args:
            action_type: Type of action performed
            description: Human-readable description
            recipe_before: Recipe state before the action
            recipe_after: Recipe state after the action
            affected_layers: List of affected adjustment layers
            metadata: Additional action metadata
            
        Returns:
            Unique action ID
        """
        # Generate unique action ID
        action_id = f"{self.session_id}_{len(self.actions)}_{datetime.now().strftime('%H%M%S')}"
        
        # If we're not at the latest position, remove future actions
        if self.current_position != -1:
            self.actions = self.actions[:self.current_position + 1]
        
        # Create action
        action = HistoryAction(
            action_id=action_id,
            timestamp=datetime.now().isoformat(),
            action_type=action_type,
            description=description,
            recipe_before=asdict(recipe_before),
            recipe_after=asdict(recipe_after),
            affected_layers=affected_layers or [],
            metadata=metadata or {}
        )
        
        self.actions.append(action)
        self.current_position = -1  # Reset to latest
        
        # Trim if exceeding limits
        if len(self.actions) > self.max_actions:
            removed_count = len(self.actions) - self.max_actions
            self.actions = self.actions[removed_count:]
            logger.debug(f"Trimmed {removed_count} old actions from history")
        
        # Create snapshot at key intervals
        if len(self.actions) % 10 == 0:  # Every 10 actions
            self.create_snapshot(
                f"Auto-snapshot at action {len(self.actions)}",
                asdict(recipe_after),
                action_count=len(self.actions)
            )
        
        logger.debug(f"Added action: {action_type} - {description}")
        return action_id
    
    def undo(self) -> Optional[ProcessingRecipe]:
        """
        Undo the last action.
        
        Returns:
            Recipe state after undo, or None if nothing to undo
        """
        if not self.can_undo():
            logger.debug("Cannot undo: no previous actions")
            return None
        
        if self.current_position == -1:
            # Moving from latest to second-to-last
            self.current_position = len(self.actions) - 2
        else:
            # Moving further back
            self.current_position -= 1
        
        # Get target state
        if self.current_position == -1:
            # Back to original state
            target_recipe = self.original_recipe
        else:
            # Get state after the action at current position
            target_recipe = self.actions[self.current_position]['recipe_after']
        
        logger.debug(f"Undo: moved to position {self.current_position}")
        return ProcessingRecipe.from_dict(target_recipe)
    
    def redo(self) -> Optional[ProcessingRecipe]:
        """
        Redo the next action.
        
        Returns:
            Recipe state after redo, or None if nothing to redo
        """
        if not self.can_redo():
            logger.debug("Cannot redo: no future actions")
            return None
        
        # Move forward
        self.current_position += 1
        
        # If we've reached the end, reset to latest
        if self.current_position >= len(self.actions) - 1:
            self.current_position = -1
            target_recipe = self.actions[-1]['recipe_after']
        else:
            target_recipe = self.actions[self.current_position]['recipe_after']
        
        logger.debug(f"Redo: moved to position {self.current_position}")
        return ProcessingRecipe.from_dict(target_recipe)
    
    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return len(self.actions) > 0 and (
            self.current_position == -1 or self.current_position > -1
        )
    
    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return (
            len(self.actions) > 0 and 
            self.current_position != -1 and 
            self.current_position < len(self.actions) - 1
        )
    
    def get_current_recipe(self) -> Optional[ProcessingRecipe]:
        """
        Get the current recipe state.
        
        Returns:
            Current ProcessingRecipe or None if no actions
        """
        if not self.actions:
            if self.original_recipe:
                return ProcessingRecipe.from_dict(self.original_recipe)
            return None
        
        if self.current_position == -1:
            # At latest state
            return ProcessingRecipe.from_dict(self.actions[-1]['recipe_after'])
        else:
            # At specific position
            return ProcessingRecipe.from_dict(self.actions[self.current_position]['recipe_after'])
    
    def create_snapshot(self, description: str, recipe: Dict[str, Any], 
                       action_count: Optional[int] = None) -> str:
        """
        Create a snapshot of the current state.
        
        Args:
            description: Snapshot description
            recipe: Current recipe state
            action_count: Number of actions at snapshot time
            
        Returns:
            Snapshot ID
        """
        snapshot_id = f"snap_{self.session_id}_{len(self.snapshots)}"
        
        snapshot = HistorySnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now().isoformat(),
            description=description,
            recipe=copy.deepcopy(recipe),
            action_count=action_count or len(self.actions)
        )
        
        self.snapshots.append(snapshot)
        
        # Trim snapshots if exceeding limit
        if len(self.snapshots) > self.max_snapshots:
            removed = self.snapshots.pop(0)
            logger.debug(f"Removed old snapshot: {removed.description}")
        
        logger.debug(f"Created snapshot: {description}")
        return snapshot_id
    
    def restore_snapshot(self, snapshot_id: str) -> Optional[ProcessingRecipe]:
        """
        Restore state from a snapshot.
        
        Args:
            snapshot_id: ID of snapshot to restore
            
        Returns:
            Recipe state from snapshot or None if not found
        """
        for snapshot in self.snapshots:
            if snapshot.snapshot_id == snapshot_id:
                logger.info(f"Restored snapshot: {snapshot.description}")
                return ProcessingRecipe.from_dict(snapshot.recipe)
        
        logger.warning(f"Snapshot not found: {snapshot_id}")
        return None
    
    def get_history_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current history state.
        
        Returns:
            History summary with actions and snapshots
        """
        return {
            'session_id': self.session_id,
            'total_actions': len(self.actions),
            'current_position': self.current_position,
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo(),
            'recent_actions': [
                {
                    'action_id': action.action_id,
                    'timestamp': action.timestamp,
                    'action_type': action.action_type,
                    'description': action.description
                }
                for action in self.actions[-5:]  # Last 5 actions
            ],
            'snapshots': [
                {
                    'snapshot_id': snapshot.snapshot_id,
                    'timestamp': snapshot.timestamp,
                    'description': snapshot.description,
                    'action_count': snapshot.action_count
                }
                for snapshot in self.snapshots
            ]
        }
    
    def clear_history(self) -> None:
        """Clear all history data."""
        self.actions.clear()
        self.snapshots.clear()
        self.current_position = -1
        logger.info(f"Cleared history for session {self.session_id}")
    
    def export_history(self, file_path: Path) -> bool:
        """
        Export history to a file.
        
        Args:
            file_path: Path to save history data
            
        Returns:
            True if successful
        """
        try:
            history_data = {
                'session_id': self.session_id,
                'original_recipe': self.original_recipe,
                'actions': [asdict(action) for action in self.actions],
                'snapshots': [asdict(snapshot) for snapshot in self.snapshots],
                'current_position': self.current_position,
                'exported_at': datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info(f"Exported history to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export history: {e}")
            return False
    
    def import_history(self, file_path: Path) -> bool:
        """
        Import history from a file.
        
        Args:
            file_path: Path to load history data from
            
        Returns:
            True if successful
        """
        try:
            with open(file_path, 'r') as f:
                history_data = json.load(f)
            
            self.session_id = history_data['session_id']
            self.original_recipe = history_data['original_recipe']
            self.current_position = history_data['current_position']
            
            # Reconstruct actions
            self.actions = [
                HistoryAction(**action_data) 
                for action_data in history_data['actions']
            ]
            
            # Reconstruct snapshots
            self.snapshots = [
                HistorySnapshot(**snapshot_data)
                for snapshot_data in history_data['snapshots']
            ]
            
            logger.info(f"Imported history from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import history: {e}")
            return False