"""
Session Manager for PhotoSight API

Manages editing sessions and coordinates with the preview system.
"""

import logging
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Callable
import uuid
import base64
import io
import numpy as np
from PIL import Image

from ..preview import PhotoSightPreviewSystem, ProxyLevel, PreviewConfig
from ..processing.raw_processor import ProcessingRecipe
from ..processing.history import HistoryStack
from .models import SessionInfo, SessionState, PreviewUpdate, ProcessingProgress

logger = logging.getLogger(__name__)


class EditingSession:
    """Represents an active editing session."""
    
    def __init__(self, session_id: str, image_path: Path, user_id: str):
        self.session_id = session_id
        self.image_path = image_path
        self.user_id = user_id
        self.state = SessionState.CREATED
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.recipe = None
        self.recipe_id = None
        self.preview_level = ProxyLevel.MEDIUM
        self.preview_system_session = None
        self.metadata = {}
        self.subscribers = set()  # WebSocket session IDs
        self._lock = threading.RLock()
        
        # Initialize history stack
        self.history = HistoryStack(max_actions=100, max_snapshots=10)
    
    def to_session_info(self) -> SessionInfo:
        """Convert to SessionInfo model."""
        return SessionInfo(
            session_id=self.session_id,
            state=self.state,
            created_at=self.created_at,
            updated_at=self.updated_at,
            image_path=str(self.image_path),
            recipe_id=self.recipe_id,
            preview_level=self.preview_level.value,
            metadata=self.metadata
        )
    
    def update_state(self, new_state: SessionState):
        """Update session state."""
        with self._lock:
            self.state = new_state
            self.updated_at = datetime.utcnow()


class SessionManager:
    """
    Manages editing sessions for the API.
    
    Coordinates between API requests and the PhotoSight preview system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.sessions: Dict[str, EditingSession] = {}
        self._lock = threading.RLock()
        
        # Initialize preview system
        preview_config = PreviewConfig(
            max_memory_cache_mb=self.config.get('max_cache_mb', 512),
            max_worker_threads=self.config.get('worker_threads', 4),
            tile_size=self.config.get('tile_size', 512)
        )
        self.preview_system = PhotoSightPreviewSystem(preview_config)
        
        # WebSocket callback
        self.ws_callback: Optional[Callable] = None
        
        logger.info("SessionManager initialized")
    
    def create_session(self, image_path: Path, user_id: str,
                      initial_recipe: Optional[Dict[str, Any]] = None) -> EditingSession:
        """
        Create a new editing session.
        
        Args:
            image_path: Path to the image file
            user_id: User ID creating the session
            initial_recipe: Optional initial processing recipe
            
        Returns:
            Created EditingSession
        """
        session_id = f"edit_{uuid.uuid4().hex[:12]}"
        
        with self._lock:
            # Create session
            session = EditingSession(session_id, image_path, user_id)
            
            # Set initial recipe
            if initial_recipe:
                session.recipe = ProcessingRecipe.from_dict(initial_recipe)
                session.recipe_id = f"recipe_{uuid.uuid4().hex[:12]}"
            else:
                # Create default recipe
                from ..processing.raw_processor import RawProcessor
                processor = RawProcessor()
                session.recipe = processor.create_default_recipe(image_path)
                session.recipe_id = f"recipe_{uuid.uuid4().hex[:12]}"
            
            # Initialize history with the initial recipe
            session.history.initialize_session(session_id, session.recipe)
            
            # Store session
            self.sessions[session_id] = session
            
            # Start preview system session
            self._start_preview_session(session)
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return session
    
    def get_session(self, session_id: str) -> Optional[EditingSession]:
        """Get an active session."""
        with self._lock:
            return self.sessions.get(session_id)
    
    def update_recipe(self, session_id: str, recipe_update: Dict[str, Any]) -> bool:
        """
        Update the processing recipe for a session.
        
        Args:
            session_id: Session ID
            recipe_update: Recipe updates to apply
            
        Returns:
            True if successful
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            # Store recipe before changes for history
            recipe_before = ProcessingRecipe.from_dict(session.recipe.to_dict())
            
            # Update recipe
            for key, value in recipe_update.items():
                if hasattr(session.recipe, key):
                    setattr(session.recipe, key, value)
            
            # Create description for the change
            field_names = list(recipe_update.keys())
            description = f"Updated {', '.join(field_names[:3])}"
            if len(field_names) > 3:
                description += f" and {len(field_names) - 3} more fields"
            
            # Add to history
            action_id = session.history.add_action(
                action_type="recipe_change",
                description=description,
                recipe_before=recipe_before,
                recipe_after=session.recipe,
                metadata={'updated_fields': field_names}
            )
            
            session.updated_at = datetime.utcnow()
            
            # Update preview
            if session.preview_system_session:
                self.preview_system.update_recipe(
                    session.preview_system_session,
                    session.recipe
                )
            
            # Notify subscribers
            self._broadcast_update(session_id, 'recipe_updated', {
                'recipe_id': session.recipe_id,
                'updated_fields': field_names,
                'action_id': action_id,
                'can_undo': session.history.can_undo(),
                'can_redo': session.history.can_redo()
            })
            
            return True
    
    def undo(self, session_id: str) -> bool:
        """
        Undo the last action in a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            # Attempt undo
            previous_recipe = session.history.undo()
            if not previous_recipe:
                return False
            
            # Update session recipe
            session.recipe = previous_recipe
            session.updated_at = datetime.utcnow()
            
            # Update preview
            if session.preview_system_session:
                self.preview_system.update_recipe(
                    session.preview_system_session,
                    session.recipe
                )
            
            # Notify subscribers
            self._broadcast_update(session_id, 'undo_completed', {
                'recipe_id': session.recipe_id,
                'can_undo': session.history.can_undo(),
                'can_redo': session.history.can_redo()
            })
            
            logger.debug(f"Undo completed for session {session_id}")
            return True
    
    def redo(self, session_id: str) -> bool:
        """
        Redo the next action in a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            # Attempt redo
            next_recipe = session.history.redo()
            if not next_recipe:
                return False
            
            # Update session recipe
            session.recipe = next_recipe
            session.updated_at = datetime.utcnow()
            
            # Update preview
            if session.preview_system_session:
                self.preview_system.update_recipe(
                    session.preview_system_session,
                    session.recipe
                )
            
            # Notify subscribers
            self._broadcast_update(session_id, 'redo_completed', {
                'recipe_id': session.recipe_id,
                'can_undo': session.history.can_undo(),
                'can_redo': session.history.can_redo()
            })
            
            logger.debug(f"Redo completed for session {session_id}")
            return True
    
    def get_history_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get history summary for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            History summary or None if session not found
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return None
            
            return session.history.get_history_summary()
    
    def create_snapshot(self, session_id: str, description: str) -> Optional[str]:
        """
        Create a snapshot of the current session state.
        
        Args:
            session_id: Session ID
            description: Snapshot description
            
        Returns:
            Snapshot ID or None if failed
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return None
            
            snapshot_id = session.history.create_snapshot(
                description,
                session.recipe.to_dict()
            )
            
            # Notify subscribers
            self._broadcast_update(session_id, 'snapshot_created', {
                'snapshot_id': snapshot_id,
                'description': description
            })
            
            logger.info(f"Created snapshot '{description}' for session {session_id}")
            return snapshot_id
    
    def restore_snapshot(self, session_id: str, snapshot_id: str) -> bool:
        """
        Restore a session from a snapshot.
        
        Args:
            session_id: Session ID
            snapshot_id: Snapshot ID to restore
            
        Returns:
            True if successful
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            # Restore from snapshot
            restored_recipe = session.history.restore_snapshot(snapshot_id)
            if not restored_recipe:
                return False
            
            # Update session recipe
            session.recipe = restored_recipe
            session.updated_at = datetime.utcnow()
            
            # Update preview
            if session.preview_system_session:
                self.preview_system.update_recipe(
                    session.preview_system_session,
                    session.recipe
                )
            
            # Notify subscribers
            self._broadcast_update(session_id, 'snapshot_restored', {
                'snapshot_id': snapshot_id,
                'recipe_id': session.recipe_id
            })
            
            logger.info(f"Restored snapshot {snapshot_id} for session {session_id}")
            return True
    
    def get_preview(self, session_id: str, level: Optional[str] = None) -> Optional[str]:
        """
        Get current preview for a session.
        
        Args:
            session_id: Session ID
            level: Optional preview level override
            
        Returns:
            Base64 encoded preview image or None
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session or not session.preview_system_session:
                return None
            
            # Get preview level
            if level:
                try:
                    preview_level = ProxyLevel(level)
                except ValueError:
                    preview_level = session.preview_level
            else:
                preview_level = session.preview_level
            
            # Get preview from system
            preview = self.preview_system.get_preview(
                session.preview_system_session,
                preview_level
            )
            
            if preview is not None:
                # Convert to base64
                return self._numpy_to_base64(preview)
            
            return None
    
    def process_high_quality(self, session_id: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Process image at full quality.
        
        Args:
            session_id: Session ID
            output_path: Optional output path
            
        Returns:
            Path to processed image or None
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return None
            
            session.update_state(SessionState.PROCESSING)
        
        try:
            # Import processor
            from ..processing.raw_processor import RawProcessor
            
            # Create processor
            processor = RawProcessor()
            
            # Load and process image
            result = processor.process_image(
                session.image_path,
                session.recipe,
                progress_callback=lambda p, m: self._processing_progress(session_id, p, m)
            )
            
            if result:
                # Save to output path
                if not output_path:
                    output_path = Path(f"./output/{session.session_id}_final.jpg")
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert and save
                img_array = (result * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='RGB')
                img.save(output_path, 'JPEG', quality=95)
                
                session.update_state(SessionState.ACTIVE)
                
                # Notify completion
                self._broadcast_update(session_id, 'processing_complete', {
                    'output_path': str(output_path)
                })
                
                return output_path
            
        except Exception as e:
            logger.error(f"Processing failed for session {session_id}: {e}")
            session.update_state(SessionState.ERROR)
            self._broadcast_update(session_id, 'processing_error', {
                'error': str(e)
            })
        
        return None
    
    def close_session(self, session_id: str) -> bool:
        """Close an editing session."""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            # Cleanup preview system session
            if session.preview_system_session:
                self.preview_system.cleanup_session(session.preview_system_session)
            
            # Update state
            session.update_state(SessionState.CLOSED)
            
            # Remove from active sessions
            del self.sessions[session_id]
            
            logger.info(f"Closed session {session_id}")
            return True
    
    def cleanup_inactive_sessions(self, inactive_hours: int = 2):
        """Clean up inactive sessions."""
        cutoff_time = datetime.utcnow() - timedelta(hours=inactive_hours)
        
        with self._lock:
            to_close = []
            for session_id, session in self.sessions.items():
                if session.updated_at < cutoff_time:
                    to_close.append(session_id)
            
            for session_id in to_close:
                self.close_session(session_id)
        
        if to_close:
            logger.info(f"Cleaned up {len(to_close)} inactive sessions")
    
    def subscribe_to_session(self, session_id: str, subscriber_id: str) -> bool:
        """Subscribe to session updates."""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.subscribers.add(subscriber_id)
                return True
            return False
    
    def unsubscribe_from_session(self, session_id: str, subscriber_id: str) -> bool:
        """Unsubscribe from session updates."""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.subscribers.discard(subscriber_id)
                return True
            return False
    
    def set_websocket_callback(self, callback: Callable):
        """Set callback for WebSocket broadcasts."""
        self.ws_callback = callback
    
    def _start_preview_session(self, session: EditingSession):
        """Start preview system session for editing session."""
        def preview_callback(preview_image: np.ndarray):
            """Handle preview updates."""
            # Convert to base64
            preview_b64 = self._numpy_to_base64(preview_image)
            
            # Create update message
            update = PreviewUpdate(
                session_id=session.session_id,
                preview_level=session.preview_level.value,
                image_data=preview_b64,
                metadata={'shape': preview_image.shape}
            )
            
            # Broadcast to subscribers
            self._broadcast_preview(session.session_id, update)
        
        # Start preview session
        session.preview_system_session = self.preview_system.load_image_for_editing(
            session.image_path,
            session.recipe,
            preview_callback,
            session.preview_level
        )
        
        session.update_state(SessionState.ACTIVE)
    
    def _processing_progress(self, session_id: str, progress: float, message: str):
        """Handle processing progress updates."""
        update = ProcessingProgress(
            session_id=session_id,
            stage='processing',
            progress=progress * 100,
            message=message
        )
        
        self._broadcast_progress(session_id, update)
    
    def _broadcast_preview(self, session_id: str, update: PreviewUpdate):
        """Broadcast preview update to subscribers."""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            # Call WebSocket callback if set
            if self.ws_callback:
                for subscriber_id in session.subscribers:
                    self.ws_callback(subscriber_id, update.to_dict())
    
    def _broadcast_progress(self, session_id: str, update: ProcessingProgress):
        """Broadcast progress update to subscribers."""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            # Call WebSocket callback if set
            if self.ws_callback:
                for subscriber_id in session.subscribers:
                    self.ws_callback(subscriber_id, update.to_dict())
    
    def _broadcast_update(self, session_id: str, event_type: str, data: Dict[str, Any]):
        """Broadcast generic update to subscribers."""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            message = {
                'type': event_type,
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': data
            }
            
            # Call WebSocket callback if set
            if self.ws_callback:
                for subscriber_id in session.subscribers:
                    self.ws_callback(subscriber_id, message)
    
    def _numpy_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array to base64 encoded JPEG."""
        # Convert to uint8
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image, mode='L')
        else:
            pil_image = Image.fromarray(image, mode='RGB')
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        # Encode to base64
        return base64.b64encode(buffer.read()).decode('utf-8')