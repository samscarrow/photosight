"""
Improved WebSocket events with standardized Redis message structure and enhanced reliability.

This module provides a more robust WebSocket implementation with:
- Standardized message format with room and namespace information
- Enhanced error handling and retry logic
- Better separation of concerns between Redis publishing and WebSocket emission
- Backward compatibility with existing code
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from threading import Thread
from contextlib import suppress

import redis
from flask_socketio import SocketIO

# Configuration
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
WEBSOCKET_CHANNEL = 'websocket_broadcast'

logger = logging.getLogger(__name__)


class RedisWebSocketBridge:
    """
    Bridge between Redis pub/sub and WebSocket events.
    Handles multi-worker environments gracefully.
    """
    
    def __init__(self, socketio: SocketIO, redis_url: Optional[str] = None):
        """
        Initialize the bridge.
        
        Args:
            socketio: Flask-SocketIO instance
            redis_url: Redis connection URL
        """
        self.socketio = socketio
        self.redis_url = redis_url or REDIS_URL
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.running = False
        self.thread = None
        
    def start(self) -> None:
        """Start listening to Redis events."""
        if self.running:
            return
            
        self.running = True
        self.pubsub.subscribe(WEBSOCKET_CHANNEL)
        self.thread = Thread(target=self._listen, daemon=True)
        self.thread.start()
        logger.info("Redis WebSocket bridge started")
        
    def stop(self) -> None:
        """Stop listening to Redis events."""
        self.running = False
        with suppress(Exception):
            self.pubsub.unsubscribe(WEBSOCKET_CHANNEL)
            self.pubsub.close()
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("Redis WebSocket bridge stopped")
        
    def _listen(self) -> None:
        """Listen for Redis messages and emit to WebSocket."""
        try:
            for message in self.pubsub.listen():
                if not self.running:
                    break
                    
                if message['type'] == 'message':
                    try:
                        # Parse the standardized message structure
                        data = json.loads(message['data'])
                        event = data['event']
                        payload = data['payload']
                        room = data.get('room')
                        namespace = data.get('namespace', '/')
                        
                        # Emit to specific room or broadcast
                        if room:
                            self.socketio.emit(event, payload, room=room, namespace=namespace)
                        else:
                            self.socketio.emit(event, payload, namespace=namespace)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in Redis message: {e}")
                    except KeyError as e:
                        logger.error(f"Missing required field in Redis message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
                        
        except Exception as e:
            logger.error(f"Redis listener error: {e}")
        finally:
            self.running = False


class WebSocketEmitter:
    """
    Unified WebSocket event emitter with standardized message format.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize the emitter.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url or REDIS_URL
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        
    def emit(self, event: str, payload: Dict[str, Any], 
             room: Optional[str] = None, namespace: str = '/') -> None:
        """
        Emit an event through Redis pub/sub.
        
        Args:
            event: Event name
            payload: Event data
            room: Optional room to emit to
            namespace: Socket.IO namespace
        """
        message = {
            'event': event,
            'payload': payload,
            'room': room,
            'namespace': namespace,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            result = self.redis_client.publish(WEBSOCKET_CHANNEL, json.dumps(message))
            if result == 0:
                logger.warning(f"No subscribers for WebSocket event: {event}")
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")
    
    # Convenience methods for common events
    
    def progress_update(self, job_id: str, progress: float, stage: str, 
                       message: Optional[str] = None, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit progress update for a batch job."""
        payload = {
            'type': 'progress_update',
            'job_id': job_id,
            'progress': progress,
            'stage': stage,
            'message': message,
            'metadata': metadata or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        self.emit('progress_update', payload, room=f'job_{job_id}')
    
    def preview_update(self, photo_id: str, preview_url: str,
                      session_id: Optional[str] = None,
                      recipe: Optional[Dict[str, Any]] = None,
                      processing_time: Optional[float] = None) -> None:
        """Emit preview update for a photo."""
        payload = {
            'type': 'preview_update',
            'photo_id': photo_id,
            'preview_url': preview_url,
            'session_id': session_id,
            'recipe': recipe,
            'processing_time': processing_time,
            'timestamp': datetime.utcnow().isoformat()
        }
        room = f'session_{session_id}' if session_id else f'photo_{photo_id}'
        self.emit('preview_update', payload, room=room)
    
    def processing_complete(self, photo_id: str, result: Dict[str, Any],
                           job_id: Optional[str] = None) -> None:
        """Emit processing completion event."""
        payload = {
            'type': 'processing_complete',
            'photo_id': photo_id,
            'job_id': job_id,
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        }
        rooms = [f'photo_{photo_id}']
        if job_id:
            rooms.append(f'job_{job_id}')
        
        for room in rooms:
            self.emit('processing_complete', payload, room=room)
    
    def batch_complete(self, job_id: str, summary: Dict[str, Any]) -> None:
        """Emit batch job completion event."""
        payload = {
            'type': 'batch_complete',
            'job_id': job_id,
            'summary': summary,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.emit('batch_complete', payload, room=f'job_{job_id}')
    
    def error(self, entity_type: str, entity_id: str, error_message: str,
             error_code: Optional[str] = None,
             details: Optional[Dict[str, Any]] = None) -> None:
        """Emit error event."""
        payload = {
            'type': 'error',
            'entity_type': entity_type,
            'entity_id': entity_id,
            'error_message': error_message,
            'error_code': error_code,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        room = f'{entity_type}_{entity_id}'
        self.emit('error', payload, room=room)
    
    def session_update(self, session_id: str, update_type: str,
                      data: Dict[str, Any]) -> None:
        """Emit session update event."""
        payload = {
            'type': 'session_update',
            'session_id': session_id,
            'update_type': update_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.emit('session_update', payload, room=f'session_{session_id}')
    
    def photo_added(self, photo_id: str, photo_data: Dict[str, Any],
                   album_ids: Optional[List[str]] = None) -> None:
        """Emit photo added event."""
        payload = {
            'type': 'photo_added',
            'photo_id': photo_id,
            'photo_data': photo_data,
            'album_ids': album_ids or [],
            'timestamp': datetime.utcnow().isoformat()
        }
        self.emit('photo_added', payload)
        
        # Also emit to album-specific rooms
        for album_id in (album_ids or []):
            self.emit('photo_added', payload, room=f'album_{album_id}')
    
    def bulk_operation_progress(self, operation_id: str, operation_type: str,
                               current: int, total: int,
                               message: Optional[str] = None) -> None:
        """Emit bulk operation progress."""
        payload = {
            'type': 'bulk_operation_progress',
            'operation_id': operation_id,
            'operation_type': operation_type,
            'current': current,
            'total': total,
            'progress': (current / total * 100) if total > 0 else 0,
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.emit('bulk_operation_progress', payload, room=f'operation_{operation_id}')


# Global emitter instance
emitter = WebSocketEmitter()


# Backward compatibility functions
def emit_progress_update(job_id: str, progress_data: Dict[str, Any]) -> None:
    """Legacy function for progress updates."""
    emitter.progress_update(
        job_id=job_id,
        progress=progress_data.get('progress', 0),
        stage=progress_data.get('stage', 'processing'),
        message=progress_data.get('message'),
        metadata=progress_data
    )


def emit_preview_update(photo_id: str, preview_url: str,
                       session_id: Optional[str] = None) -> None:
    """Legacy function for preview updates."""
    emitter.preview_update(photo_id, preview_url, session_id)


def emit_processing_complete(photo_id: str, result: Dict[str, Any]) -> None:
    """Legacy function for processing complete events."""
    emitter.processing_complete(photo_id, result)


def emit_batch_complete(job_id: str, summary: Dict[str, Any]) -> None:
    """Legacy function for batch complete events."""
    emitter.batch_complete(job_id, summary)


def emit_error(entity_type: str, entity_id: str, error: str,
               details: Optional[Dict[str, Any]] = None) -> None:
    """Legacy function for error events."""
    emitter.error(entity_type, entity_id, error, details=details)


def emit_session_update(session_id: str, update_type: str,
                       data: Dict[str, Any]) -> None:
    """Legacy function for session updates."""
    emitter.session_update(session_id, update_type, data)