"""
WebSocket event emitters for real-time updates from Celery tasks
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import redis
from flask_socketio import SocketIO

logger = logging.getLogger(__name__)

# Redis client for pub/sub
redis_client = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))

# Global SocketIO instance (will be set by app)
socketio: Optional[SocketIO] = None


def init_socketio(sio: SocketIO) -> None:
    """Initialize the global SocketIO instance"""
    global socketio
    socketio = sio


def emit_progress_update(job_id: str, progress_data: Dict[str, Any]) -> None:
    """
    Emit progress update for a batch job
    
    Args:
        job_id: Batch job ID
        progress_data: Progress information
    """
    try:
        message = {
            'type': 'progress_update',
            'job_id': job_id,
            'timestamp': datetime.utcnow().isoformat(),
            **progress_data
        }
        
        # Emit via Redis pub/sub for multi-worker setup
        redis_client.publish(f'job_progress:{job_id}', json.dumps(message))
        
        # Direct emit if we have socketio instance
        if socketio:
            socketio.emit('progress_update', message, room=f'job_{job_id}')
            
    except Exception as e:
        logger.error(f"Failed to emit progress update: {e}")


def emit_preview_update(photo_id: str, preview_url: str, 
                       session_id: Optional[str] = None) -> None:
    """
    Emit preview update for a photo
    
    Args:
        photo_id: Photo ID
        preview_url: URL of the updated preview
        session_id: Optional session ID for targeted update
    """
    try:
        message = {
            'type': 'preview_update',
            'photo_id': photo_id,
            'preview_url': preview_url,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if session_id:
            message['session_id'] = session_id
        
        # Emit via Redis pub/sub
        channel = f'preview:{session_id}' if session_id else f'preview:{photo_id}'
        redis_client.publish(channel, json.dumps(message))
        
        # Direct emit if we have socketio instance
        if socketio:
            room = f'session_{session_id}' if session_id else f'photo_{photo_id}'
            socketio.emit('preview_update', message, room=room)
            
    except Exception as e:
        logger.error(f"Failed to emit preview update: {e}")


def emit_processing_complete(photo_id: str, result: Dict[str, Any]) -> None:
    """
    Emit processing completion event
    
    Args:
        photo_id: Photo ID
        result: Processing result
    """
    try:
        message = {
            'type': 'processing_complete',
            'photo_id': photo_id,
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Emit via Redis pub/sub
        redis_client.publish(f'processing:{photo_id}', json.dumps(message))
        
        # Direct emit if we have socketio instance
        if socketio:
            socketio.emit('processing_complete', message, room=f'photo_{photo_id}')
            
    except Exception as e:
        logger.error(f"Failed to emit processing complete: {e}")


def emit_batch_complete(job_id: str, summary: Dict[str, Any]) -> None:
    """
    Emit batch job completion event
    
    Args:
        job_id: Batch job ID
        summary: Job summary
    """
    try:
        message = {
            'type': 'batch_complete',
            'job_id': job_id,
            'summary': summary,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Emit via Redis pub/sub
        redis_client.publish(f'batch:{job_id}', json.dumps(message))
        
        # Direct emit if we have socketio instance
        if socketio:
            socketio.emit('batch_complete', message, room=f'job_{job_id}')
            
    except Exception as e:
        logger.error(f"Failed to emit batch complete: {e}")


def emit_error(entity_type: str, entity_id: str, error: str, 
               details: Optional[Dict[str, Any]] = None) -> None:
    """
    Emit error event
    
    Args:
        entity_type: Type of entity (photo, job, session)
        entity_id: Entity ID
        error: Error message
        details: Optional error details
    """
    try:
        message = {
            'type': 'error',
            'entity_type': entity_type,
            'entity_id': entity_id,
            'error': error,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Emit via Redis pub/sub
        redis_client.publish(f'error:{entity_type}:{entity_id}', json.dumps(message))
        
        # Direct emit if we have socketio instance
        if socketio:
            room = f'{entity_type}_{entity_id}'
            socketio.emit('error', message, room=room)
            
    except Exception as e:
        logger.error(f"Failed to emit error: {e}")


def emit_session_update(session_id: str, update_type: str, 
                       data: Dict[str, Any]) -> None:
    """
    Emit session update event
    
    Args:
        session_id: Session ID
        update_type: Type of update (recipe_change, undo, redo, etc.)
        data: Update data
    """
    try:
        message = {
            'type': 'session_update',
            'session_id': session_id,
            'update_type': update_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Emit via Redis pub/sub
        redis_client.publish(f'session:{session_id}', json.dumps(message))
        
        # Direct emit if we have socketio instance
        if socketio:
            socketio.emit('session_update', message, room=f'session_{session_id}')
            
    except Exception as e:
        logger.error(f"Failed to emit session update: {e}")


class RedisWebSocketBridge:
    """Bridge between Redis pub/sub and WebSocket for multi-worker setups"""
    
    def __init__(self, socketio_instance: SocketIO):
        self.socketio = socketio_instance
        self.pubsub = redis_client.pubsub()
        self.running = False
    
    def start(self):
        """Start listening to Redis channels"""
        self.running = True
        
        # Subscribe to relevant channels
        patterns = [
            'job_progress:*',
            'preview:*',
            'processing:*',
            'batch:*',
            'error:*',
            'session:*'
        ]
        
        for pattern in patterns:
            self.pubsub.psubscribe(pattern)
        
        # Start listening in background thread
        self.socketio.start_background_task(self._listen)
    
    def stop(self):
        """Stop listening"""
        self.running = False
        self.pubsub.close()
    
    def _listen(self):
        """Listen for Redis messages and forward to WebSocket"""
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=1)
                if message and message['type'] == 'pmessage':
                    channel = message['channel'].decode('utf-8')
                    data = json.loads(message['data'])
                    
                    # Determine room based on channel
                    room = self._get_room_from_channel(channel)
                    event_type = data.get('type', 'update')
                    
                    # Emit to appropriate room
                    self.socketio.emit(event_type, data, room=room)
                    
            except Exception as e:
                logger.error(f"Redis bridge error: {e}")
    
    def _get_room_from_channel(self, channel: str) -> str:
        """Extract room name from Redis channel"""
        parts = channel.split(':')
        if len(parts) >= 2:
            entity_type = parts[0]
            entity_id = parts[1]
            return f'{entity_type}_{entity_id}'
        return 'global'