"""
Chunked file upload handler for large RAW files
"""

import os
import uuid
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
import json

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import redis

from .auth import require_auth
from .models_pydantic import ErrorResponse, APIResponse
from ..config.security import check_file_upload_security

logger = logging.getLogger(__name__)

# Blueprint for chunked uploads
chunked_upload = Blueprint('chunked_upload', __name__)

# Redis client for upload session management
redis_client = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))

# Upload session TTL (24 hours)
UPLOAD_SESSION_TTL = 86400

# Maximum chunk size (10MB)
MAX_CHUNK_SIZE = 10 * 1024 * 1024


class ChunkedUploadHandler:
    """Handle chunked file uploads with resume capability"""
    
    def __init__(self, upload_dir: Path):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.upload_dir / 'chunks'
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
    
    def init_upload(self, filename: str, total_size: int, 
                   total_chunks: int, file_hash: Optional[str] = None) -> str:
        """
        Initialize a new chunked upload session
        
        Args:
            filename: Original filename
            total_size: Total file size in bytes
            total_chunks: Total number of chunks
            file_hash: Optional file hash for integrity check
            
        Returns:
            Upload session ID
        """
        # Additional security: validate filename again
        secure_name = secure_filename(filename)
        if not secure_name or secure_name != filename:
            logger.warning(f"Filename was sanitized: '{filename}' -> '{secure_name}'")
            filename = secure_name
        
        upload_id = str(uuid.uuid4())
        session_data = {
            'upload_id': upload_id,
            'filename': filename,
            'total_size': total_size,
            'total_chunks': total_chunks,
            'uploaded_chunks': [],
            'file_hash': file_hash,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        # Create upload directory with secure permissions
        upload_path = self.chunks_dir / upload_id
        upload_path.mkdir(mode=0o750, exist_ok=True)  # Restricted permissions
        
        # Store session in Redis
        redis_client.setex(
            f'upload:{upload_id}',
            UPLOAD_SESSION_TTL,
            json.dumps(session_data)
        )
        
        logger.info(f"Initialized upload session {upload_id} for {filename}")
        return upload_id
    
    def upload_chunk(self, upload_id: str, chunk_index: int, 
                    chunk_data: bytes) -> Dict[str, Any]:
        """
        Upload a single chunk
        
        Args:
            upload_id: Upload session ID
            chunk_index: Index of this chunk (0-based)
            chunk_data: Chunk data
            
        Returns:
            Upload progress information
        """
        # Get session data
        session_data = self._get_session(upload_id)
        if not session_data:
            raise ValueError(f"Invalid upload session: {upload_id}")
        
        if session_data['status'] != 'active':
            raise ValueError(f"Upload session {upload_id} is not active")
        
        # Validate chunk
        if chunk_index >= session_data['total_chunks']:
            raise ValueError(f"Invalid chunk index: {chunk_index}")
        
        if len(chunk_data) > MAX_CHUNK_SIZE:
            raise ValueError(f"Chunk size exceeds maximum: {len(chunk_data)} > {MAX_CHUNK_SIZE}")
        
        # Check if chunk already uploaded
        if chunk_index in session_data['uploaded_chunks']:
            logger.info(f"Chunk {chunk_index} already uploaded for {upload_id}")
            return self._get_progress(session_data)
        
        # Save chunk
        chunk_path = self.chunks_dir / upload_id / f"chunk_{chunk_index:06d}"
        chunk_path.write_bytes(chunk_data)
        
        # Update session
        session_data['uploaded_chunks'].append(chunk_index)
        session_data['uploaded_chunks'].sort()
        session_data['last_update'] = datetime.utcnow().isoformat()
        
        # Check if upload is complete
        if len(session_data['uploaded_chunks']) == session_data['total_chunks']:
            session_data['status'] = 'assembling'
        
        # Update session in Redis
        redis_client.setex(
            f'upload:{upload_id}',
            UPLOAD_SESSION_TTL,
            json.dumps(session_data)
        )
        
        return self._get_progress(session_data)
    
    def complete_upload(self, upload_id: str) -> Tuple[str, str]:
        """
        Complete the upload by assembling all chunks
        
        Args:
            upload_id: Upload session ID
            
        Returns:
            Tuple of (final_path, file_hash)
        """
        session_data = self._get_session(upload_id)
        if not session_data:
            raise ValueError(f"Invalid upload session: {upload_id}")
        
        if len(session_data['uploaded_chunks']) != session_data['total_chunks']:
            missing = set(range(session_data['total_chunks'])) - set(session_data['uploaded_chunks'])
            raise ValueError(f"Missing chunks: {sorted(missing)}")
        
        # Assemble file with secure path and permissions
        safe_filename = secure_filename(session_data['filename'])
        final_path = self.upload_dir / f"{upload_id}_{safe_filename}"
        
        # Ensure we're writing within the upload directory (prevent path traversal)
        if not str(final_path.resolve()).startswith(str(self.upload_dir.resolve())):
            raise ValueError("Invalid file path detected")
        
        hasher = hashlib.sha256()
        
        with open(final_path, 'wb') as output:
            for i in range(session_data['total_chunks']):
                chunk_path = self.chunks_dir / upload_id / f"chunk_{i:06d}"
                
                # Validate chunk path (prevent directory traversal)
                if not str(chunk_path.resolve()).startswith(str(self.chunks_dir.resolve())):
                    raise ValueError(f"Invalid chunk path detected: {chunk_path}")
                
                chunk_data = chunk_path.read_bytes()
                output.write(chunk_data)
                hasher.update(chunk_data)
        
        # Set secure permissions on final file
        os.chmod(final_path, 0o640)  # Owner read/write, group read only
        
        file_hash = hasher.hexdigest()
        
        # Verify hash if provided
        if session_data.get('file_hash') and session_data['file_hash'] != file_hash:
            final_path.unlink()  # Delete assembled file
            raise ValueError(f"File hash mismatch: expected {session_data['file_hash']}, got {file_hash}")
        
        # Clean up chunks
        self._cleanup_chunks(upload_id)
        
        # Update session
        session_data['status'] = 'completed'
        session_data['final_path'] = str(final_path)
        session_data['completed_at'] = datetime.utcnow().isoformat()
        session_data['file_hash'] = file_hash
        
        redis_client.setex(
            f'upload:{upload_id}',
            3600,  # Keep completed session for 1 hour
            json.dumps(session_data)
        )
        
        logger.info(f"Completed upload {upload_id}: {final_path}")
        return str(final_path), file_hash
    
    def get_upload_status(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an upload"""
        session_data = self._get_session(upload_id)
        if not session_data:
            return None
        
        return self._get_progress(session_data)
    
    def cancel_upload(self, upload_id: str) -> bool:
        """Cancel an upload and clean up"""
        session_data = self._get_session(upload_id)
        if not session_data:
            return False
        
        # Clean up chunks
        self._cleanup_chunks(upload_id)
        
        # Delete session
        redis_client.delete(f'upload:{upload_id}')
        
        logger.info(f"Cancelled upload {upload_id}")
        return True
    
    def resume_upload(self, upload_id: str) -> Dict[str, Any]:
        """Get information needed to resume an upload"""
        session_data = self._get_session(upload_id)
        if not session_data:
            raise ValueError(f"Invalid upload session: {upload_id}")
        
        if session_data['status'] != 'active':
            raise ValueError(f"Cannot resume completed upload: {upload_id}")
        
        # Find missing chunks
        uploaded = set(session_data['uploaded_chunks'])
        missing = sorted(set(range(session_data['total_chunks'])) - uploaded)
        
        return {
            'upload_id': upload_id,
            'uploaded_chunks': sorted(uploaded),
            'missing_chunks': missing,
            'total_chunks': session_data['total_chunks'],
            'progress': len(uploaded) / session_data['total_chunks'] * 100
        }
    
    def _get_session(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get upload session from Redis"""
        data = redis_client.get(f'upload:{upload_id}')
        if not data:
            return None
        return json.loads(data)
    
    def _get_progress(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate upload progress"""
        uploaded = len(session_data['uploaded_chunks'])
        total = session_data['total_chunks']
        
        return {
            'upload_id': session_data['upload_id'],
            'status': session_data['status'],
            'progress': uploaded / total * 100 if total > 0 else 0,
            'uploaded_chunks': uploaded,
            'total_chunks': total,
            'uploaded_bytes': uploaded * (session_data['total_size'] // total),
            'total_bytes': session_data['total_size']
        }
    
    def _cleanup_chunks(self, upload_id: str) -> None:
        """Clean up chunk files"""
        chunk_dir = self.chunks_dir / upload_id
        if chunk_dir.exists():
            import shutil
            shutil.rmtree(chunk_dir)


# Initialize handler
upload_handler = ChunkedUploadHandler(Path(os.environ.get('UPLOAD_DIR', './uploads')))


# Routes

@chunked_upload.route('/init', methods=['POST'])
@require_auth
def init_chunked_upload():
    """Initialize a chunked upload session"""
    try:
        data = request.json
        
        filename = data.get('filename')
        total_size = data.get('total_size')
        total_chunks = data.get('total_chunks')
        file_hash = data.get('file_hash')
        
        if not all([filename, total_size, total_chunks]):
            return jsonify(ErrorResponse(
                error="missing_parameters",
                message="filename, total_size, and total_chunks are required"
            ).dict()), 400
        
        # Perform comprehensive file upload security check
        security_check = check_file_upload_security(
            filename=filename,
            content_type=request.headers.get('Content-Type', 'application/octet-stream'),
            file_size=total_size
        )
        
        if not security_check['valid']:
            return jsonify(ErrorResponse(
                error="security_validation_failed",
                message="File upload security validation failed",
                details=security_check['issues']
            ).dict()), 400
        
        # Log security warnings if any
        if security_check['warnings']:
            logger.warning(f"File upload warnings for {filename}: {security_check['warnings']}")
        
        # Use sanitized filename
        filename = security_check['sanitized_filename']
        
        upload_id = upload_handler.init_upload(
            filename=filename,
            total_size=total_size,
            total_chunks=total_chunks,
            file_hash=file_hash
        )
        
        return jsonify(APIResponse(
            status='success',
            data={
                'upload_id': upload_id,
                'chunk_size': MAX_CHUNK_SIZE
            }
        ).dict())
        
    except Exception as e:
        logger.error(f"Failed to init upload: {e}")
        return jsonify(ErrorResponse(
            error="init_failed",
            message=str(e)
        ).dict()), 500


@chunked_upload.route('/<upload_id>/chunk/<int:chunk_index>', methods=['PUT'])
@require_auth
def upload_chunk(upload_id: str, chunk_index: int):
    """Upload a single chunk"""
    try:
        chunk_data = request.data
        
        if not chunk_data:
            return jsonify(ErrorResponse(
                error="no_data",
                message="No chunk data provided"
            ).dict()), 400
        
        progress = upload_handler.upload_chunk(upload_id, chunk_index, chunk_data)
        
        return jsonify(APIResponse(
            status='success',
            data=progress
        ).dict())
        
    except ValueError as e:
        return jsonify(ErrorResponse(
            error="upload_error",
            message=str(e)
        ).dict()), 400
    except Exception as e:
        logger.error(f"Failed to upload chunk: {e}")
        return jsonify(ErrorResponse(
            error="upload_failed",
            message=str(e)
        ).dict()), 500


@chunked_upload.route('/<upload_id>/complete', methods=['POST'])
@require_auth
def complete_chunked_upload(upload_id: str):
    """Complete the upload and assemble chunks"""
    try:
        final_path, file_hash = upload_handler.complete_upload(upload_id)
        
        # Create photo record
        photo_id = str(uuid.uuid4())
        
        # Would create database record here
        
        return jsonify(APIResponse(
            status='success',
            data={
                'photo_id': photo_id,
                'file_path': final_path,
                'file_hash': file_hash
            }
        ).dict())
        
    except ValueError as e:
        return jsonify(ErrorResponse(
            error="completion_error",
            message=str(e)
        ).dict()), 400
    except Exception as e:
        logger.error(f"Failed to complete upload: {e}")
        return jsonify(ErrorResponse(
            error="completion_failed",
            message=str(e)
        ).dict()), 500


@chunked_upload.route('/<upload_id>/status', methods=['GET'])
@require_auth
def get_upload_status(upload_id: str):
    """Get upload status"""
    status = upload_handler.get_upload_status(upload_id)
    
    if not status:
        return jsonify(ErrorResponse(
            error="not_found",
            message=f"Upload session {upload_id} not found"
        ).dict()), 404
    
    return jsonify(APIResponse(
        status='success',
        data=status
    ).dict())


@chunked_upload.route('/<upload_id>/resume', methods=['GET'])
@require_auth
def resume_upload(upload_id: str):
    """Get information to resume an upload"""
    try:
        resume_info = upload_handler.resume_upload(upload_id)
        
        return jsonify(APIResponse(
            status='success',
            data=resume_info
        ).dict())
        
    except ValueError as e:
        return jsonify(ErrorResponse(
            error="resume_error",
            message=str(e)
        ).dict()), 400


@chunked_upload.route('/<upload_id>/cancel', methods=['DELETE'])
@require_auth
def cancel_upload(upload_id: str):
    """Cancel an upload"""
    success = upload_handler.cancel_upload(upload_id)
    
    if not success:
        return jsonify(ErrorResponse(
            error="not_found",
            message=f"Upload session {upload_id} not found"
        ).dict()), 404
    
    return jsonify(APIResponse(
        status='success',
        message=f"Upload {upload_id} cancelled"
    ).dict())