"""
API Routes for PhotoSight

Defines all RESTful endpoints for the PhotoSight API.
"""

import logging
import os
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
import uuid

from .models import (
    APIResponse, ErrorResponse, APIStatus, ProcessingRequest,
    BatchProcessingRequest, RecipeValidator
)
from .auth import UserSession
from .album_routes import register_album_routes
from .photo_routes import register_photo_routes
from .web_routes import register_web_routes

logger = logging.getLogger(__name__)


def register_routes(app):
    """Register all API routes with the Flask app."""
    
    # Create API blueprint
    api = Blueprint('api', __name__, url_prefix='/api/v1')
    
    # Authentication endpoints
    @api.route('/auth/login', methods=['POST'])
    def login():
        """Authenticate and create session."""
        data = request.get_json()
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify(ErrorResponse(
                message="API key required",
                error_code="MISSING_API_KEY"
            ).to_dict()), 400
        
        # Create session
        session = app.auth.create_session(api_key)
        if not session:
            return jsonify(ErrorResponse(
                message="Invalid API key",
                error_code="INVALID_API_KEY"
            ).to_dict()), 401
        
        return jsonify(APIResponse(
            status=APIStatus.SUCCESS,
            data={
                'token': session.session_token,
                'expires_in': app.auth.token_expiry_hours * 3600,
                'permissions': list(session.permissions)
            }
        ).to_dict())
    
    @api.route('/auth/logout', methods=['POST'])
    def logout():
        """Revoke current session."""
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if app.auth.revoke_session(token):
            return jsonify(APIResponse(
                status=APIStatus.SUCCESS,
                message="Session revoked"
            ).to_dict())
        else:
            return jsonify(ErrorResponse(
                message="Session not found",
                error_code="SESSION_NOT_FOUND"
            ).to_dict()), 404
    
    @api.route('/auth/refresh', methods=['POST'])
    def refresh_token():
        """Refresh authentication token."""
        # Current session is in request context
        session = request.user_session
        
        # Create new token
        new_session = app.auth.create_session(session.api_key, session.permissions)
        
        # Revoke old session
        app.auth.revoke_session(session.session_token)
        
        return jsonify(APIResponse(
            status=APIStatus.SUCCESS,
            data={
                'token': new_session.session_token,
                'expires_in': app.auth.token_expiry_hours * 3600
            }
        ).to_dict())
    
    # Session management endpoints
    @api.route('/sessions', methods=['POST'])
    def create_session():
        """Create new editing session."""
        # Check permission
        if not app.auth.check_permission(request.user_session, 'sessions.create'):
            return jsonify(ErrorResponse(
                message="Permission denied",
                error_code="PERMISSION_DENIED"
            ).to_dict()), 403
        
        data = request.get_json()
        image_path = data.get('image_path')
        initial_recipe = data.get('recipe')
        
        if not image_path:
            return jsonify(ErrorResponse(
                message="Image path required",
                error_code="MISSING_IMAGE_PATH"
            ).to_dict()), 400
        
        # Validate recipe if provided
        if initial_recipe:
            valid, error = RecipeValidator.validate(initial_recipe)
            if not valid:
                return jsonify(ErrorResponse(
                    message=f"Invalid recipe: {error}",
                    error_code="INVALID_RECIPE"
                ).to_dict()), 400
        
        # Create session
        try:
            session = app.session_manager.create_session(
                Path(image_path),
                request.user_session.user_id,
                initial_recipe
            )
            
            return jsonify(APIResponse(
                status=APIStatus.SUCCESS,
                data=session.to_session_info().to_dict()
            ).to_dict())
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return jsonify(ErrorResponse(
                message="Failed to create session",
                error_code="SESSION_CREATION_FAILED",
                error_details={'error': str(e)}
            ).to_dict()), 500
    
    @api.route('/sessions/<session_id>', methods=['GET'])
    def get_session(session_id):
        """Get session information."""
        session = app.session_manager.get_session(session_id)
        
        if not session:
            return jsonify(ErrorResponse(
                message="Session not found",
                error_code="SESSION_NOT_FOUND"
            ).to_dict()), 404
        
        # Check if user owns this session
        if session.user_id != request.user_session.user_id:
            return jsonify(ErrorResponse(
                message="Access denied",
                error_code="ACCESS_DENIED"
            ).to_dict()), 403
        
        return jsonify(APIResponse(
            status=APIStatus.SUCCESS,
            data=session.to_session_info().to_dict()
        ).to_dict())
    
    @api.route('/sessions/<session_id>', methods=['DELETE'])
    def close_session(session_id):
        """Close editing session."""
        session = app.session_manager.get_session(session_id)
        
        if not session:
            return jsonify(ErrorResponse(
                message="Session not found",
                error_code="SESSION_NOT_FOUND"
            ).to_dict()), 404
        
        # Check if user owns this session
        if session.user_id != request.user_session.user_id:
            return jsonify(ErrorResponse(
                message="Access denied",
                error_code="ACCESS_DENIED"
            ).to_dict()), 403
        
        if app.session_manager.close_session(session_id):
            return jsonify(APIResponse(
                status=APIStatus.SUCCESS,
                message="Session closed"
            ).to_dict())
        else:
            return jsonify(ErrorResponse(
                message="Failed to close session",
                error_code="SESSION_CLOSE_FAILED"
            ).to_dict()), 500
    
    # History management endpoints
    @api.route('/sessions/<session_id>/undo', methods=['POST'])
    def undo_action(session_id):
        """Undo the last action in a session."""
        # Check permission
        if not app.auth.check_permission(request.user_session, 'sessions.modify'):
            return jsonify(ErrorResponse(
                message="Permission denied",
                error_code="PERMISSION_DENIED"
            ).to_dict()), 403
        
        # Get session
        session = app.session_manager.get_session(session_id)
        if not session:
            return jsonify(ErrorResponse(
                message="Session not found",
                error_code="SESSION_NOT_FOUND"
            ).to_dict()), 404
        
        # Check ownership
        if session.user_id != request.user_session.user_id:
            return jsonify(ErrorResponse(
                message="Access denied",
                error_code="ACCESS_DENIED"
            ).to_dict()), 403
        
        # Perform undo
        if app.session_manager.undo(session_id):
            # Get updated session info
            updated_session = app.session_manager.get_session(session_id)
            history_summary = app.session_manager.get_history_summary(session_id)
            
            return jsonify(APIResponse(
                status=APIStatus.SUCCESS,
                data={
                    'session_id': session_id,
                    'can_undo': history_summary.get('can_undo', False),
                    'can_redo': history_summary.get('can_redo', False),
                    'total_actions': history_summary.get('total_actions', 0)
                }
            ).to_dict())
        else:
            return jsonify(ErrorResponse(
                message="Nothing to undo",
                error_code="NOTHING_TO_UNDO"
            ).to_dict()), 400
    
    @api.route('/sessions/<session_id>/redo', methods=['POST'])
    def redo_action(session_id):
        """Redo the next action in a session."""
        # Check permission
        if not app.auth.check_permission(request.user_session, 'sessions.modify'):
            return jsonify(ErrorResponse(
                message="Permission denied",
                error_code="PERMISSION_DENIED"
            ).to_dict()), 403
        
        # Get session
        session = app.session_manager.get_session(session_id)
        if not session:
            return jsonify(ErrorResponse(
                message="Session not found",
                error_code="SESSION_NOT_FOUND"
            ).to_dict()), 404
        
        # Check ownership
        if session.user_id != request.user_session.user_id:
            return jsonify(ErrorResponse(
                message="Access denied",
                error_code="ACCESS_DENIED"
            ).to_dict()), 403
        
        # Perform redo
        if app.session_manager.redo(session_id):
            # Get updated session info
            updated_session = app.session_manager.get_session(session_id)
            history_summary = app.session_manager.get_history_summary(session_id)
            
            return jsonify(APIResponse(
                status=APIStatus.SUCCESS,
                data={
                    'session_id': session_id,
                    'can_undo': history_summary.get('can_undo', False),
                    'can_redo': history_summary.get('can_redo', False),
                    'total_actions': history_summary.get('total_actions', 0)
                }
            ).to_dict())
        else:
            return jsonify(ErrorResponse(
                message="Nothing to redo",
                error_code="NOTHING_TO_REDO"
            ).to_dict()), 400
    
    @api.route('/sessions/<session_id>/history', methods=['GET'])
    def get_session_history(session_id):
        """Get session history summary."""
        # Check permission
        if not app.auth.check_permission(request.user_session, 'sessions.read'):
            return jsonify(ErrorResponse(
                message="Permission denied",
                error_code="PERMISSION_DENIED"
            ).to_dict()), 403
        
        # Get session
        session = app.session_manager.get_session(session_id)
        if not session:
            return jsonify(ErrorResponse(
                message="Session not found",
                error_code="SESSION_NOT_FOUND"
            ).to_dict()), 404
        
        # Check ownership
        if session.user_id != request.user_session.user_id:
            return jsonify(ErrorResponse(
                message="Access denied",
                error_code="ACCESS_DENIED"
            ).to_dict()), 403
        
        # Get history summary
        history_summary = app.session_manager.get_history_summary(session_id)
        
        return jsonify(APIResponse(
            status=APIStatus.SUCCESS,
            data=history_summary
        ).to_dict())
    
    @api.route('/sessions/<session_id>/snapshots', methods=['POST'])
    def create_session_snapshot(session_id):
        """Create a snapshot of the current session state."""
        # Check permission
        if not app.auth.check_permission(request.user_session, 'sessions.modify'):
            return jsonify(ErrorResponse(
                message="Permission denied",
                error_code="PERMISSION_DENIED"
            ).to_dict()), 403
        
        # Get session
        session = app.session_manager.get_session(session_id)
        if not session:
            return jsonify(ErrorResponse(
                message="Session not found",
                error_code="SESSION_NOT_FOUND"
            ).to_dict()), 404
        
        # Check ownership
        if session.user_id != request.user_session.user_id:
            return jsonify(ErrorResponse(
                message="Access denied",
                error_code="ACCESS_DENIED"
            ).to_dict()), 403
        
        # Get description from request
        data = request.get_json() or {}
        description = data.get('description', f"Snapshot at {datetime.now().strftime('%H:%M:%S')}")
        
        # Create snapshot
        snapshot_id = app.session_manager.create_snapshot(session_id, description)
        
        if snapshot_id:
            return jsonify(APIResponse(
                status=APIStatus.SUCCESS,
                data={
                    'snapshot_id': snapshot_id,
                    'description': description,
                    'session_id': session_id
                }
            ).to_dict())
        else:
            return jsonify(ErrorResponse(
                message="Failed to create snapshot",
                error_code="SNAPSHOT_FAILED"
            ).to_dict()), 500
    
    @api.route('/sessions/<session_id>/snapshots/<snapshot_id>/restore', methods=['POST'])
    def restore_session_snapshot(session_id, snapshot_id):
        """Restore a session from a snapshot."""
        # Check permission
        if not app.auth.check_permission(request.user_session, 'sessions.modify'):
            return jsonify(ErrorResponse(
                message="Permission denied",
                error_code="PERMISSION_DENIED"
            ).to_dict()), 403
        
        # Get session
        session = app.session_manager.get_session(session_id)
        if not session:
            return jsonify(ErrorResponse(
                message="Session not found",
                error_code="SESSION_NOT_FOUND"
            ).to_dict()), 404
        
        # Check ownership
        if session.user_id != request.user_session.user_id:
            return jsonify(ErrorResponse(
                message="Access denied",
                error_code="ACCESS_DENIED"
            ).to_dict()), 403
        
        # Restore snapshot
        if app.session_manager.restore_snapshot(session_id, snapshot_id):
            return jsonify(APIResponse(
                status=APIStatus.SUCCESS,
                data={
                    'snapshot_id': snapshot_id,
                    'session_id': session_id,
                    'message': 'Snapshot restored successfully'
                }
            ).to_dict())
        else:
            return jsonify(ErrorResponse(
                message="Failed to restore snapshot",
                error_code="RESTORE_FAILED"
            ).to_dict()), 400
    
    # Image processing endpoints
    @api.route('/process/preview', methods=['POST'])
    def update_preview():
        """Update processing and get preview."""
        # Check permission
        if not app.auth.check_permission(request.user_session, 'images.process'):
            return jsonify(ErrorResponse(
                message="Permission denied",
                error_code="PERMISSION_DENIED"
            ).to_dict()), 403
        
        data = request.get_json()
        
        # Validate request
        try:
            process_request = ProcessingRequest.from_dict(data)
        except KeyError as e:
            return jsonify(ErrorResponse(
                message=f"Missing required field: {e}",
                error_code="INVALID_REQUEST"
            ).to_dict()), 400
        
        # Validate recipe
        valid, error = RecipeValidator.validate(process_request.recipe)
        if not valid:
            return jsonify(ErrorResponse(
                message=f"Invalid recipe: {error}",
                error_code="INVALID_RECIPE"
            ).to_dict()), 400
        
        # Get session
        session = app.session_manager.get_session(process_request.session_id)
        if not session:
            return jsonify(ErrorResponse(
                message="Session not found",
                error_code="SESSION_NOT_FOUND"
            ).to_dict()), 404
        
        # Update recipe
        if not app.session_manager.update_recipe(process_request.session_id, process_request.recipe):
            return jsonify(ErrorResponse(
                message="Failed to update recipe",
                error_code="UPDATE_FAILED"
            ).to_dict()), 500
        
        # Get preview
        preview_data = app.session_manager.get_preview(
            process_request.session_id,
            process_request.preview_level
        )
        
        return jsonify(APIResponse(
            status=APIStatus.SUCCESS,
            data={
                'session_id': process_request.session_id,
                'preview_level': process_request.preview_level,
                'preview_data': preview_data
            }
        ).to_dict())
    
    @api.route('/process/export/<session_id>', methods=['POST'])
    def export_image(session_id):
        """Export processed image at full quality."""
        # Check permission
        if not app.auth.check_permission(request.user_session, 'images.process'):
            return jsonify(ErrorResponse(
                message="Permission denied",
                error_code="PERMISSION_DENIED"
            ).to_dict()), 403
        
        # Get session
        session = app.session_manager.get_session(session_id)
        if not session:
            return jsonify(ErrorResponse(
                message="Session not found",
                error_code="SESSION_NOT_FOUND"
            ).to_dict()), 404
        
        # Check ownership
        if session.user_id != request.user_session.user_id:
            return jsonify(ErrorResponse(
                message="Access denied",
                error_code="ACCESS_DENIED"
            ).to_dict()), 403
        
        # Process at full quality
        output_path = app.session_manager.process_high_quality(session_id)
        
        if output_path:
            return jsonify(APIResponse(
                status=APIStatus.SUCCESS,
                data={
                    'session_id': session_id,
                    'output_path': str(output_path),
                    'download_url': f"/api/v1/download/{output_path.name}"
                }
            ).to_dict())
        else:
            return jsonify(ErrorResponse(
                message="Export failed",
                error_code="EXPORT_FAILED"
            ).to_dict()), 500
    
    # Batch processing endpoints
    @api.route('/batch', methods=['POST'])
    def create_batch_job():
        """Create new batch processing job."""
        # Check permission
        if not app.auth.check_permission(request.user_session, 'images.process'):
            return jsonify(ErrorResponse(
                message="Permission denied",
                error_code="PERMISSION_DENIED"
            ).to_dict()), 403
        
        data = request.get_json()
        
        # Validate request
        try:
            batch_request = BatchProcessingRequest.from_dict(data)
        except KeyError as e:
            return jsonify(ErrorResponse(
                message=f"Missing required field: {e}",
                error_code="INVALID_REQUEST"
            ).to_dict()), 400
        
        # Validate recipe
        valid, error = RecipeValidator.validate(batch_request.recipe)
        if not valid:
            return jsonify(ErrorResponse(
                message=f"Invalid recipe: {error}",
                error_code="INVALID_RECIPE"
            ).to_dict()), 400
        
        # Submit job
        job_id = app.batch_processor.submit_job(batch_request)
        
        return jsonify(APIResponse(
            status=APIStatus.SUCCESS,
            data={
                'job_id': job_id,
                'status_url': f"/api/v1/batch/{job_id}"
            }
        ).to_dict())
    
    @api.route('/batch/<job_id>', methods=['GET'])
    def get_batch_status(job_id):
        """Get batch job status."""
        status = app.batch_processor.get_job_status(job_id)
        
        if not status:
            return jsonify(ErrorResponse(
                message="Job not found",
                error_code="JOB_NOT_FOUND"
            ).to_dict()), 404
        
        return jsonify(APIResponse(
            status=APIStatus.SUCCESS,
            data=status.to_dict()
        ).to_dict())
    
    @api.route('/batch/<job_id>', methods=['DELETE'])
    def cancel_batch_job(job_id):
        """Cancel batch processing job."""
        if app.batch_processor.cancel_job(job_id):
            return jsonify(APIResponse(
                status=APIStatus.SUCCESS,
                message="Job cancelled"
            ).to_dict())
        else:
            return jsonify(ErrorResponse(
                message="Job not found or already complete",
                error_code="CANNOT_CANCEL"
            ).to_dict()), 400
    
    # File upload/download endpoints
    @api.route('/upload', methods=['POST'])
    def upload_image():
        """Upload image for processing."""
        # Check permission
        if not app.auth.check_permission(request.user_session, 'images.read'):
            return jsonify(ErrorResponse(
                message="Permission denied",
                error_code="PERMISSION_DENIED"
            ).to_dict()), 403
        
        if 'file' not in request.files:
            return jsonify(ErrorResponse(
                message="No file provided",
                error_code="NO_FILE"
            ).to_dict()), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify(ErrorResponse(
                message="No file selected",
                error_code="NO_FILE"
            ).to_dict()), 400
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.raw', '.dng', '.cr2', '.nef', '.arw'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return jsonify(ErrorResponse(
                message=f"Invalid file type: {file_ext}",
                error_code="INVALID_FILE_TYPE"
            ).to_dict()), 400
        
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = app.config['UPLOAD_FOLDER'] / unique_filename
        
        file.save(str(file_path))
        
        return jsonify(APIResponse(
            status=APIStatus.SUCCESS,
            data={
                'file_path': str(file_path),
                'original_name': filename,
                'size': file_path.stat().st_size
            }
        ).to_dict())
    
    @api.route('/download/<filename>', methods=['GET'])
    def download_file(filename):
        """Download processed image."""
        # Validate filename
        filename = secure_filename(filename)
        file_path = app.config['OUTPUT_FOLDER'] / filename
        
        if not file_path.exists():
            return jsonify(ErrorResponse(
                message="File not found",
                error_code="FILE_NOT_FOUND"
            ).to_dict()), 404
        
        return send_file(
            str(file_path),
            as_attachment=True,
            download_name=filename
        )
    
    # System endpoints
    @api.route('/stats', methods=['GET'])
    def get_system_stats():
        """Get system statistics."""
        # Get preview system stats
        preview_stats = app.session_manager.preview_system.get_system_stats()
        
        # Get batch processor stats
        batch_stats = app.batch_processor.get_stats()
        
        # Get session stats
        session_stats = {
            'active_sessions': len(app.session_manager.sessions),
            'total_sessions': sum(
                1 for s in app.session_manager.sessions.values()
                if s.user_id == request.user_session.user_id
            )
        }
        
        return jsonify(APIResponse(
            status=APIStatus.SUCCESS,
            data={
                'preview_system': preview_stats,
                'batch_processor': batch_stats,
                'sessions': session_stats
            }
        ).to_dict())
    
    # Register blueprint
    app.register_blueprint(api)
    
    # Register album and photo routes
    register_album_routes(app)
    register_photo_routes(app)
    
    # Register web viewer routes
    register_web_routes(app)
    
    logger.info("API routes registered")