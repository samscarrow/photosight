"""
PhotoSight API Application

Main Flask application with RESTful endpoints and WebSocket support.
"""

import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.middleware.proxy_fix import ProxyFix
import uuid
from datetime import datetime
from typing import Dict, Optional, Any

from .models import (
    APIResponse, ErrorResponse, SessionInfo, ProcessingRequest,
    BatchProcessingRequest, BatchJobStatus, PreviewUpdate,
    ProcessingProgress, SystemStats, APIStatus, SessionState,
    RecipeValidator, WS_MESSAGE_TYPES
)
from .auth import APIAuth, RateLimiter
from .session_manager import SessionManager
from .batch_processor import BatchProcessor
from .routes import register_routes
from .swagger_config import init_swagger

logger = logging.getLogger(__name__)


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """
    Create and configure the PhotoSight API application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Default configuration
    app.config.update(
        SECRET_KEY=os.environ.get('PHOTOSIGHT_SECRET_KEY', 'dev-secret-key'),
        MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB max file size
        UPLOAD_FOLDER=Path('./uploads'),
        OUTPUT_FOLDER=Path('./output'),
        CORS_ORIGINS=['http://localhost:3000', 'http://localhost:5000'],
        SOCKETIO_ASYNC_MODE='threading',
        API_VERSION='v1'
    )
    
    # Update with provided config
    if config:
        app.config.update(config)
    
    # Ensure directories exist
    app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
    app.config['OUTPUT_FOLDER'].mkdir(parents=True, exist_ok=True)
    
    # Initialize CORS
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Initialize SocketIO
    socketio = SocketIO(
        app,
        cors_allowed_origins=app.config['CORS_ORIGINS'],
        async_mode=app.config['SOCKETIO_ASYNC_MODE']
    )
    
    # Initialize services
    auth = APIAuth(app.config['SECRET_KEY'])
    rate_limiter = RateLimiter(requests_per_minute=60)
    session_manager = SessionManager()
    batch_processor = BatchProcessor()
    
    # Store services in app context
    app.auth = auth
    app.rate_limiter = rate_limiter
    app.session_manager = session_manager
    app.batch_processor = batch_processor
    app.socketio = socketio
    
    # Initialize Swagger documentation
    init_swagger(app)
    
    # Middleware
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify(ErrorResponse(
            message="Bad request",
            error_code="BAD_REQUEST",
            error_details={'error': str(error)}
        ).to_dict()), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify(ErrorResponse(
            message="Unauthorized",
            error_code="UNAUTHORIZED"
        ).to_dict()), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify(ErrorResponse(
            message="Forbidden",
            error_code="FORBIDDEN"
        ).to_dict()), 403
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify(ErrorResponse(
            message="Not found",
            error_code="NOT_FOUND"
        ).to_dict()), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return jsonify(ErrorResponse(
            message="Internal server error",
            error_code="INTERNAL_ERROR"
        ).to_dict()), 500
    
    # Request hooks
    @app.before_request
    def before_request():
        """Handle authentication and rate limiting."""
        # Skip auth for health check and docs
        if request.path in ['/health', '/docs', '/']:
            return
        
        # Skip auth for OPTIONS requests (CORS preflight)
        if request.method == 'OPTIONS':
            return
        
        # Extract token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return unauthorized("Missing or invalid authorization header")
        
        token = auth_header.split(' ')[1]
        
        # Validate token
        session = auth.validate_token(token)
        if not session:
            return unauthorized("Invalid or expired token")
        
        # Check rate limit
        if not rate_limiter.check_rate_limit(session.user_id):
            return jsonify(ErrorResponse(
                message="Rate limit exceeded",
                error_code="RATE_LIMIT_EXCEEDED",
                error_details={
                    'remaining': rate_limiter.get_remaining_requests(session.user_id)
                }
            ).to_dict()), 429
        
        # Store session in request context
        request.user_session = session
    
    @app.after_request
    def after_request(response):
        """Add security headers."""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'version': app.config['API_VERSION'],
            'timestamp': datetime.utcnow().isoformat()
        })
    
    # Root endpoint
    @app.route('/')
    def root():
        """API information endpoint."""
        return jsonify({
            'name': 'PhotoSight API',
            'version': app.config['API_VERSION'],
            'endpoints': {
                'health': '/health',
                'auth': '/api/v1/auth',
                'sessions': '/api/v1/sessions',
                'images': '/api/v1/images',
                'process': '/api/v1/process',
                'batch': '/api/v1/batch',
                'websocket': '/socket.io'
            }
        })
    
    # Register API routes
    register_routes(app)
    
    # WebSocket event handlers
    @socketio.on('connect')
    def handle_connect():
        """Handle WebSocket connection."""
        logger.info(f"WebSocket client connected: {request.sid}")
        emit('connected', {'session_id': request.sid})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle WebSocket disconnection."""
        logger.info(f"WebSocket client disconnected: {request.sid}")
    
    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Subscribe to session updates."""
        session_id = data.get('session_id')
        if not session_id:
            emit('error', {'message': 'Missing session_id'})
            return
        
        # Join room for this session
        join_room(f"session_{session_id}")
        
        # Send current session state
        session = session_manager.get_session(session_id)
        if session:
            emit('session_state', session.to_dict())
        
        logger.info(f"Client {request.sid} subscribed to session {session_id}")
    
    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        """Unsubscribe from session updates."""
        session_id = data.get('session_id')
        if session_id:
            leave_room(f"session_{session_id}")
            logger.info(f"Client {request.sid} unsubscribed from session {session_id}")
    
    @socketio.on('heartbeat')
    def handle_heartbeat():
        """Handle heartbeat to keep connection alive."""
        emit('heartbeat', {'timestamp': datetime.utcnow().isoformat()})
    
    # Periodic tasks
    def cleanup_sessions():
        """Periodic cleanup of expired sessions."""
        with app.app_context():
            auth.cleanup_expired_sessions()
            session_manager.cleanup_inactive_sessions()
    
    # Schedule cleanup every hour
    if app.config.get('ENABLE_CLEANUP', True):
        from threading import Timer
        def run_cleanup():
            cleanup_sessions()
            Timer(3600, run_cleanup).start()  # Run every hour
        
        Timer(3600, run_cleanup).start()
    
    logger.info(f"PhotoSight API v{app.config['API_VERSION']} initialized")
    return app


def run_development_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = True):
    """Run the development server with SocketIO support."""
    app = create_app()
    app.socketio.run(app, host=host, port=port, debug=debug)