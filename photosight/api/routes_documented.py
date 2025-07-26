"""
Enhanced routes with Swagger documentation
"""

from flask import Blueprint, request, jsonify, send_file
from flasgger import swag_from
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

from .models import APIResponse, ErrorResponse, ProcessingRequest
from .auth import require_auth

api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')


@api_v1.route('/photos/upload', methods=['POST'])
@require_auth
@swag_from({
    'tags': ['photos'],
    'summary': 'Upload a RAW photo for processing',
    'description': 'Upload a RAW photo file and optionally start processing immediately',
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'in': 'formData',
            'name': 'file',
            'type': 'file',
            'required': True,
            'description': 'RAW photo file (CR2, NEF, ARW, DNG, etc.)'
        },
        {
            'in': 'formData',
            'name': 'auto_process',
            'type': 'boolean',
            'default': False,
            'description': 'Automatically start processing after upload'
        },
        {
            'in': 'formData',
            'name': 'project_id',
            'type': 'string',
            'description': 'Optional project ID to associate with the photo'
        }
    ],
    'responses': {
        '200': {
            'description': 'Photo uploaded successfully',
            'schema': {
                '$ref': '#/definitions/PhotoResponse'
            }
        },
        '400': {
            'description': 'Invalid file or parameters',
            'schema': {
                '$ref': '#/definitions/ErrorResponse'
            }
        },
        '401': {
            'description': 'Authentication required'
        },
        '413': {
            'description': 'File too large'
        }
    }
})
def upload_photo():
    """Upload a RAW photo file"""
    if 'file' not in request.files:
        return jsonify(ErrorResponse(
            message="No file provided",
            error_code="NO_FILE"
        ).to_dict()), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(ErrorResponse(
            message="No file selected",
            error_code="NO_FILE_SELECTED"
        ).to_dict()), 400
    
    # Validate file extension
    allowed_extensions = {'.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.raf'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify(ErrorResponse(
            message=f"Unsupported file type: {ext}",
            error_code="INVALID_FILE_TYPE"
        ).to_dict()), 400
    
    # Save file
    photo_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    filepath = current_app.config['UPLOAD_FOLDER'] / f"{photo_id}_{filename}"
    file.save(filepath)
    
    # Create photo record
    photo = {
        'id': photo_id,
        'filename': filename,
        'file_path': str(filepath),
        'status': 'uploaded',
        'created_at': datetime.utcnow().isoformat()
    }
    
    # Auto-process if requested
    if request.form.get('auto_process', 'false').lower() == 'true':
        # Queue for processing
        photo['status'] = 'queued'
    
    return jsonify(APIResponse(
        status='success',
        data=photo
    ).to_dict())


@api_v1.route('/photos/<photo_id>/process', methods=['POST'])
@require_auth
@swag_from({
    'tags': ['photos'],
    'summary': 'Process a photo with a recipe',
    'description': 'Apply processing settings to a RAW photo',
    'parameters': [
        {
            'in': 'path',
            'name': 'photo_id',
            'type': 'string',
            'required': True,
            'description': 'Photo ID'
        },
        {
            'in': 'body',
            'name': 'body',
            'required': True,
            'schema': {
                '$ref': '#/definitions/ProcessingRecipe'
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Processing started successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'data': {
                        'type': 'object',
                        'properties': {
                            'session_id': {'type': 'string'},
                            'preview_url': {'type': 'string'}
                        }
                    }
                }
            }
        },
        '404': {
            'description': 'Photo not found'
        },
        '400': {
            'description': 'Invalid processing recipe'
        }
    }
})
def process_photo(photo_id):
    """Process a photo with given settings"""
    try:
        recipe = ProcessingRequest.from_dict(request.json)
    except Exception as e:
        return jsonify(ErrorResponse(
            message=f"Invalid processing recipe: {str(e)}",
            error_code="INVALID_RECIPE"
        ).to_dict()), 400
    
    # Validate recipe
    is_valid, errors = RecipeValidator.validate(recipe.recipe)
    if not is_valid:
        return jsonify(ErrorResponse(
            message="Invalid processing parameters",
            error_code="VALIDATION_ERROR",
            error_details={'validation_errors': errors}
        ).to_dict()), 400
    
    # Start processing
    session_id = str(uuid.uuid4())
    
    return jsonify(APIResponse(
        status='success',
        data={
            'session_id': session_id,
            'preview_url': f'/api/v1/photos/{photo_id}/preview'
        }
    ).to_dict())


@api_v1.route('/sessions/<session_id>', methods=['GET'])
@require_auth
@swag_from({
    'tags': ['sessions'],
    'summary': 'Get session information',
    'description': 'Retrieve current state and settings for an editing session',
    'parameters': [
        {
            'in': 'path',
            'name': 'session_id',
            'type': 'string',
            'required': True,
            'description': 'Session ID'
        }
    ],
    'responses': {
        '200': {
            'description': 'Session information',
            'schema': {
                '$ref': '#/definitions/SessionResponse'
            }
        },
        '404': {
            'description': 'Session not found'
        }
    }
})
def get_session(session_id):
    """Get session information"""
    session = current_app.session_manager.get_session(session_id)
    if not session:
        return jsonify(ErrorResponse(
            message="Session not found",
            error_code="SESSION_NOT_FOUND"
        ).to_dict()), 404
    
    return jsonify(APIResponse(
        status='success',
        data=session.to_dict()
    ).to_dict())


@api_v1.route('/batch', methods=['POST'])
@require_auth
@swag_from({
    'tags': ['batch'],
    'summary': 'Create batch processing job',
    'description': 'Process multiple photos with the same settings',
    'parameters': [
        {
            'in': 'body',
            'name': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'required': ['photo_ids', 'recipe'],
                'properties': {
                    'photo_ids': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of photo IDs to process'
                    },
                    'recipe': {
                        '$ref': '#/definitions/ProcessingRecipe'
                    },
                    'output_format': {
                        'type': 'string',
                        'enum': ['jpeg', 'png', 'tiff', 'dng'],
                        'default': 'jpeg'
                    },
                    'priority': {
                        'type': 'string',
                        'enum': ['low', 'normal', 'high'],
                        'default': 'normal'
                    }
                }
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Batch job created',
            'schema': {
                '$ref': '#/definitions/BatchJobResponse'
            }
        },
        '400': {
            'description': 'Invalid request'
        }
    }
})
def create_batch_job():
    """Create a batch processing job"""
    try:
        batch_request = BatchProcessingRequest.from_dict(request.json)
    except Exception as e:
        return jsonify(ErrorResponse(
            message=f"Invalid batch request: {str(e)}",
            error_code="INVALID_REQUEST"
        ).to_dict()), 400
    
    # Create batch job
    job_id = current_app.batch_processor.create_job(
        photo_ids=batch_request.photo_ids,
        recipe=batch_request.recipe,
        output_format=batch_request.output_format,
        priority=batch_request.priority
    )
    
    return jsonify(APIResponse(
        status='success',
        data={
            'job_id': job_id,
            'status': 'queued',
            'total_items': len(batch_request.photo_ids)
        }
    ).to_dict())


@api_v1.route('/batch/<job_id>', methods=['GET'])
@require_auth
@swag_from({
    'tags': ['batch'],
    'summary': 'Get batch job status',
    'description': 'Retrieve status and progress of a batch processing job',
    'parameters': [
        {
            'in': 'path',
            'name': 'job_id',
            'type': 'string',
            'required': True,
            'description': 'Batch job ID'
        }
    ],
    'responses': {
        '200': {
            'description': 'Batch job status',
            'schema': {
                '$ref': '#/definitions/BatchJobResponse'
            }
        },
        '404': {
            'description': 'Job not found'
        }
    }
})
def get_batch_status(job_id):
    """Get batch job status"""
    job_status = current_app.batch_processor.get_job_status(job_id)
    if not job_status:
        return jsonify(ErrorResponse(
            message="Job not found",
            error_code="JOB_NOT_FOUND"
        ).to_dict()), 404
    
    return jsonify(APIResponse(
        status='success',
        data=job_status.to_dict()
    ).to_dict())


@api_v1.route('/system/health', methods=['GET'])
@swag_from({
    'tags': ['system'],
    'summary': 'Health check endpoint',
    'description': 'Check API server health and status',
    'responses': {
        '200': {
            'description': 'System is healthy',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string', 'example': 'healthy'},
                    'timestamp': {'type': 'string', 'format': 'date-time'},
                    'version': {'type': 'string'},
                    'uptime': {'type': 'number'}
                }
            }
        }
    }
})
def health_check():
    """System health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'uptime': 0  # Would calculate actual uptime
    })