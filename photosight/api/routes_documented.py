"""
Enhanced routes with Swagger documentation
"""

from flask import Blueprint, request, jsonify, send_file, current_app
from flasgger import swag_from
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid
from datetime import datetime

from .models import APIResponse, ErrorResponse, ProcessingRequest
from .auth import require_auth
from ..db.mcp_operations import MCPMetadataOperations
from ..utils.xmp_sidecar import sync_xmp_with_database

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


@api_v1.route('/photos/<int:photo_id>/metadata', methods=['GET'])
@require_auth
@swag_from({
    'tags': ['metadata'],
    'summary': 'Get photo metadata',
    'description': 'Retrieve metadata for a photo including keywords, collections, and IPTC data',
    'parameters': [
        {
            'in': 'path',
            'name': 'photo_id',
            'type': 'integer',
            'required': True,
            'description': 'Photo ID'
        }
    ],
    'responses': {
        '200': {
            'description': 'Metadata retrieved successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'data': {
                        'type': 'object',
                        'properties': {
                            'photo': {
                                'type': 'object',
                                'description': 'Basic photo information'
                            },
                            'keywords': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'properties': {
                                        'id': {'type': 'integer'},
                                        'keyword': {'type': 'string'},
                                        'category': {'type': 'string'},
                                        'source': {'type': 'string'}
                                    }
                                }
                            },
                            'collections': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'properties': {
                                        'id': {'type': 'integer'},
                                        'name': {'type': 'string'},
                                        'description': {'type': 'string'},
                                        'collection_type': {'type': 'string'}
                                    }
                                }
                            },
                            'iptc': {
                                'type': 'object',
                                'description': 'IPTC metadata fields'
                            }
                        }
                    }
                }
            }
        },
        '404': {
            'description': 'Photo not found'
        },
        '401': {
            'description': 'Authentication required'
        }
    }
})
def get_photo_metadata(photo_id):
    """Get metadata for a specific photo"""
    metadata = MCPMetadataOperations.get_photo_metadata(photo_id)
    
    if 'error' in metadata:
        return jsonify(ErrorResponse(
            message=metadata['error'],
            error_code="PHOTO_NOT_FOUND"
        ).to_dict()), 404
    
    return jsonify(APIResponse(
        status='success',
        data=metadata
    ).to_dict())


@api_v1.route('/photos/<int:photo_id>/metadata', methods=['PUT'])
@require_auth
@swag_from({
    'tags': ['metadata'],
    'summary': 'Update photo metadata',
    'description': 'Update keywords, collections, and IPTC metadata for a photo',
    'parameters': [
        {
            'in': 'path',
            'name': 'photo_id',
            'type': 'integer',
            'required': True,
            'description': 'Photo ID'
        },
        {
            'in': 'body',
            'name': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'keywords': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of keywords to set (replaces existing)'
                    },
                    'collections': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of collection names to add photo to'
                    },
                    'iptc': {
                        'type': 'object',
                        'properties': {
                            'title': {'type': 'string'},
                            'caption': {'type': 'string'},
                            'headline': {'type': 'string'},
                            'creator': {'type': 'string'},
                            'copyright_notice': {'type': 'string'},
                            'city': {'type': 'string'},
                            'region': {'type': 'string'},
                            'country': {'type': 'string'},
                            'event': {'type': 'string'}
                        }
                    },
                    'sync_xmp': {
                        'type': 'boolean',
                        'default': True,
                        'description': 'Whether to sync metadata to XMP sidecar file'
                    }
                }
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Metadata updated successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'message': {'type': 'string'},
                    'data': {
                        'type': 'object',
                        'properties': {
                            'keywords_updated': {'type': 'boolean'},
                            'iptc_updated': {'type': 'boolean'},
                            'collections_updated': {'type': 'boolean'},
                            'xmp_synced': {'type': 'boolean'}
                        }
                    }
                }
            }
        },
        '404': {
            'description': 'Photo not found'
        },
        '400': {
            'description': 'Invalid metadata format'
        },
        '401': {
            'description': 'Authentication required'
        }
    }
})
def update_photo_metadata(photo_id):
    """Update metadata for a specific photo"""
    data = request.get_json()
    
    if not data:
        return jsonify(ErrorResponse(
            message="No metadata provided",
            error_code="NO_DATA"
        ).to_dict()), 400
    
    # Check if photo exists
    existing = MCPMetadataOperations.get_photo_metadata(photo_id)
    if 'error' in existing:
        return jsonify(ErrorResponse(
            message="Photo not found",
            error_code="PHOTO_NOT_FOUND"
        ).to_dict()), 404
    
    results = {
        'keywords_updated': False,
        'iptc_updated': False,
        'collections_updated': False,
        'xmp_synced': False
    }
    
    # Update keywords
    if 'keywords' in data:
        if not isinstance(data['keywords'], list):
            return jsonify(ErrorResponse(
                message="Keywords must be a list",
                error_code="INVALID_FORMAT"
            ).to_dict()), 400
        
        results['keywords_updated'] = MCPMetadataOperations.update_keywords(
            photo_id, data['keywords']
        )
    
    # Update IPTC metadata
    if 'iptc' in data:
        if not isinstance(data['iptc'], dict):
            return jsonify(ErrorResponse(
                message="IPTC data must be an object",
                error_code="INVALID_FORMAT"
            ).to_dict()), 400
        
        results['iptc_updated'] = MCPMetadataOperations.update_iptc_metadata(
            photo_id, data['iptc']
        )
    
    # Add to collections
    if 'collections' in data:
        if not isinstance(data['collections'], list):
            return jsonify(ErrorResponse(
                message="Collections must be a list",
                error_code="INVALID_FORMAT"
            ).to_dict()), 400
        
        for collection_name in data['collections']:
            if MCPMetadataOperations.add_to_collection(photo_id, collection_name):
                results['collections_updated'] = True
    
    # Sync to XMP if requested
    if data.get('sync_xmp', True) and existing.get('photo'):
        file_path = existing['photo'].get('FILE_PATH')
        if file_path:
            # Get updated metadata
            updated_metadata = MCPMetadataOperations.get_photo_metadata(photo_id)
            results['xmp_synced'] = sync_xmp_with_database(file_path, updated_metadata)
    
    return jsonify(APIResponse(
        status='success',
        message='Metadata updated successfully',
        data=results
    ).to_dict())


@api_v1.route('/photos/search/metadata', methods=['POST'])
@require_auth
@swag_from({
    'tags': ['metadata'],
    'summary': 'Search photos by metadata',
    'description': 'Search for photos using keywords, collections, or IPTC fields',
    'parameters': [
        {
            'in': 'body',
            'name': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'keywords': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Keywords to search for (OR logic)'
                    },
                    'collection': {
                        'type': 'string',
                        'description': 'Collection name'
                    },
                    'creator': {
                        'type': 'string',
                        'description': 'Creator/photographer name (partial match)'
                    },
                    'city': {
                        'type': 'string',
                        'description': 'City (partial match)'
                    },
                    'event': {
                        'type': 'string',
                        'description': 'Event name (partial match)'
                    },
                    'caption': {
                        'type': 'string',
                        'description': 'Caption text (partial match)'
                    }
                }
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Search results',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'data': {
                        'type': 'object',
                        'properties': {
                            'count': {'type': 'integer'},
                            'photos': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'properties': {
                                        'id': {'type': 'integer'},
                                        'file_path': {'type': 'string'},
                                        'filename': {'type': 'string'},
                                        'date_taken': {'type': 'string'},
                                        'camera_model': {'type': 'string'},
                                        'lens_model': {'type': 'string'},
                                        'processing_status': {'type': 'string'}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        '400': {
            'description': 'Invalid search parameters'
        },
        '401': {
            'description': 'Authentication required'
        }
    }
})
def search_photos_by_metadata():
    """Search photos using metadata criteria"""
    search_params = request.get_json()
    
    if not search_params:
        return jsonify(ErrorResponse(
            message="No search criteria provided",
            error_code="NO_CRITERIA"
        ).to_dict()), 400
    
    # Perform search
    results = MCPMetadataOperations.search_by_metadata(search_params)
    
    return jsonify(APIResponse(
        status='success',
        data={
            'count': len(results),
            'photos': results
        }
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