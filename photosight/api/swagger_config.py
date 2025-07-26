"""
Swagger/OpenAPI configuration for PhotoSight API
"""

from flasgger import Swagger

# Swagger template configuration
template = {
    "swagger": "2.0",
    "info": {
        "title": "PhotoSight API",
        "description": "Professional RAW photo processing API with real-time preview updates",
        "contact": {
            "responsibleOrganization": "PhotoSight",
            "responsibleDeveloper": "PhotoSight Team",
            "email": "api@photosight.io",
            "url": "https://photosight.io",
        },
        "version": "1.0.0"
    },
    "basePath": "/api/v1",
    "schemes": [
        "https",
        "http"
    ],
    "securityDefinitions": {
        "Bearer": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "JWT Authorization header using the Bearer scheme. Example: \"Authorization: Bearer {token}\""
        }
    },
    "security": [
        {
            "Bearer": []
        }
    ],
    "tags": [
        {
            "name": "auth",
            "description": "Authentication endpoints"
        },
        {
            "name": "photos",
            "description": "Photo management and processing"
        },
        {
            "name": "sessions",
            "description": "Editing session management"
        },
        {
            "name": "batch",
            "description": "Batch processing operations"
        },
        {
            "name": "websocket",
            "description": "WebSocket endpoints for real-time updates"
        }
    ]
}

# Swagger configuration
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,  # Include all endpoints
            "model_filter": lambda tag: True,  # Include all models
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/api/docs"
}


def init_swagger(app):
    """Initialize Swagger documentation for the Flask app"""
    app.config['SWAGGER'] = {
        'title': 'PhotoSight API',
        'uiversion': 3,
        'openapi': '3.0.2'
    }
    
    swagger = Swagger(app, template=template, config=swagger_config)
    return swagger


# API endpoint documentation schemas
schemas = {
    "PhotoResponse": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "format": "uuid"},
            "filename": {"type": "string"},
            "file_path": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "processing", "completed", "failed"]},
            "metadata": {
                "type": "object",
                "properties": {
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "camera_make": {"type": "string"},
                    "camera_model": {"type": "string"},
                    "lens_model": {"type": "string"},
                    "iso": {"type": "integer"},
                    "aperture": {"type": "number"},
                    "shutter_speed": {"type": "string"},
                    "focal_length": {"type": "number"},
                    "date_taken": {"type": "string", "format": "date-time"}
                }
            },
            "processing_results": {
                "type": "object",
                "properties": {
                    "preview_url": {"type": "string"},
                    "histogram": {"type": "object"},
                    "quality_metrics": {"type": "object"}
                }
            },
            "created_at": {"type": "string", "format": "date-time"},
            "updated_at": {"type": "string", "format": "date-time"}
        }
    },
    "ProcessingRecipe": {
        "type": "object",
        "properties": {
            "exposure": {"type": "number", "minimum": -5, "maximum": 5},
            "contrast": {"type": "number", "minimum": -100, "maximum": 100},
            "highlights": {"type": "number", "minimum": -100, "maximum": 100},
            "shadows": {"type": "number", "minimum": -100, "maximum": 100},
            "whites": {"type": "number", "minimum": -100, "maximum": 100},
            "blacks": {"type": "number", "minimum": -100, "maximum": 100},
            "clarity": {"type": "number", "minimum": -100, "maximum": 100},
            "vibrance": {"type": "number", "minimum": -100, "maximum": 100},
            "saturation": {"type": "number", "minimum": -100, "maximum": 100},
            "white_balance": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "integer", "minimum": 2000, "maximum": 10000},
                    "tint": {"type": "integer", "minimum": -100, "maximum": 100}
                }
            },
            "tone_curve": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "lens_corrections": {
                "type": "object",
                "properties": {
                    "enable_profile": {"type": "boolean"},
                    "chromatic_aberration": {"type": "boolean"},
                    "vignetting": {"type": "boolean"},
                    "distortion": {"type": "boolean"}
                }
            },
            "noise_reduction": {
                "type": "object",
                "properties": {
                    "luminance": {"type": "number", "minimum": 0, "maximum": 100},
                    "color": {"type": "number", "minimum": 0, "maximum": 100},
                    "detail": {"type": "number", "minimum": 0, "maximum": 100}
                }
            },
            "sharpening": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "minimum": 0, "maximum": 150},
                    "radius": {"type": "number", "minimum": 0.5, "maximum": 3},
                    "detail": {"type": "number", "minimum": 0, "maximum": 100},
                    "masking": {"type": "number", "minimum": 0, "maximum": 100}
                }
            },
            "export_settings": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["jpeg", "png", "tiff", "dng"]},
                    "quality": {"type": "integer", "minimum": 1, "maximum": 100},
                    "color_space": {"type": "string", "enum": ["sRGB", "AdobeRGB", "ProPhotoRGB"]},
                    "resize": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "width": {"type": "integer"},
                            "height": {"type": "integer"},
                            "unit": {"type": "string", "enum": ["pixels", "percent"]}
                        }
                    }
                }
            }
        }
    },
    "SessionResponse": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "format": "uuid"},
            "photo_id": {"type": "string", "format": "uuid"},
            "created_at": {"type": "string", "format": "date-time"},
            "updated_at": {"type": "string", "format": "date-time"},
            "current_recipe": {"$ref": "#/definitions/ProcessingRecipe"},
            "preview_url": {"type": "string"},
            "undo_stack_size": {"type": "integer"},
            "redo_stack_size": {"type": "integer"}
        }
    },
    "BatchJobResponse": {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "format": "uuid"},
            "status": {"type": "string", "enum": ["pending", "processing", "completed", "failed", "cancelled"]},
            "total_items": {"type": "integer"},
            "processed_items": {"type": "integer"},
            "failed_items": {"type": "integer"},
            "progress": {"type": "number", "minimum": 0, "maximum": 100},
            "created_at": {"type": "string", "format": "date-time"},
            "started_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": "string", "format": "date-time"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "photo_id": {"type": "string", "format": "uuid"},
                        "status": {"type": "string"},
                        "output_path": {"type": "string"},
                        "error": {"type": "string"}
                    }
                }
            }
        }
    },
    "ErrorResponse": {
        "type": "object",
        "properties": {
            "error": {"type": "string"},
            "message": {"type": "string"},
            "status_code": {"type": "integer"}
        }
    }
}