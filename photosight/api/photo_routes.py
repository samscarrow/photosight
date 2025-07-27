"""
Photo API Routes for PhotoSight

Photo viewing and serving endpoints for the PhotoSight web viewer.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from flask import Blueprint, request, jsonify, send_file, current_app
from PIL import Image
import io
import base64

from .models import APIResponse, ErrorResponse

logger = logging.getLogger(__name__)

# Create photo blueprint
photo_api = Blueprint('photo_api', __name__, url_prefix='/api/v1/photos')


class PhotoAPIHandler:
    """Handles photo-related API operations."""
    
    def __init__(self):
        self.thumbnail_cache = {}  # Simple in-memory cache
    
    def get_photo_details(self, photo_id: int) -> Optional[dict]:
        """Get detailed information about a specific photo."""
        try:
            from ...mcp import oracle
            
            result = oracle.execute_query(f"""
                SELECT 
                    p.id,
                    p.filename,
                    p.file_path,
                    p.captured_at,
                    p.camera_make,
                    p.camera_model,
                    p.lens_model,
                    p.aperture,
                    p.shutter_speed,
                    p.iso,
                    p.focal_length,
                    p.width,
                    p.height,
                    p.file_size,
                    p.overall_score,
                    p.technical_score,
                    p.artistic_score,
                    p.emotional_score,
                    p.created_at,
                    p.updated_at
                FROM PHOTOSIGHT.PHOTOS p
                WHERE p.id = {photo_id}
            """)
            
            if not result['rows']:
                return None
            
            row = result['rows'][0]
            return {
                'id': row['ID'],
                'filename': row['FILENAME'],
                'file_path': row['FILE_PATH'],
                'captured_at': row['CAPTURED_AT'],
                'camera_make': row['CAMERA_MAKE'],
                'camera_model': row['CAMERA_MODEL'],
                'lens_model': row['LENS_MODEL'],
                'aperture': row['APERTURE'],
                'shutter_speed': row['SHUTTER_SPEED'],
                'iso': row['ISO'],
                'focal_length': row['FOCAL_LENGTH'],
                'dimensions': {
                    'width': row['WIDTH'],
                    'height': row['HEIGHT']
                },
                'file_size': row['FILE_SIZE'],
                'scores': {
                    'overall': row['OVERALL_SCORE'],
                    'technical': row['TECHNICAL_SCORE'],
                    'artistic': row['ARTISTIC_SCORE'],
                    'emotional': row['EMOTIONAL_SCORE']
                },
                'created_at': row['CREATED_AT'],
                'updated_at': row['UPDATED_AT']
            }
            
        except Exception as e:
            logger.error(f"Error fetching photo details: {e}")
            raise
    
    def get_photo_albums(self, photo_id: int) -> list:
        """Get albums that contain this photo."""
        try:
            from ...mcp import oracle
            
            result = oracle.execute_query(f"""
                SELECT 
                    at.tag_name,
                    at.display_name,
                    pat.sort_order,
                    pat.association_metadata
                FROM PHOTOSIGHT.ALBUM_TAGS at
                JOIN PHOTOSIGHT.PHOTO_ALBUM_TAGS pat ON at.id = pat.album_tag_id
                WHERE pat.photo_id = {photo_id}
                ORDER BY at.display_name
            """)
            
            albums = []
            for row in result['rows']:
                albums.append({
                    'tag_name': row['TAG_NAME'],
                    'display_name': row['DISPLAY_NAME'],
                    'sort_order': row['SORT_ORDER'],
                    'metadata': row['ASSOCIATION_METADATA'] or {}
                })
            
            return albums
            
        except Exception as e:
            logger.error(f"Error fetching photo albums: {e}")
            raise
    
    def generate_thumbnail(self, file_path: str, size: tuple = (300, 300)) -> bytes:
        """Generate a thumbnail for the given image file."""
        cache_key = f"{file_path}_{size[0]}x{size[1]}"
        
        # Check cache first
        if cache_key in self.thumbnail_cache:
            return self.thumbnail_cache[cache_key]
        
        try:
            # Open and resize image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary (for JPEG output)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Calculate size maintaining aspect ratio
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Save to bytes
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=85, optimize=True)
                thumbnail_bytes = output.getvalue()
                
                # Cache the result (simple cache, could be improved with TTL)
                if len(self.thumbnail_cache) < 1000:  # Limit cache size
                    self.thumbnail_cache[cache_key] = thumbnail_bytes
                
                return thumbnail_bytes
                
        except Exception as e:
            logger.error(f"Error generating thumbnail for {file_path}: {e}")
            raise
    
    def generate_preview(self, file_path: str, size: tuple = (1200, 800)) -> bytes:
        """Generate a preview for the given image file."""
        cache_key = f"{file_path}_preview_{size[0]}x{size[1]}"
        
        # Check cache first
        if cache_key in self.thumbnail_cache:
            return self.thumbnail_cache[cache_key]
        
        try:
            with Image.open(file_path) as img:
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Calculate size maintaining aspect ratio
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Save to bytes
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=90, optimize=True)
                preview_bytes = output.getvalue()
                
                # Cache the result
                if len(self.thumbnail_cache) < 500:  # Smaller cache for previews
                    self.thumbnail_cache[cache_key] = preview_bytes
                
                return preview_bytes
                
        except Exception as e:
            logger.error(f"Error generating preview for {file_path}: {e}")
            raise


# Initialize handler
photo_handler = PhotoAPIHandler()


@photo_api.route('/<int:photo_id>', methods=['GET'])
def get_photo_details(photo_id: int):
    """
    Get detailed photo information
    ---
    get:
      summary: Get photo details
      description: Retrieve detailed information about a specific photo
      parameters:
        - in: path
          name: photo_id
          required: true
          schema:
            type: integer
          description: Photo ID
      responses:
        200:
          description: Successful response
        404:
          description: Photo not found
    """
    try:
        photo_details = photo_handler.get_photo_details(photo_id)
        
        if not photo_details:
            return jsonify(ErrorResponse(
                message="Photo not found",
                error_code="PHOTO_NOT_FOUND"
            ).to_dict()), 404
        
        # Get albums containing this photo
        albums = photo_handler.get_photo_albums(photo_id)
        photo_details['albums'] = albums
        
        return jsonify(APIResponse(
            message="Photo details retrieved successfully",
            data=photo_details
        ).to_dict())
        
    except Exception as e:
        logger.error(f"Error in get_photo_details: {e}")
        return jsonify(ErrorResponse(
            message="Failed to retrieve photo details",
            error_code="PHOTO_FETCH_ERROR",
            error_details={"error": str(e)}
        ).to_dict()), 500


@photo_api.route('/<int:photo_id>/thumbnail', methods=['GET'])
def get_photo_thumbnail(photo_id: int):
    """
    Get photo thumbnail
    ---
    get:
      summary: Get photo thumbnail
      description: Retrieve a thumbnail image for the specified photo
      parameters:
        - in: path
          name: photo_id
          required: true
          schema:
            type: integer
          description: Photo ID
        - in: query
          name: size
          schema:
            type: integer
            default: 300
          description: Thumbnail size (width and height)
      responses:
        200:
          description: Thumbnail image
          content:
            image/jpeg:
              schema:
                type: string
                format: binary
        404:
          description: Photo not found
    """
    try:
        # Get photo details to find file path
        photo_details = photo_handler.get_photo_details(photo_id)
        
        if not photo_details:
            return jsonify(ErrorResponse(
                message="Photo not found",
                error_code="PHOTO_NOT_FOUND"
            ).to_dict()), 404
        
        file_path = photo_details['file_path']
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify(ErrorResponse(
                message="Photo file not found on disk",
                error_code="PHOTO_FILE_NOT_FOUND"
            ).to_dict()), 404
        
        # Get size parameter
        size = int(request.args.get('size', 300))
        size = min(max(size, 50), 1000)  # Limit between 50 and 1000 pixels
        
        # Generate thumbnail
        thumbnail_bytes = photo_handler.generate_thumbnail(file_path, (size, size))
        
        # Return thumbnail as response
        return send_file(
            io.BytesIO(thumbnail_bytes),
            mimetype='image/jpeg',
            as_attachment=False,
            download_name=f"thumbnail_{photo_id}.jpg"
        )
        
    except Exception as e:
        logger.error(f"Error in get_photo_thumbnail: {e}")
        return jsonify(ErrorResponse(
            message="Failed to generate thumbnail",
            error_code="THUMBNAIL_ERROR",
            error_details={"error": str(e)}
        ).to_dict()), 500


@photo_api.route('/<int:photo_id>/preview', methods=['GET'])
def get_photo_preview(photo_id: int):
    """
    Get photo preview
    ---
    get:
      summary: Get photo preview
      description: Retrieve a preview image for the specified photo
      parameters:
        - in: path
          name: photo_id
          required: true
          schema:
            type: integer
          description: Photo ID
      responses:
        200:
          description: Preview image
          content:
            image/jpeg:
              schema:
                type: string
                format: binary
        404:
          description: Photo not found
    """
    try:
        # Get photo details to find file path
        photo_details = photo_handler.get_photo_details(photo_id)
        
        if not photo_details:
            return jsonify(ErrorResponse(
                message="Photo not found",
                error_code="PHOTO_NOT_FOUND"
            ).to_dict()), 404
        
        file_path = photo_details['file_path']
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify(ErrorResponse(
                message="Photo file not found on disk",
                error_code="PHOTO_FILE_NOT_FOUND"
            ).to_dict()), 404
        
        # Generate preview
        preview_bytes = photo_handler.generate_preview(file_path, (1200, 800))
        
        # Return preview as response
        return send_file(
            io.BytesIO(preview_bytes),
            mimetype='image/jpeg',
            as_attachment=False,
            download_name=f"preview_{photo_id}.jpg"
        )
        
    except Exception as e:
        logger.error(f"Error in get_photo_preview: {e}")
        return jsonify(ErrorResponse(
            message="Failed to generate preview",
            error_code="PREVIEW_ERROR",
            error_details={"error": str(e)}
        ).to_dict()), 500


@photo_api.route('/search', methods=['GET'])
def search_photos():
    """
    Search photos
    ---
    get:
      summary: Search photos
      description: Search photos by various criteria
      parameters:
        - in: query
          name: q
          schema:
            type: string
          description: Search query
        - in: query
          name: camera_make
          schema:
            type: string
          description: Filter by camera make
        - in: query
          name: min_score
          schema:
            type: number
          description: Minimum overall score
        - in: query
          name: page
          schema:
            type: integer
            default: 1
          description: Page number
        - in: query
          name: per_page
          schema:
            type: integer
            default: 20
          description: Number of photos per page
      responses:
        200:
          description: Successful response
    """
    try:
        # Get query parameters
        query = request.args.get('q', '').strip()
        camera_make = request.args.get('camera_make', '').strip()
        min_score = request.args.get('min_score', type=float)
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        
        # Build WHERE clause
        where_clauses = []
        if query:
            where_clauses.append(f"(UPPER(p.filename) LIKE UPPER('%{query}%') OR UPPER(p.camera_model) LIKE UPPER('%{query}%'))")
        if camera_make:
            where_clauses.append(f"UPPER(p.camera_make) LIKE UPPER('%{camera_make}%')")
        if min_score is not None:
            where_clauses.append(f"p.overall_score >= {min_score}")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        offset = (page - 1) * per_page
        
        from ...mcp import oracle
        
        # Get photos
        result = oracle.execute_query(f"""
            SELECT 
                p.id,
                p.filename,
                p.file_path,
                p.captured_at,
                p.camera_make,
                p.camera_model,
                p.overall_score,
                p.technical_score,
                p.artistic_score,
                p.emotional_score
            FROM PHOTOSIGHT.PHOTOS p
            WHERE {where_clause}
            ORDER BY p.overall_score DESC, p.captured_at DESC
            OFFSET {offset} ROWS FETCH NEXT {per_page} ROWS ONLY
        """)
        
        # Get total count
        count_result = oracle.execute_query(f"""
            SELECT COUNT(*) as total
            FROM PHOTOSIGHT.PHOTOS p
            WHERE {where_clause}
        """)
        
        total_count = count_result['rows'][0]['TOTAL'] if count_result['rows'] else 0
        
        photos = []
        for row in result['rows']:
            photo = {
                'id': row['ID'],
                'filename': row['FILENAME'],
                'file_path': row['FILE_PATH'],
                'captured_at': row['CAPTURED_AT'],
                'camera_make': row['CAMERA_MAKE'],
                'camera_model': row['CAMERA_MODEL'],
                'scores': {
                    'overall': row['OVERALL_SCORE'],
                    'technical': row['TECHNICAL_SCORE'],
                    'artistic': row['ARTISTIC_SCORE'],
                    'emotional': row['EMOTIONAL_SCORE']
                },
                'thumbnail_url': f'/api/v1/photos/{row["ID"]}/thumbnail',
                'preview_url': f'/api/v1/photos/{row["ID"]}/preview'
            }
            photos.append(photo)
        
        return jsonify(APIResponse(
            message="Photo search completed successfully",
            data={
                'photos': photos,
                'total_count': total_count,
                'page': page,
                'per_page': per_page,
                'total_pages': (total_count + per_page - 1) // per_page,
                'search_params': {
                    'query': query,
                    'camera_make': camera_make,
                    'min_score': min_score
                }
            }
        ).to_dict())
        
    except ValueError as e:
        return jsonify(ErrorResponse(
            message="Invalid search parameters",
            error_code="INVALID_PARAMS",
            error_details={"error": str(e)}
        ).to_dict()), 400
    except Exception as e:
        logger.error(f"Error in search_photos: {e}")
        return jsonify(ErrorResponse(
            message="Failed to search photos",
            error_code="SEARCH_ERROR",
            error_details={"error": str(e)}
        ).to_dict()), 500


def register_photo_routes(app):
    """Register photo routes with the Flask app."""
    app.register_blueprint(photo_api)
    logger.info("Photo API routes registered")