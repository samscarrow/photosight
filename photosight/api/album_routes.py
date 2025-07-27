"""
Album API Routes for PhotoSight

Album management endpoints for the PhotoSight web viewer.
"""

import logging
from typing import List, Dict, Optional, Any
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text

from ..db.models import AlbumTag, PhotoAlbumTag, Photo
from ..db.album_utils import AlbumManager
from .models import APIResponse, ErrorResponse

logger = logging.getLogger(__name__)

# Create album blueprint
album_api = Blueprint('album_api', __name__, url_prefix='/api/v1/albums')


class AlbumAPIHandler:
    """Handles album-related API operations with Oracle database."""
    
    def __init__(self):
        self.album_manager = AlbumManager()
    
    def get_albums_list(self) -> Dict[str, Any]:
        """Get list of all albums with photo counts."""
        try:
            # Import necessary function to make MCP call
            import subprocess
            import json
            
            # For now, create a simple method to execute Oracle queries
            # This would be replaced with proper MCP integration
            query = """
                SELECT 
                    at.id,
                    at.tag_name,
                    at.display_name,
                    at.description,
                    at.photo_count,
                    at.created_at,
                    at.updated_at,
                    at.album_metadata
                FROM PHOTOSIGHT.ALBUM_TAGS at
                ORDER BY at.created_at DESC
            """
            
            # Placeholder result structure - in real implementation this would call MCP
            result = {
                'rows': [
                    {
                        'ID': 1,
                        'TAG_NAME': 'enneagram-top-30',
                        'DISPLAY_NAME': 'Enneagram Workshop - Top 30',
                        'DESCRIPTION': 'Top 30 highest-scoring photos from the enneagram workshop',
                        'PHOTO_COUNT': 30,
                        'CREATED_AT': '2025-01-27T00:00:00Z',
                        'UPDATED_AT': '2025-01-27T00:00:00Z',
                        'ALBUM_METADATA': {}
                    },
                    {
                        'ID': 2,
                        'TAG_NAME': 'enneagram-decisive-moments',
                        'DISPLAY_NAME': 'Enneagram Workshop - Decisive Moments',
                        'DESCRIPTION': 'Photos that captured decisive moments and genuine interactions',
                        'PHOTO_COUNT': 22,
                        'CREATED_AT': '2025-01-27T00:00:00Z',
                        'UPDATED_AT': '2025-01-27T00:00:00Z',
                        'ALBUM_METADATA': {}
                    }
                ]
            }
            
            albums = []
            for row in result['rows']:
                album = {
                    'id': row['ID'],
                    'tag_name': row['TAG_NAME'],
                    'display_name': row['DISPLAY_NAME'],
                    'description': row['DESCRIPTION'],
                    'photo_count': row['PHOTO_COUNT'],
                    'created_at': row['CREATED_AT'],
                    'updated_at': row['UPDATED_AT'],
                    'metadata': row['ALBUM_METADATA'] or {}
                }
                albums.append(album)
            
            return {
                'albums': albums,
                'total_count': len(albums)
            }
            
        except Exception as e:
            logger.error(f"Error fetching albums: {e}")
            raise
    
    def get_album_photos(self, album_tag: str, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """Get photos in a specific album with pagination."""
        try:
            offset = (page - 1) * per_page
            
            # Use MCP Oracle connection
            from ...mcp import oracle
            
            # Get photos with metadata
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
                    p.overall_score,
                    p.technical_score,
                    p.artistic_score,
                    p.emotional_score,
                    pat.sort_order,
                    pat.association_metadata
                FROM PHOTOSIGHT.PHOTOS p
                JOIN PHOTOSIGHT.PHOTO_ALBUM_TAGS pat ON p.id = pat.photo_id
                JOIN PHOTOSIGHT.ALBUM_TAGS at ON pat.album_tag_id = at.id
                WHERE at.tag_name = '{album_tag}'
                ORDER BY pat.sort_order ASC, p.captured_at DESC
                OFFSET {offset} ROWS FETCH NEXT {per_page} ROWS ONLY
            """)
            
            # Get total count
            count_result = oracle.execute_query(f"""
                SELECT COUNT(*) as total
                FROM PHOTOSIGHT.PHOTO_ALBUM_TAGS pat
                JOIN PHOTOSIGHT.ALBUM_TAGS at ON pat.album_tag_id = at.id
                WHERE at.tag_name = '{album_tag}'
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
                    'lens_model': row['LENS_MODEL'],
                    'aperture': row['APERTURE'],
                    'shutter_speed': row['SHUTTER_SPEED'],
                    'iso': row['ISO'],
                    'focal_length': row['FOCAL_LENGTH'],
                    'scores': {
                        'overall': row['OVERALL_SCORE'],
                        'technical': row['TECHNICAL_SCORE'],
                        'artistic': row['ARTISTIC_SCORE'],
                        'emotional': row['EMOTIONAL_SCORE']
                    },
                    'sort_order': row['SORT_ORDER'],
                    'album_metadata': row['ASSOCIATION_METADATA'] or {},
                    'thumbnail_url': f'/api/v1/photos/{row["ID"]}/thumbnail',
                    'preview_url': f'/api/v1/photos/{row["ID"]}/preview'
                }
                photos.append(photo)
            
            return {
                'photos': photos,
                'total_count': total_count,
                'page': page,
                'per_page': per_page,
                'total_pages': (total_count + per_page - 1) // per_page
            }
            
        except Exception as e:
            logger.error(f"Error fetching album photos: {e}")
            raise
    
    def get_album_details(self, album_tag: str) -> Dict[str, Any]:
        """Get detailed information about a specific album."""
        try:
            from ...mcp import oracle
            
            result = oracle.execute_query(f"""
                SELECT 
                    at.id,
                    at.tag_name,
                    at.display_name,
                    at.description,
                    at.photo_count,
                    at.created_at,
                    at.updated_at,
                    at.last_photo_added,
                    at.album_metadata
                FROM PHOTOSIGHT.ALBUM_TAGS at
                WHERE at.tag_name = '{album_tag}'
            """)
            
            if not result['rows']:
                return None
            
            row = result['rows'][0]
            return {
                'id': row['ID'],
                'tag_name': row['TAG_NAME'],
                'display_name': row['DISPLAY_NAME'],
                'description': row['DESCRIPTION'],
                'photo_count': row['PHOTO_COUNT'],
                'created_at': row['CREATED_AT'],
                'updated_at': row['UPDATED_AT'],
                'last_photo_added': row['LAST_PHOTO_ADDED'],
                'metadata': row['ALBUM_METADATA'] or {}
            }
            
        except Exception as e:
            logger.error(f"Error fetching album details: {e}")
            raise


# Initialize handler
album_handler = AlbumAPIHandler()


@album_api.route('/', methods=['GET'])
def list_albums():
    """
    List all albums with photo counts
    ---
    get:
      summary: Get all albums
      description: Retrieve a list of all albums with their metadata and photo counts
      responses:
        200:
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    type: object
                    properties:
                      albums:
                        type: array
                        items:
                          type: object
                      total_count:
                        type: integer
    """
    try:
        data = album_handler.get_albums_list()
        
        return jsonify(APIResponse(
            message="Albums retrieved successfully",
            data=data
        ).to_dict())
        
    except Exception as e:
        logger.error(f"Error in list_albums: {e}")
        return jsonify(ErrorResponse(
            message="Failed to retrieve albums",
            error_code="ALBUM_FETCH_ERROR",
            error_details={"error": str(e)}
        ).to_dict()), 500


@album_api.route('/<album_tag>', methods=['GET'])
def get_album(album_tag: str):
    """
    Get album details
    ---
    get:
      summary: Get album details
      description: Retrieve detailed information about a specific album
      parameters:
        - in: path
          name: album_tag
          required: true
          schema:
            type: string
          description: Album tag name
      responses:
        200:
          description: Successful response
        404:
          description: Album not found
    """
    try:
        album_details = album_handler.get_album_details(album_tag)
        
        if not album_details:
            return jsonify(ErrorResponse(
                message="Album not found",
                error_code="ALBUM_NOT_FOUND"
            ).to_dict()), 404
        
        return jsonify(APIResponse(
            message="Album details retrieved successfully",
            data=album_details
        ).to_dict())
        
    except Exception as e:
        logger.error(f"Error in get_album: {e}")
        return jsonify(ErrorResponse(
            message="Failed to retrieve album details",
            error_code="ALBUM_FETCH_ERROR",
            error_details={"error": str(e)}
        ).to_dict()), 500


@album_api.route('/<album_tag>/photos', methods=['GET'])
def get_album_photos(album_tag: str):
    """
    Get photos in an album
    ---
    get:
      summary: Get album photos
      description: Retrieve photos in a specific album with pagination
      parameters:
        - in: path
          name: album_tag
          required: true
          schema:
            type: string
          description: Album tag name
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
        404:
          description: Album not found
    """
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)  # Max 100 per page
        
        # First check if album exists
        album_details = album_handler.get_album_details(album_tag)
        if not album_details:
            return jsonify(ErrorResponse(
                message="Album not found",
                error_code="ALBUM_NOT_FOUND"
            ).to_dict()), 404
        
        # Get photos
        data = album_handler.get_album_photos(album_tag, page, per_page)
        
        return jsonify(APIResponse(
            message="Album photos retrieved successfully",
            data=data
        ).to_dict())
        
    except ValueError as e:
        return jsonify(ErrorResponse(
            message="Invalid pagination parameters",
            error_code="INVALID_PARAMS",
            error_details={"error": str(e)}
        ).to_dict()), 400
    except Exception as e:
        logger.error(f"Error in get_album_photos: {e}")
        return jsonify(ErrorResponse(
            message="Failed to retrieve album photos",
            error_code="ALBUM_PHOTOS_FETCH_ERROR",
            error_details={"error": str(e)}
        ).to_dict()), 500


@album_api.route('/stats', methods=['GET'])
def get_album_stats():
    """
    Get album statistics
    ---
    get:
      summary: Get album statistics
      description: Retrieve statistics about all albums
      responses:
        200:
          description: Successful response
    """
    try:
        from ...mcp import oracle
        
        result = oracle.execute_query("""
            SELECT 
                COUNT(*) as total_albums,
                SUM(photo_count) as total_photos_in_albums,
                AVG(photo_count) as avg_photos_per_album,
                MAX(photo_count) as max_photos_in_album,
                MIN(photo_count) as min_photos_in_album
            FROM PHOTOSIGHT.ALBUM_TAGS
        """)
        
        stats = {}
        if result['rows']:
            row = result['rows'][0]
            stats = {
                'total_albums': row['TOTAL_ALBUMS'],
                'total_photos_in_albums': row['TOTAL_PHOTOS_IN_ALBUMS'],
                'avg_photos_per_album': float(row['AVG_PHOTOS_PER_ALBUM']) if row['AVG_PHOTOS_PER_ALBUM'] else 0,
                'max_photos_in_album': row['MAX_PHOTOS_IN_ALBUM'],
                'min_photos_in_album': row['MIN_PHOTOS_IN_ALBUM']
            }
        
        return jsonify(APIResponse(
            message="Album statistics retrieved successfully",
            data=stats
        ).to_dict())
        
    except Exception as e:
        logger.error(f"Error in get_album_stats: {e}")
        return jsonify(ErrorResponse(
            message="Failed to retrieve album statistics",
            error_code="ALBUM_STATS_ERROR",
            error_details={"error": str(e)}
        ).to_dict()), 500


def register_album_routes(app):
    """Register album routes with the Flask app."""
    app.register_blueprint(album_api)
    logger.info("Album API routes registered")