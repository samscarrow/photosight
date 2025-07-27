"""
Web Interface Routes for PhotoSight

HTML-based web viewer for albums and photos.
"""

import logging
from flask import Blueprint, render_template, request, jsonify
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Create web blueprint
web_viewer = Blueprint('web_viewer', __name__, url_prefix='/viewer')


class WebDataProvider:
    """Provides data for the web interface using MCP Oracle connection."""
    
    def get_albums_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all albums for web display."""
        try:
            # In a real implementation, this would use the MCP Oracle connection
            # For now, return mock data that matches our actual database
            return [
                {
                    'id': 1,
                    'tag_name': 'enneagram-top-30',
                    'display_name': 'Enneagram Workshop - Top 30',
                    'description': 'Top 30 highest-scoring photos from the enneagram workshop',
                    'photo_count': 30,
                    'thumbnail_url': '/api/v1/photos/2080/thumbnail',
                    'created_at': '2025-01-27'
                },
                {
                    'id': 2,
                    'tag_name': 'enneagram-decisive-moments',
                    'display_name': 'Enneagram Workshop - Decisive Moments',
                    'description': 'Photos that captured decisive moments and genuine interactions',
                    'photo_count': 22,
                    'thumbnail_url': '/api/v1/photos/2040/thumbnail',
                    'created_at': '2025-01-27'
                }
            ]
        except Exception as e:
            logger.error(f"Error getting albums summary: {e}")
            return []
    
    def get_album_photos_preview(self, album_tag: str, limit: int = 12) -> List[Dict[str, Any]]:
        """Get preview photos for an album."""
        try:
            # Mock data for the top-scoring photos in each album
            if album_tag == 'enneagram-top-30':
                return [
                    {'id': 2080, 'filename': 'DSC_2080.jpg', 'score': 0.872},
                    {'id': 2052, 'filename': 'DSC_2052.jpg', 'score': 0.830},
                    {'id': 2004, 'filename': 'DSC_2004.jpg', 'score': 0.824},
                    {'id': 2094, 'filename': 'DSC_2094.jpg', 'score': 0.821},
                    {'id': 2077, 'filename': 'DSC_2077.jpg', 'score': 0.821},
                    {'id': 2098, 'filename': 'DSC_2098.jpg', 'score': 0.819},
                    {'id': 2049, 'filename': 'DSC_2049.jpg', 'score': 0.818},
                    {'id': 2010, 'filename': 'DSC_2010.jpg', 'score': 0.816},
                    {'id': 2104, 'filename': 'DSC_2104.jpg', 'score': 0.810},
                    {'id': 2019, 'filename': 'DSC_2019.jpg', 'score': 0.804},
                    {'id': 2120, 'filename': 'DSC_2120.jpg', 'score': 0.803},
                    {'id': 2101, 'filename': 'DSC_2101.jpg', 'score': 0.799}
                ]
            elif album_tag == 'enneagram-decisive-moments':
                return [
                    {'id': 2040, 'filename': 'DSC_2040.jpg', 'confidence': 0.944},
                    {'id': 2132, 'filename': 'DSC_2132.jpg', 'confidence': 0.937},
                    {'id': 2131, 'filename': 'DSC_2131.jpg', 'confidence': 0.936},
                    {'id': 2082, 'filename': 'DSC_2082.jpg', 'confidence': 0.923},
                    {'id': 2080, 'filename': 'DSC_2080.jpg', 'confidence': 0.915},
                    {'id': 2102, 'filename': 'DSC_2102.jpg', 'confidence': 0.914},
                    {'id': 2134, 'filename': 'DSC_2134.jpg', 'confidence': 0.902},
                    {'id': 2030, 'filename': 'DSC_2030.jpg', 'confidence': 0.898},
                    {'id': 2049, 'filename': 'DSC_2049.jpg', 'confidence': 0.863},
                    {'id': 2098, 'filename': 'DSC_2098.jpg', 'confidence': 0.860},
                    {'id': 2018, 'filename': 'DSC_2018.jpg', 'confidence': 0.819},
                    {'id': 2010, 'filename': 'DSC_2010.jpg', 'confidence': 0.817}
                ]
            return []
        except Exception as e:
            logger.error(f"Error getting album photos preview: {e}")
            return []


# Initialize data provider
data_provider = WebDataProvider()


@web_viewer.route('/')
def index():
    """Album gallery homepage."""
    try:
        albums = data_provider.get_albums_summary()
        return render_template('viewer/index.html', albums=albums)
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return render_template('viewer/error.html', error="Failed to load albums"), 500


@web_viewer.route('/album/<album_tag>')
def album_view(album_tag: str):
    """Individual album view with photo grid."""
    try:
        # Get album info
        albums = data_provider.get_albums_summary()
        album = next((a for a in albums if a['tag_name'] == album_tag), None)
        
        if not album:
            return render_template('viewer/error.html', error="Album not found"), 404
        
        # Get photos for this album
        photos = data_provider.get_album_photos_preview(album_tag, limit=50)
        
        return render_template('viewer/album.html', album=album, photos=photos)
    except Exception as e:
        logger.error(f"Error in album_view route: {e}")
        return render_template('viewer/error.html', error="Failed to load album"), 500


@web_viewer.route('/photo/<int:photo_id>')
def photo_detail(photo_id: int):
    """Individual photo detail view."""
    try:
        # Mock photo data - in real implementation would query database
        photo = {
            'id': photo_id,
            'filename': f'DSC_{photo_id}.jpg',
            'captured_at': '2025-01-15',
            'camera_make': 'SONY',
            'camera_model': 'ILCE-7RM3',
            'lens_model': 'FE 24-70mm F2.8 GM',
            'aperture': 'f/2.8',
            'shutter_speed': '1/200',
            'iso': 800,
            'focal_length': '50mm',
            'scores': {
                'overall': 0.85,
                'technical': 0.82,
                'artistic': 0.87,
                'emotional': 0.86
            }
        }
        
        return render_template('viewer/photo.html', photo=photo)
    except Exception as e:
        logger.error(f"Error in photo_detail route: {e}")
        return render_template('viewer/error.html', error="Failed to load photo"), 500


@web_viewer.route('/api/albums')
def api_albums():
    """JSON API endpoint for albums (for AJAX requests)."""
    try:
        albums = data_provider.get_albums_summary()
        return jsonify({'success': True, 'albums': albums})
    except Exception as e:
        logger.error(f"Error in api_albums: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@web_viewer.route('/api/album/<album_tag>/photos')
def api_album_photos(album_tag: str):
    """JSON API endpoint for album photos (for AJAX requests)."""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        photos = data_provider.get_album_photos_preview(album_tag, limit=per_page * page)
        
        # Simple pagination simulation
        start = (page - 1) * per_page
        end = start + per_page
        page_photos = photos[start:end]
        
        return jsonify({
            'success': True,
            'photos': page_photos,
            'page': page,
            'per_page': per_page,
            'total': len(photos),
            'has_more': end < len(photos)
        })
    except Exception as e:
        logger.error(f"Error in api_album_photos: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def register_web_routes(app):
    """Register web viewer routes with the Flask app."""
    app.register_blueprint(web_viewer)
    logger.info("Web viewer routes registered")