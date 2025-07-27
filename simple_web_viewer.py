#!/usr/bin/env python3
"""
Simple PhotoSight Web Viewer

Minimal web viewer that bypasses all authentication for demo purposes.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, send_file, request
from flask_cors import CORS
import io
from PIL import Image
import rawpy
import numpy as np

# Add photosight to path
sys.path.insert(0, str(Path(__file__).parent))

# Set template folder
template_folder = Path(__file__).parent / 'photosight' / 'api' / 'templates'
app = Flask(__name__, template_folder=str(template_folder))

# Enable CORS for all routes
CORS(app)

# Simple config
app.config.update({
    'SECRET_KEY': 'simple-demo-key',
    'DEBUG': True
})

# Mock data for albums
ALBUMS = [
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

# Mock photo data
PHOTOS = {
    'enneagram-top-30': [
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
    ],
    'enneagram-decisive-moments': [
        {'id': 2040, 'filename': 'DSC_2040.jpg', 'confidence': 0.944},
        {'id': 2132, 'filename': 'DSC_2132.jpg', 'confidence': 0.937},
        {'id': 2131, 'filename': 'DSC_2131.jpg', 'confidence': 0.936},
        {'id': 2082, 'filename': 'DSC_2082.jpg', 'confidence': 0.923},
        {'id': 2080, 'filename': 'DSC_2080.jpg', 'confidence': 0.915},
        {'id': 2102, 'filename': 'DSC_2102.jpg', 'confidence': 0.914},
        {'id': 2134, 'filename': 'DSC_2134.jpg', 'confidence': 0.902},
        {'id': 2030, 'filename': 'DSC_2030.jpg', 'confidence': 0.898},
        {'id': 2049, 'filename': 'DSC_2049.jpg', 'confidence': 0.863},
        {'id': 2098, 'filename': 'DSC_2098.jpg', 'confidence': 0.860}
    ]
}

def generate_placeholder_image(width=300, height=200, text="PhotoSight"):
    """Generate a placeholder image."""
    img = Image.new('RGB', (width, height), color='lightgray')
    output = io.BytesIO()
    img.save(output, format='JPEG', quality=85)
    return output.getvalue()

def find_actual_photo(photo_id):
    """Try to find actual photo file on disk using available ARW files."""
    try:
        # Available ARW files we actually have:
        # DSC04778.ARW, DSC04834.ARW, DSC04835.ARW, DSC04836.ARW, 
        # DSC04837.ARW, DSC04838.ARW, DSC04839.ARW, DSC04872.ARW, DSC04904.ARW
        
        available_arw_files = {
            # Map some of our album photos to available ARW files
            2040: "DSC04778.ARW",  # Top-ranked Decisive Moment
            2080: "DSC04834.ARW",  # Top-ranked overall
            2132: "DSC04835.ARW",  # 2nd Decisive Moment
            2131: "DSC04836.ARW",  # 3rd Decisive Moment
            2082: "DSC04837.ARW",  # 4th Decisive Moment
            2102: "DSC04838.ARW",  # Another high-ranked photo
            2134: "DSC04839.ARW",  # Another high-ranked photo
            2030: "DSC04872.ARW",  # Another Decisive Moment
            2049: "DSC04904.ARW",  # High-ranked Top 30 photo
        }
        
        # If we have a mapped ARW file for this photo ID, use it
        if photo_id in available_arw_files:
            arw_path = f"/Users/sam/temp/raw_ranking_test/{available_arw_files[photo_id]}"
            if os.path.exists(arw_path):
                return arw_path
        
        # Fallback: try to calculate DSC number and search for any available formats
        base_dsc = 4779
        dsc_number = base_dsc + (photo_id - 2000)
        
        search_paths = [
            f"/Users/sam/temp/raw_ranking_test/DSC0{dsc_number}.ARW",
            f"/Users/sam/temp/raw_ranking_test/DSC{dsc_number}.ARW",
            f"/Users/sam/Downloads/workshop_photos/DSC0{dsc_number}.jpg",
            f"/Users/sam/Downloads/enneagram/DSC0{dsc_number}.jpg",
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        return None
        
    except Exception:
        return None

def generate_thumbnail_from_file(file_path, size=(300, 300)):
    """Generate thumbnail from actual photo file (supports JPG and ARW)."""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.arw':
            # Handle Sony ARW files
            with rawpy.imread(file_path) as raw:
                # Extract embedded JPEG thumbnail for speed
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        # Create PIL Image from thumbnail data
                        img = Image.open(io.BytesIO(thumb.data))
                    else:
                        # Fallback: process RAW to RGB
                        rgb = raw.postprocess(use_camera_wb=True, 
                                            half_size=True,  # Faster processing
                                            no_auto_bright=True)
                        img = Image.fromarray(rgb)
                except Exception:
                    # Fallback: full RAW processing
                    rgb = raw.postprocess(use_camera_wb=True, 
                                        half_size=True,
                                        no_auto_bright=True)
                    img = Image.fromarray(rgb)
        else:
            # Handle regular image files (JPG, PNG, etc.)
            img = Image.open(file_path)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
            
        # Generate thumbnail
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Save to bytes
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=85, optimize=True)
        return output.getvalue()
        
    except Exception as e:
        print(f"Error generating thumbnail from {file_path}: {e}")
        return None

@app.route('/')
def root():
    """Redirect to viewer."""
    return '<h1>PhotoSight Web Viewer</h1><p><a href="/viewer">View Albums</a></p>'

@app.route('/viewer')
def index():
    """Album gallery homepage."""
    return render_template('viewer/index.html', albums=ALBUMS)

@app.route('/viewer/album/<album_tag>')
def album_view(album_tag):
    """Individual album view with photo grid."""
    album = next((a for a in ALBUMS if a['tag_name'] == album_tag), None)
    if not album:
        return render_template('viewer/error.html', error="Album not found"), 404
    
    photos = PHOTOS.get(album_tag, [])
    return render_template('viewer/album.html', album=album, photos=photos)

@app.route('/viewer/photo/<int:photo_id>')
def photo_detail(photo_id):
    """Individual photo detail view."""
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

@app.route('/api/v1/photos/<int:photo_id>/thumbnail')
def get_thumbnail(photo_id):
    """Generate thumbnail - try real photo first, fallback to placeholder."""
    size = int(request.args.get('size', 300))
    
    # Try to find actual photo file
    photo_path = find_actual_photo(photo_id)
    if photo_path:
        thumbnail_bytes = generate_thumbnail_from_file(photo_path, (size, size))
        if thumbnail_bytes:
            return send_file(
                io.BytesIO(thumbnail_bytes),
                mimetype='image/jpeg',
                as_attachment=False,
                download_name=f"thumbnail_{photo_id}.jpg"
            )
    
    # Fallback to placeholder
    thumbnail_bytes = generate_placeholder_image(size, size, f"Photo {photo_id}")
    return send_file(
        io.BytesIO(thumbnail_bytes),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name=f"thumbnail_{photo_id}.jpg"
    )

@app.route('/api/v1/photos/<int:photo_id>/preview')
def get_preview(photo_id):
    """Generate preview - try real photo first, fallback to placeholder."""
    # Try to find actual photo file
    photo_path = find_actual_photo(photo_id)
    if photo_path:
        preview_bytes = generate_thumbnail_from_file(photo_path, (1200, 800))
        if preview_bytes:
            return send_file(
                io.BytesIO(preview_bytes),
                mimetype='image/jpeg',
                as_attachment=False,
                download_name=f"preview_{photo_id}.jpg"
            )
    
    # Fallback to placeholder
    preview_bytes = generate_placeholder_image(800, 600, f"Photo {photo_id}\nPreview")
    return send_file(
        io.BytesIO(preview_bytes),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name=f"preview_{photo_id}.jpg"
    )

@app.route('/api/v1/albums')
def api_albums():
    """JSON API for albums."""
    return jsonify({'success': True, 'albums': ALBUMS})

@app.route('/api/v1/albums/<album_tag>/photos')
def api_album_photos(album_tag):
    """JSON API for album photos."""
    photos = PHOTOS.get(album_tag, [])
    return jsonify({
        'success': True,
        'photos': photos,
        'total': len(photos)
    })

@app.route('/health')
def health():
    """Health check."""
    return jsonify({'status': 'healthy', 'message': 'Simple PhotoSight Web Viewer'})

if __name__ == '__main__':
    port = 8765
    print("🚀 Starting Simple PhotoSight Web Viewer...")
    print("📸 PhotoSight Web Viewer is ready!")
    print(f"🌐 Open your browser to: http://localhost:{port}/viewer")
    print(f"📊 Albums: http://localhost:{port}/viewer")
    print("\n💡 This is a simplified demo with placeholder images.")
    print("   The full version would display actual photos from Oracle database.\n")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True,
        use_reloader=True
    )