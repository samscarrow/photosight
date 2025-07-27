#!/usr/bin/env python3
"""
PhotoSight Web Viewer Runner

Simple script to run the PhotoSight web viewer for album browsing.
"""

import os
import sys
from pathlib import Path

# Add photosight to path
sys.path.insert(0, str(Path(__file__).parent))

from photosight.api.app import create_app

def main():
    """Run the PhotoSight web viewer."""
    print("🚀 Starting PhotoSight Web Viewer...")
    
    # Create minimal config for web viewer
    config = {
        'DEBUG': True,
        'SECRET_KEY': 'dev-secret-key-for-web-viewer',
        'CORS_ORIGINS': ['http://localhost:5000', 'http://127.0.0.1:5000'],
        'ENABLE_CLEANUP': False,  # Disable cleanup in dev mode
    }
    
    try:
        app = create_app(config)
        
        print("📸 PhotoSight Web Viewer is ready!")
        print("🌐 Open your browser to: http://localhost:5000/viewer")
        print("📊 API endpoints available at: http://localhost:5000/api/v1/")
        print("🏠 Albums: http://localhost:5000/viewer")
        print("\n💡 Note: This is running with mock data for demonstration.")
        print("   In production, it would connect to the Oracle database.\n")
        
        # Run the development server
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True
        )
        
    except Exception as e:
        print(f"❌ Error starting web viewer: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()