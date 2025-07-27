#!/usr/bin/env python3
"""
Test Google Drive upload with domain-wide delegation
Run this after configuring Google Workspace Admin Console
"""

import os
import sys
from pathlib import Path
import tempfile
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add photosight to path
sys.path.insert(0, str(Path(__file__).parent))

from photosight.storage.gdrive_delegation_manager import GoogleDriveDelegationManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_file() -> str:
    """Create a test file to upload."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        test_content = f"""
PhotoSight Domain-Wide Delegation Test
Created: {os.popen('date').read().strip()}

This test verifies that the PhotoSight service account can:
1. Authenticate using domain-wide delegation
2. Access sscarrow@gmail.com's Google Drive
3. Upload files to PhotoSight folders
4. Manage files without storage quota issues

If you can see this file in Google Drive, delegation is working! 🎉
"""
        f.write(test_content.encode())
        return f.name

def test_delegation_upload():
    """Test uploading a file using domain-wide delegation."""
    print("🚀 PhotoSight Domain-Wide Delegation Test")
    print("=" * 50)
    
    # Create test file
    test_file = create_test_file()
    print(f"📝 Created test file: {test_file}")
    
    try:
        manager = GoogleDriveDelegationManager()
        
        print("🔐 Testing authentication...")
        if not manager.authenticate():
            print("❌ Authentication failed")
            print("💡 Make sure you configured domain-wide delegation in Google Workspace Admin Console")
            print("📖 See: setup_domain_wide_delegation.md")
            return False
        
        print("✅ Authentication successful!")
        
        print("📁 Testing folder access...")
        folders = manager.list_folders()
        print(f"✅ Found {len(folders)} accessible folders")
        
        # Find PhotoSight folder
        photosight_folders = [f for f in folders if f['name'] == 'PhotoSight']
        if not photosight_folders:
            print("❌ PhotoSight folder not found")
            return False
        
        print(f"✅ PhotoSight folder found: {photosight_folders[0]['id']}")
        
        print("📤 Testing file upload to RAW Photos...")
        drive_photo = manager.upload_raw_photo(test_file)
        
        if drive_photo:
            print("✅ Upload successful!")
            print(f"   File ID: {drive_photo.file_id}")
            print(f"   Name: {drive_photo.name}")
            print(f"   Size: {drive_photo.size} bytes")
            print(f"   View: {drive_photo.web_view_link}")
            
            print("🗑️  Cleaning up test file...")
            try:
                manager.service.files().delete(fileId=drive_photo.file_id).execute()
                print("✅ Test file deleted from Google Drive")
            except Exception as e:
                print(f"⚠️  Failed to clean up test file: {e}")
            
            print("\n🎉 Domain-wide delegation is working correctly!")
            print("📋 You can now:")
            print("   • Upload photos to Google Drive without quota issues")
            print("   • Move existing photos to PhotoSight folders")
            print("   • Use PhotoSight with full Google Drive integration")
            
            return True
        else:
            print("❌ Upload failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
    finally:
        # Clean up local test file
        if os.path.exists(test_file):
            os.unlink(test_file)
            print(f"🗑️  Cleaned up local test file")

if __name__ == "__main__":
    success = test_delegation_upload()
    sys.exit(0 if success else 1)