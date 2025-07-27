#!/usr/bin/env python3
"""
Test Google Drive upload functionality with service account
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

from photosight.storage.gdrive_manager import GoogleDriveManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_file() -> str:
    """Create a test file to upload."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        test_content = f"""
PhotoSight Test File
Created: {os.popen('date').read().strip()}
Purpose: Testing Google Drive service account upload

This is a test file to verify that the PhotoSight service account
can successfully upload files to the configured Google Drive folders.
"""
        f.write(test_content.encode())
        return f.name

def test_folder_config():
    """Test that folder configuration is loaded correctly."""
    print("🔧 Testing folder configuration...")
    
    manager = GoogleDriveManager()
    print(f"📁 Folder IDs loaded:")
    for folder_type, folder_id in manager.folder_ids.items():
        print(f"  {folder_type}: {folder_id}")
    
    if not manager.folder_ids.get('main'):
        print("❌ No folder configuration found")
        return False
    
    print("✅ Folder configuration loaded successfully")
    return True

def test_authentication():
    """Test service account authentication."""
    print("\n🔐 Testing authentication...")
    
    manager = GoogleDriveManager()
    if manager.authenticate():
        print("✅ Service account authentication successful")
        return True
    else:
        print("❌ Service account authentication failed")
        return False

def test_folder_access():
    """Test access to configured folders."""
    print("\n📁 Testing folder access...")
    
    manager = GoogleDriveManager()
    if not manager.authenticate():
        print("❌ Authentication failed")
        return False
    
    # Check access to each configured folder
    for folder_type, folder_id in manager.folder_ids.items():
        if folder_id:
            try:
                folder = manager.service.files().get(fileId=folder_id, fields='id,name').execute()
                print(f"✅ {folder_type}: {folder.get('name')} ({folder_id})")
            except Exception as e:
                print(f"❌ {folder_type}: Access failed - {e}")
                return False
    
    return True

def test_file_upload():
    """Test uploading a file to the RAW Photos folder."""
    print("\n📤 Testing file upload...")
    
    # Create test file
    test_file = create_test_file()
    print(f"📝 Created test file: {test_file}")
    
    try:
        manager = GoogleDriveManager()
        
        # Upload to RAW Photos folder
        drive_photo = manager.upload_raw_photo(test_file)
        
        if drive_photo:
            print(f"✅ Upload successful!")
            print(f"   File ID: {drive_photo.file_id}")
            print(f"   Name: {drive_photo.name}")
            print(f"   Size: {drive_photo.size} bytes")
            print(f"   View: {drive_photo.web_view_link}")
            
            # Clean up - delete the test file from Google Drive
            try:
                manager.service.files().delete(fileId=drive_photo.file_id).execute()
                print(f"🗑️  Cleaned up test file from Google Drive")
            except Exception as e:
                print(f"⚠️  Failed to clean up test file: {e}")
            
            return True
        else:
            print("❌ Upload failed")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False
    finally:
        # Clean up local test file
        if os.path.exists(test_file):
            os.unlink(test_file)
            print(f"🗑️  Cleaned up local test file")

def main():
    print("🚀 PhotoSight Google Drive Integration Test")
    print("=" * 50)
    
    tests = [
        ("Folder Configuration", test_folder_config),
        ("Authentication", test_authentication),
        ("Folder Access", test_folder_access),
        ("File Upload", test_file_upload)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Google Drive integration is working correctly.")
        return 0
    else:
        print("💥 Some tests failed. Please check the configuration and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())