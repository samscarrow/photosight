#!/usr/bin/env python3
"""
Move existing photos from sscarrow@gmail.com Drive to PhotoSight folders

This script can:
1. List photos in existing folders 
2. Move them to PhotoSight organized folders
3. Update database with new Google Drive file IDs

The service account can manage files that are shared with it!
"""

import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add photosight to path
sys.path.insert(0, str(Path(__file__).parent))

from photosight.storage.gdrive_manager import GoogleDriveManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_existing_photos(manager, source_folder_name="USB and External Devices"):
    """Find existing photos in the source folder."""
    try:
        # Search for the source folder
        folders = manager.list_folders(source_folder_name)
        if not folders:
            print(f"❌ Folder '{source_folder_name}' not found")
            return []
        
        source_folder_id = folders[0]['id']
        print(f"📁 Found source folder: {source_folder_name} ({source_folder_id})")
        
        # List ARW files in the folder
        query = f"'{source_folder_id}' in parents and name contains '.ARW'"
        results = manager.service.files().list(
            q=query,
            fields="files(id,name,size,parents)",
            pageSize=100
        ).execute()
        
        photos = results.get('files', [])
        print(f"📸 Found {len(photos)} ARW files in source folder")
        
        return photos
        
    except Exception as e:
        logger.error(f"Error finding photos: {e}")
        return []

def move_photos_to_photosight(manager, photos, target_folder_type="enneagram"):
    """Move photos to PhotoSight organized folders."""
    try:
        target_folder_id = manager.folder_ids.get(target_folder_type)
        if not target_folder_id:
            print(f"❌ Target folder ID not configured for: {target_folder_type}")
            return []
        
        print(f"📤 Moving {len(photos)} photos to {target_folder_type} folder...")
        
        moved_photos = []
        for i, photo in enumerate(photos, 1):
            try:
                # Get current parents
                current_parents = ",".join(photo.get('parents', []))
                
                # Move file
                manager.service.files().update(
                    fileId=photo['id'],
                    addParents=target_folder_id,
                    removeParents=current_parents,
                    fields='id,parents'
                ).execute()
                
                moved_photos.append(photo)
                print(f"  ✅ {i}/{len(photos)}: {photo['name']} -> {photo['id']}")
                
            except Exception as e:
                print(f"  ❌ Failed to move {photo['name']}: {e}")
        
        print(f"✅ Successfully moved {len(moved_photos)}/{len(photos)} photos")
        return moved_photos
        
    except Exception as e:
        logger.error(f"Error moving photos: {e}")
        return []

def find_photos_in_untitled_folder(manager):
    """Find ARW photos in the Untitled folder."""
    try:
        # The Untitled folder ID we just shared
        untitled_folder_id = "1-6IiS3HL87sg4JBf7mmTBr2FQL46SRlJ"
        
        # Search for ARW files in the Untitled folder
        query = f"'{untitled_folder_id}' in parents and name contains '.ARW'"
        results = manager.service.files().list(
            q=query,
            fields="files(id,name,size,parents)",
            pageSize=1000  # Get all photos
        ).execute()
        
        photos = results.get('files', [])
        return photos
        
    except Exception as e:
        logger.error(f"Error finding photos in Untitled folder: {e}")
        return []

def create_photo_mapping(moved_photos):
    """Create filename to Google Drive ID mapping for database updates."""
    mapping = {}
    for photo in moved_photos:
        # Extract base filename (e.g., DSC04778.ARW -> DSC04778)
        filename_base = Path(photo['name']).stem
        mapping[filename_base] = photo['id']
    
    return mapping

def save_photo_mapping(mapping):
    """Save photo mapping to JSON file for database updates."""
    import json
    
    output_file = "photosight_photo_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"💾 Saved photo mapping to: {output_file}")
    print(f"📋 Use this file to update the PhotoSight database with Google Drive IDs")

def main():
    print("📁 PhotoSight Photo Organization Tool")
    print("=" * 50)
    
    print("🔐 Authenticating with Google Drive...")
    manager = GoogleDriveManager()
    
    if not manager.authenticate():
        print("❌ Authentication failed")
        print("💡 Make sure folders are shared with: photosight-storage@home-dev-sam.iam.gserviceaccount.com")
        return 1
    
    print("✅ Authentication successful")
    
    # Check folder access
    print("\n📋 Checking PhotoSight folder access...")
    for folder_type, folder_id in manager.folder_ids.items():
        if folder_id:
            try:
                folder = manager.service.files().get(fileId=folder_id, fields='id,name').execute()
                print(f"  ✅ {folder_type}: {folder.get('name')} ({folder_id})")
            except Exception as e:
                print(f"  ❌ {folder_type}: Access failed - {e}")
    
    # List source folders that might contain photos
    print("\n📁 Searching for photo source folders...")
    all_folders = manager.list_folders()
    
    photo_folders = [f for f in all_folders if any(keyword in f['name'].lower() 
                    for keyword in ['usb', 'external', 'sony', 'photo', 'raw', 'arw'])]
    
    if not photo_folders:
        print("❌ No potential photo folders found")
        print("💡 Make sure source folders are shared with the service account")
        return 1
    
    print(f"📸 Found {len(photo_folders)} potential photo folders:")
    for i, folder in enumerate(photo_folders, 1):
        print(f"  {i}. {folder['name']} ({folder['id']})")
    
    # Move photos from the Untitled folder to PhotoSight Enneagram folder
    print(f"\n🚀 Moving ARW files to PhotoSight organization...")
    
    # Find photos in the Untitled folder
    photos = find_photos_in_untitled_folder(manager)
    if photos:
        print(f"📸 Found {len(photos)} ARW files to move")
        
        # Proceed with moving photos
        print(f"\n🚀 Proceeding to move {len(photos)} ARW files to PhotoSight Enneagram folder...")
        if True:  # Auto-confirm
            moved = move_photos_to_photosight(manager, photos, "enneagram")
            if moved:
                mapping = create_photo_mapping(moved)
                print(f"📊 Created mapping for {len(mapping)} photos")
                
                # Save mapping to file for database updates
                save_photo_mapping(mapping)
            else:
                print("❌ No photos were moved")
        else:
            print("❌ Photo movement cancelled")
    else:
        print("❌ No ARW files found in shared folders")
    
    print(f"\n✅ PhotoSight can manage shared Google Drive files!")
    print(f"🔑 Share more folders with: photosight-storage@home-dev-sam.iam.gserviceaccount.com")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())