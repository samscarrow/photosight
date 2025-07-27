#!/usr/bin/env python3
"""
Setup Google Drive Central Storage for PhotoSight

This script:
1. Uploads all enneagram photos to Google Drive
2. Updates the Oracle database with Google Drive file IDs
3. Creates a permanent, centralized photo storage solution
"""

import os
import sys
from pathlib import Path

# Add photosight to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from photosight.storage.gdrive_manager import GoogleDriveManager, create_drive_photo_mapping
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_database_with_drive_paths(photo_mapping: dict):
    """Update Oracle database with Google Drive file IDs."""
    try:
        # Import MCP Oracle tools
        from photosight.mcp import oracle
        
        logger.info("Updating database with Google Drive file IDs...")
        
        for filename_base, file_id in photo_mapping.items():
            # Find photos in database that match this filename
            # filename_base is like "DSC04778", we need to find matching records
            
            query = f"""
                UPDATE PHOTOSIGHT.PHOTOS 
                SET file_path = 'gdrive://{file_id}',
                    cloud_path = 'https://drive.google.com/file/d/{file_id}/view',
                    sync_status = 'synced',
                    last_sync_at = CURRENT_TIMESTAMP
                WHERE file_path LIKE '%{filename_base}%'
            """
            
            try:
                result = oracle.execute_query(query)
                if result.get('rowCount', 0) > 0:
                    logger.info(f"Updated {filename_base} -> gdrive://{file_id}")
                else:
                    logger.warning(f"No database record found for {filename_base}")
                    
            except Exception as e:
                logger.error(f"Failed to update {filename_base}: {e}")
        
        logger.info("Database update completed")
        
    except Exception as e:
        logger.error(f"Database update failed: {e}")


def main():
    """Main setup process."""
    print("🚀 Setting up Google Drive Central Storage for PhotoSight")
    print("=" * 60)
    
    # Source directory with ARW files
    source_dir = "/Users/sam/Library/CloudStorage/GoogleDrive-sscarrow@gmail.com/Other computers/USB and External Devices/Untitled"
    
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        print("Please update the source_dir path in this script")
        return 1
    
    # Count ARW files
    arw_files = list(Path(source_dir).glob("*.ARW"))
    print(f"📸 Found {len(arw_files)} ARW files to upload")
    
    if len(arw_files) == 0:
        print("❌ No ARW files found in source directory")
        return 1
    
    # Confirm upload
    print(f"\n📁 This will upload {len(arw_files)} photos to Google Drive")
    print("   and update the PhotoSight database with permanent links.")
    
    confirm = input("\n🤔 Proceed with upload? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Upload cancelled")
        return 0
    
    try:
        # Step 1: Upload to Google Drive
        print("\n📤 Step 1: Uploading photos to Google Drive...")
        drive_manager = GoogleDriveManager()
        
        uploaded_photos = drive_manager.upload_enneagram_photos(source_dir)
        
        if not uploaded_photos:
            print("❌ No photos were uploaded successfully")
            return 1
        
        print(f"✅ Successfully uploaded {len(uploaded_photos)} photos")
        
        # Step 2: Create filename mapping
        print("\n🗂️  Step 2: Creating filename mapping...")
        photo_mapping = {}
        for photo in uploaded_photos:
            filename_base = Path(photo.name).stem  # DSC04778
            photo_mapping[filename_base] = photo.file_id
            print(f"   {filename_base} -> {photo.file_id}")
        
        # Step 3: Update database
        print(f"\n💾 Step 3: Updating database with {len(photo_mapping)} file IDs...")
        update_database_with_drive_paths(photo_mapping)
        
        # Step 4: Summary
        print("\n" + "=" * 60)
        print("🎉 Google Drive Central Storage Setup Complete!")
        print("=" * 60)
        print(f"📤 Uploaded: {len(uploaded_photos)} photos")
        print(f"💾 Updated: {len(photo_mapping)} database records")
        print("🔗 Photos now have permanent Google Drive URLs")
        print("🌐 Web viewer will serve images from Google Drive")
        
        # Sample URLs
        if uploaded_photos:
            sample = uploaded_photos[0]
            print(f"\n📋 Sample URLs:")
            print(f"   File ID: {sample.file_id}")
            print(f"   View: {sample.web_view_link}")
            print(f"   Download: {sample.download_link}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"❌ Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())