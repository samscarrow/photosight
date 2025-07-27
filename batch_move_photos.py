#!/usr/bin/env python3
"""
Batch move photos with progress tracking and error handling
"""

import os
import sys
import time
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

def batch_move_photos(manager, photos, target_folder_id, batch_size=10):
    """Move photos in batches with progress tracking."""
    total_photos = len(photos)
    moved_photos = []
    failed_photos = []
    
    print(f"📤 Moving {total_photos} photos in batches of {batch_size}...")
    
    for i in range(0, total_photos, batch_size):
        batch = photos[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_photos + batch_size - 1) // batch_size
        
        print(f"\n📦 Batch {batch_num}/{total_batches}: Moving {len(batch)} photos...")
        
        for j, photo in enumerate(batch):
            try:
                photo_num = i + j + 1
                print(f"  {photo_num:3d}/{total_photos}: {photo['name'][:50]}...", end=" ")
                
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
                print("✅")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                failed_photos.append((photo, str(e)))
                print(f"❌ {e}")
        
        # Progress summary after each batch
        success_rate = len(moved_photos) / photo_num * 100
        print(f"📊 Batch {batch_num} complete: {len(moved_photos)}/{photo_num} moved ({success_rate:.1f}%)")
        
        # Longer delay between batches
        if i + batch_size < total_photos:
            time.sleep(1)
    
    return moved_photos, failed_photos

def main():
    print("📁 PhotoSight Batch Photo Organizer")
    print("=" * 50)
    
    # Authenticate
    manager = GoogleDriveManager()
    if not manager.authenticate():
        print("❌ Authentication failed")
        return 1
    
    print("✅ Authentication successful")
    
    # Get target folder
    enneagram_folder_id = manager.folder_ids.get('enneagram')
    if not enneagram_folder_id:
        print("❌ Enneagram folder ID not configured")
        return 1
    
    print(f"📁 Target folder: {enneagram_folder_id}")
    
    # Find photos in Untitled folder
    untitled_folder_id = "1-6IiS3HL87sg4JBf7mmTBr2FQL46SRlJ"
    
    print("🔍 Finding ARW files in Untitled folder...")
    query = f"'{untitled_folder_id}' in parents and name contains '.ARW'"
    results = manager.service.files().list(
        q=query,
        fields="files(id,name,size,parents)",
        pageSize=1000
    ).execute()
    
    photos = results.get('files', [])
    print(f"📸 Found {len(photos)} ARW files")
    
    if not photos:
        print("❌ No ARW files found")
        return 1
    
    # Move photos in batches
    start_time = time.time()
    moved_photos, failed_photos = batch_move_photos(manager, photos, enneagram_folder_id)
    end_time = time.time()
    
    # Results summary
    print("\n" + "=" * 50)
    print("📊 BATCH MOVE RESULTS")
    print("=" * 50)
    print(f"✅ Successfully moved: {len(moved_photos)}/{len(photos)} photos")
    print(f"❌ Failed to move: {len(failed_photos)} photos")
    print(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
    
    if failed_photos:
        print(f"\n❌ Failed photos:")
        for photo, error in failed_photos[:5]:  # Show first 5 failures
            print(f"  - {photo['name']}: {error}")
        if len(failed_photos) > 5:
            print(f"  ... and {len(failed_photos) - 5} more")
    
    if moved_photos:
        # Create mapping for database updates
        print(f"\n💾 Creating photo mapping...")
        mapping = {}
        for photo in moved_photos:
            filename_base = Path(photo['name']).stem
            mapping[filename_base] = photo['id']
        
        # Save mapping
        import json
        output_file = "photosight_photo_mapping.json"
        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"📋 Saved {len(mapping)} photo mappings to: {output_file}")
        print(f"🔗 Use this file to update PhotoSight database with Google Drive IDs")
    
    print(f"\n🎉 Photo organization complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())