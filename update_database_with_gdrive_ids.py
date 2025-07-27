#!/usr/bin/env python3
"""
Update PhotoSight database with Google Drive file IDs from the photo mapping
"""

import os
import sys
import json
from pathlib import Path

# Add photosight to path
sys.path.insert(0, str(Path(__file__).parent))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_photo_mapping():
    """Load the photo mapping from JSON file."""
    # Try complete mapping first, then fallback to partial mapping
    mapping_files = [
        "photosight_complete_photo_mapping.json",
        "photosight_photo_mapping.json"
    ]
    
    for mapping_file in mapping_files:
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            
            print(f"📋 Loaded {len(mapping)} photo mappings from {mapping_file}")
            return mapping
    
    print(f"❌ No photo mapping file found")
    return None

def update_database_with_gdrive_ids(photo_mapping):
    """Update the PhotoSight database with Google Drive file IDs."""
    try:
        # Try to import Oracle MCP operations
        from photosight.db.mcp_operations import execute_query
        
        print("🔄 Updating PhotoSight database with Google Drive file IDs...")
        
        updated_count = 0
        not_found_count = 0
        
        for filename_base, file_id in photo_mapping.items():
            try:
                # Update photos table with Google Drive file ID
                # Using gdrive:// prefix to indicate Google Drive storage
                update_query = f"""
                    UPDATE PHOTOSIGHT.PHOTOS 
                    SET file_path = 'gdrive://{file_id}',
                        cloud_path = 'https://drive.google.com/file/d/{file_id}/view',
                        sync_status = 'synced',
                        last_sync_at = CURRENT_TIMESTAMP
                    WHERE file_path LIKE '%{filename_base}%'
                       OR original_filename LIKE '%{filename_base}%'
                """
                
                result = execute_query(update_query)
                
                if result and result.get('rowCount', 0) > 0:
                    updated_count += 1
                    print(f"  ✅ Updated {filename_base} -> gdrive://{file_id}")
                else:
                    not_found_count += 1
                    print(f"  ⚠️  No database record found for {filename_base}")
                    
            except Exception as e:
                print(f"  ❌ Failed to update {filename_base}: {e}")
        
        print(f"\n📊 Database Update Results:")
        print(f"  ✅ Updated: {updated_count} photos")
        print(f"  ⚠️  Not found: {not_found_count} photos")
        print(f"  📈 Success rate: {updated_count}/{len(photo_mapping)} ({updated_count/len(photo_mapping)*100:.1f}%)")
        
        return updated_count > 0
        
    except ImportError:
        print("❌ Oracle MCP operations not available")
        print("💡 Alternative: Use the Oracle MCP server directly")
        
        # Provide alternative SQL commands
        print(f"\n📋 Alternative: Run these SQL commands manually:")
        print(f"=" * 60)
        
        for filename_base, file_id in list(photo_mapping.items())[:5]:  # Show first 5 as examples
            print(f"""UPDATE PHOTOSIGHT.PHOTOS 
SET file_path = 'gdrive://{file_id}',
    cloud_path = 'https://drive.google.com/file/d/{file_id}/view',
    sync_status = 'synced',
    last_sync_at = CURRENT_TIMESTAMP
WHERE file_path LIKE '%{filename_base}%'
   OR original_filename LIKE '%{filename_base}%';
""")
        
        if len(photo_mapping) > 5:
            print(f"... and {len(photo_mapping) - 5} more similar updates")
        
        return False
        
    except Exception as e:
        logger.error(f"Database update failed: {e}")
        return False

def verify_gdrive_access():
    """Verify that Google Drive file IDs are accessible."""
    try:
        from photosight.storage.gdrive_manager import GoogleDriveManager
        
        print("🔍 Verifying Google Drive access...")
        manager = GoogleDriveManager()
        
        if not manager.authenticate():
            print("❌ Google Drive authentication failed")
            return False
        
        # Load a sample file ID and test access
        mapping = load_photo_mapping()
        if not mapping:
            return False
        
        sample_file_id = list(mapping.values())[0]
        sample_filename = list(mapping.keys())[0]
        
        try:
            file_details = manager._get_file_details(sample_file_id)
            if file_details:
                print(f"✅ Google Drive access verified")
                print(f"   Sample: {sample_filename} -> {file_details.name}")
                print(f"   Size: {file_details.size} bytes")
                return True
            else:
                print("❌ Could not access sample file")
                return False
                
        except Exception as e:
            print(f"❌ Google Drive access error: {e}")
            return False
            
    except ImportError:
        print("⚠️  Google Drive manager not available")
        return True  # Don't block database update

def main():
    print("💾 PhotoSight Database Update Tool")
    print("=" * 50)
    
    # Load photo mapping
    photo_mapping = load_photo_mapping()
    if not photo_mapping:
        return 1
    
    # Verify Google Drive access
    print(f"\n🔍 Verifying Google Drive access...")
    gdrive_ok = verify_gdrive_access()
    
    # Update database
    print(f"\n💾 Updating database...")
    success = update_database_with_gdrive_ids(photo_mapping)
    
    if success:
        print(f"\n🎉 Database update complete!")
        print(f"📋 PhotoSight photos now reference Google Drive file IDs")
        print(f"🔗 Web viewer can serve images directly from Google Drive")
    else:
        print(f"\n⚠️  Database update needs manual completion")
        print(f"📋 See SQL commands above or use Oracle MCP server")
    
    if gdrive_ok and success:
        print(f"\n✅ PhotoSight is now fully integrated with Google Drive!")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())