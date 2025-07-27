#!/usr/bin/env python3
"""
Generate SQL script to update PhotoSight database with Google Drive file IDs
"""

import json
import os

def generate_sql_updates():
    """Generate SQL UPDATE statements for all photos."""
    
    # Load complete photo mapping
    if not os.path.exists("photosight_complete_photo_mapping.json"):
        print("❌ Complete photo mapping not found")
        return
    
    with open("photosight_complete_photo_mapping.json", 'r') as f:
        photo_mapping = json.load(f)
    
    print(f"📋 Generating SQL for {len(photo_mapping)} photos...")
    
    sql_statements = []
    
    # Header comment
    sql_statements.append("""-- PhotoSight Google Drive Integration SQL Update
-- Updates all photo records with Google Drive file IDs
-- Generated automatically from photo mapping

""")
    
    for filename_base, file_id in photo_mapping.items():
        sql = f"""UPDATE PHOTOSIGHT.PHOTOS 
SET file_path = 'gdrive://{file_id}',
    cloud_path = 'https://drive.google.com/file/d/{file_id}/view',
    sync_status = 'synced',
    last_sync_at = CURRENT_TIMESTAMP
WHERE file_path LIKE '%{filename_base}%'
   OR original_filename LIKE '%{filename_base}%';

"""
        sql_statements.append(sql)
    
    # Add verification query at the end
    sql_statements.append("""-- Verification: Count updated records
SELECT 
    COUNT(*) as total_photos,
    COUNT(CASE WHEN file_path LIKE 'gdrive://%' THEN 1 END) as gdrive_photos,
    COUNT(CASE WHEN sync_status = 'synced' THEN 1 END) as synced_photos
FROM PHOTOSIGHT.PHOTOS;

-- Show sample of updated records
SELECT photo_id, original_filename, file_path, cloud_path, sync_status, last_sync_at
FROM PHOTOSIGHT.PHOTOS 
WHERE file_path LIKE 'gdrive://%'
ORDER BY last_sync_at DESC
FETCH FIRST 10 ROWS ONLY;
""")
    
    # Write to file
    output_file = "photosight_gdrive_update.sql"
    with open(output_file, 'w') as f:
        f.writelines(sql_statements)
    
    print(f"💾 Generated SQL script: {output_file}")
    print(f"📊 Contains {len(photo_mapping)} UPDATE statements")
    
    # Show sample
    print(f"\n📋 Sample SQL (first 3 statements):")
    print("=" * 60)
    for statement in sql_statements[1:4]:  # Skip header, show first 3
        print(statement)
    
    print(f"📋 To execute:")
    print(f"   1. Use Oracle MCP server")
    print(f"   2. Run: {output_file}")
    print(f"   3. Verify with the queries at the end")

if __name__ == "__main__":
    generate_sql_updates()