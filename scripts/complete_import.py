#!/usr/bin/env python3
"""
Complete the enneagram photo import and album creation by generating
the remaining SQL statements for execution via MCP.
"""

import json
from pathlib import Path

def load_analysis_results():
    """Load the enneagram analysis results."""
    results_file = Path("enneagram_analysis_output/enneagram_analysis_results.json")
    with open(results_file, 'r') as f:
        return json.load(f)

def create_complete_import_sql():
    """Create SQL for importing remaining photos and creating albums."""
    data = load_analysis_results()
    
    # Create photo ID mapping (starting from 2000, we've imported up to 2029)
    filenames = list(data['results'].keys())
    photo_mapping = {}
    
    for i, filename in enumerate(filenames):
        photo_mapping[filename] = 2000 + i
    
    print(f"Creating mapping for {len(filenames)} photos (IDs 2000-{2000 + len(filenames) - 1})")
    
    # Generate remaining photo imports (batches 2-7)
    remaining_inserts = []
    for i in range(30, len(filenames), 25):  # Process remaining in chunks of 25
        batch = filenames[i:i+25]
        
        insert_lines = ["INSERT ALL"]
        for filename in batch:
            photo_id = photo_mapping[filename]
            insert_lines.append(
                f"    INTO PHOTOSIGHT.PHOTOS (id, file_path, capture_date, width, height, file_size, camera_make, camera_model, processing_status, created_at) "
                f"VALUES ({photo_id}, '/enneagram/{filename}', CURRENT_TIMESTAMP, 4000, 2664, 5000000, 'Sony', 'A7R V', 'completed', CURRENT_TIMESTAMP)"
            )
        
        insert_lines.append("SELECT * FROM dual;")
        remaining_inserts.append('\n'.join(insert_lines))
    
    # Write remaining inserts
    with open('remaining_photos.sql', 'w') as f:
        for i, sql in enumerate(remaining_inserts, 1):
            f.write(f"-- Remaining Batch {i}\n")
            f.write(sql)
            f.write('\n\n')
    
    print(f"Generated {len(remaining_inserts)} remaining batches")
    
    # Generate complete album associations
    top_30_inserts = []
    decisive_inserts = []
    
    # Top 30 album
    top_30 = data['rankings']['overall'][:30]
    for rank, entry in enumerate(top_30, 1):
        filename = entry['filename']
        if filename in photo_mapping:
            photo_id = photo_mapping[filename]
            scores = entry['scores']
            scene = entry.get('scene', 'unknown')
            mood = entry.get('mood', 'unknown')
            
            sql = f"""INSERT INTO PHOTOSIGHT.PHOTO_ALBUM_TAGS (
    photo_id, album_tag_id, sort_order, added_by, association_metadata
) VALUES (
    {photo_id}, 1, {rank}, 'enneagram_analysis', 
    JSON_OBJECT(
        'rank' VALUE {rank},
        'score' VALUE {scores['overall']},
        'technical_score' VALUE {scores['technical']},
        'artistic_score' VALUE {scores['artistic']},
        'emotional_score' VALUE {scores['emotional']},
        'scene' VALUE '{scene}',
        'mood' VALUE '{mood}'
    )
);"""
            top_30_inserts.append(sql)
    
    # Decisive moments album
    decisive_moments = data['rankings']['decisive_moments']
    for rank, moment in enumerate(decisive_moments, 1):
        filename = moment['filename']
        if filename in photo_mapping:
            photo_id = photo_mapping[filename]
            confidence = moment['confidence']
            reason = moment.get('reason', '').replace("'", "''")  # Escape quotes
            
            sql = f"""INSERT INTO PHOTOSIGHT.PHOTO_ALBUM_TAGS (
    photo_id, album_tag_id, sort_order, added_by, association_metadata
) VALUES (
    {photo_id}, 2, {rank}, 'enneagram_analysis',
    JSON_OBJECT(
        'rank' VALUE {rank},
        'confidence' VALUE {confidence},
        'reason' VALUE '{reason}',
        'capture_type' VALUE 'decisive_moment'
    )
);"""
            decisive_inserts.append(sql)
    
    # Write complete album SQL
    with open('complete_albums.sql', 'w') as f:
        f.write("-- Clear existing album associations\n")
        f.write("DELETE FROM PHOTOSIGHT.PHOTO_ALBUM_TAGS;\n\n")
        
        f.write("-- Top 30 Album Associations\n")
        for sql in top_30_inserts:
            f.write(sql + '\n')
        
        f.write("\n-- Decisive Moments Album Associations\n")
        for sql in decisive_inserts:
            f.write(sql + '\n')
        
        f.write("\n-- Update album photo counts\n")
        f.write("""UPDATE PHOTOSIGHT.ALBUM_TAGS SET 
    photo_count = (SELECT COUNT(*) FROM PHOTOSIGHT.PHOTO_ALBUM_TAGS WHERE album_tag_id = PHOTOSIGHT.ALBUM_TAGS.id),
    last_photo_added = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP;
""")
    
    print(f"Generated complete albums SQL:")
    print(f"  - {len(top_30_inserts)} Top 30 associations")
    print(f"  - {len(decisive_inserts)} Decisive moment associations")
    
    # Create summary report
    with open('import_summary.txt', 'w') as f:
        f.write("ENNEAGRAM PHOTO IMPORT SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Photos: {len(filenames)}\n")
        f.write(f"Photo ID Range: 2000-{2000 + len(filenames) - 1}\n")
        f.write(f"Albums Created: 2\n")
        f.write(f"  - Top 30: {len(top_30_inserts)} photos\n")
        f.write(f"  - Decisive Moments: {len(decisive_inserts)} photos\n")
        f.write(f"Remaining Import Batches: {len(remaining_inserts)}\n\n")
        
        f.write("FILES GENERATED:\n")
        f.write("- remaining_photos.sql: Import remaining photos\n")
        f.write("- complete_albums.sql: Create all album associations\n")
        f.write("- import_summary.txt: This summary\n")

if __name__ == "__main__":
    create_complete_import_sql()