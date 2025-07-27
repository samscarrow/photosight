#!/usr/bin/env python3
"""
Import all enneagram photos and create complete albums using Oracle MCP.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_analysis_results():
    """Load the enneagram analysis results."""
    results_file = Path("enneagram_analysis_output/enneagram_analysis_results.json")
    with open(results_file, 'r') as f:
        return json.load(f)

def generate_photo_inserts(data):
    """Generate SQL INSERT statements for all photos."""
    inserts = []
    photo_id = 2000  # Start from 2000 to avoid conflicts
    
    for filename, result in data['results'].items():
        insert_sql = f"""
INSERT INTO PHOTOSIGHT.PHOTOS (
    id, file_path, capture_date, width, height, file_size,
    camera_make, camera_model, processing_status, created_at
) VALUES (
    {photo_id}, '/enneagram/{filename}', CURRENT_TIMESTAMP, 4000, 2664, 5000000,
    'Sony', 'A7R V', 'completed', CURRENT_TIMESTAMP
)"""
        inserts.append((photo_id, filename, insert_sql.strip()))
        photo_id += 1
    
    return inserts

def generate_top30_album_inserts(data, photo_mapping):
    """Generate album associations for top 30 photos."""
    inserts = []
    top_30 = data['rankings']['overall'][:30]
    
    for rank, entry in enumerate(top_30, 1):
        filename = entry['filename']
        if filename in photo_mapping:
            photo_id = photo_mapping[filename]
            scores = entry['scores']
            scene = entry.get('scene', 'unknown')
            mood = entry.get('mood', 'unknown')
            
            metadata_json = f"""JSON_OBJECT(
                'rank' VALUE {rank},
                'score' VALUE {scores['overall']},
                'technical_score' VALUE {scores['technical']},
                'artistic_score' VALUE {scores['artistic']},
                'emotional_score' VALUE {scores['emotional']},
                'scene' VALUE '{scene}',
                'mood' VALUE '{mood}'
            )"""
            
            insert_sql = f"""
INSERT INTO PHOTOSIGHT.PHOTO_ALBUM_TAGS (
    photo_id, album_tag_id, sort_order, added_by, association_metadata
) VALUES (
    {photo_id}, 1, {rank}, 'enneagram_analysis', {metadata_json}
)"""
            inserts.append(insert_sql.strip())
    
    return inserts

def generate_decisive_moments_inserts(data, photo_mapping):
    """Generate album associations for decisive moments."""
    inserts = []
    decisive_moments = data['rankings']['decisive_moments']
    
    for rank, moment in enumerate(decisive_moments, 1):
        filename = moment['filename']
        if filename in photo_mapping:
            photo_id = photo_mapping[filename]
            confidence = moment['confidence']
            reason = moment.get('reason', '').replace("'", "''")  # Escape quotes
            
            metadata_json = f"""JSON_OBJECT(
                'rank' VALUE {rank},
                'confidence' VALUE {confidence},
                'reason' VALUE '{reason}',
                'capture_type' VALUE 'decisive_moment'
            )"""
            
            insert_sql = f"""
INSERT INTO PHOTOSIGHT.PHOTO_ALBUM_TAGS (
    photo_id, album_tag_id, sort_order, added_by, association_metadata
) VALUES (
    {photo_id}, 2, {rank}, 'enneagram_analysis', {metadata_json}
)"""
            inserts.append(insert_sql.strip())
    
    return inserts

def main():
    """Generate all SQL statements for import."""
    print("🎯 Generating SQL for Complete Enneagram Import")
    
    # Load analysis data
    data = load_analysis_results()
    print(f"Loaded data for {len(data['results'])} photos")
    
    # Generate photo inserts
    photo_inserts = generate_photo_inserts(data)
    photo_mapping = {filename: photo_id for photo_id, filename, _ in photo_inserts}
    
    print(f"Generated {len(photo_inserts)} photo insert statements")
    
    # Write photo inserts to file
    with open('photo_inserts.sql', 'w') as f:
        for _, _, sql in photo_inserts:
            f.write(sql + ';\n\n')
    
    # Generate album inserts
    top30_inserts = generate_top30_album_inserts(data, photo_mapping)
    decisive_inserts = generate_decisive_moments_inserts(data, photo_mapping)
    
    print(f"Generated {len(top30_inserts)} top-30 album associations")
    print(f"Generated {len(decisive_inserts)} decisive moment associations")
    
    # Write album inserts to file
    with open('album_inserts.sql', 'w') as f:
        f.write("-- Top 30 Album Associations\n")
        for sql in top30_inserts:
            f.write(sql + ';\n\n')
        
        f.write("\n-- Decisive Moments Album Associations\n")
        for sql in decisive_inserts:
            f.write(sql + ';\n\n')
        
        f.write("\n-- Update album photo counts\n")
        f.write("""
UPDATE PHOTOSIGHT.ALBUM_TAGS SET 
    photo_count = (SELECT COUNT(*) FROM PHOTOSIGHT.PHOTO_ALBUM_TAGS WHERE album_tag_id = PHOTOSIGHT.ALBUM_TAGS.id),
    last_photo_added = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP;
""")
    
    print("\n✅ SQL files generated:")
    print("   - photo_inserts.sql: All photo records")
    print("   - album_inserts.sql: Album associations and updates")
    print("\nNext steps:")
    print("1. Execute photo_inserts.sql to import all photos")
    print("2. Execute album_inserts.sql to create album associations")

if __name__ == "__main__":
    main()