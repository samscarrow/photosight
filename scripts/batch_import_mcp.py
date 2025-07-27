#!/usr/bin/env python3
"""
Batch import all enneagram photos using Oracle MCP in efficient batches.
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

def generate_batch_inserts():
    """Generate INSERT ALL statements for remaining photos."""
    data = load_analysis_results()
    
    # Get all filenames and create mapping
    filenames = list(data['results'].keys())
    
    # We already imported first 10 (2000-2009), start from 2010
    batches = []
    batch_size = 20
    start_id = 2010
    
    for i in range(10, len(filenames), batch_size):  # Start from index 10
        batch_filenames = filenames[i:i+batch_size]
        
        # Create INSERT ALL statement
        insert_lines = ["INSERT ALL"]
        
        for j, filename in enumerate(batch_filenames):
            photo_id = start_id + j
            insert_lines.append(
                f"    INTO PHOTOSIGHT.PHOTOS (id, file_path, capture_date, width, height, file_size, camera_make, camera_model, processing_status, created_at) "
                f"VALUES ({photo_id}, '/enneagram/{filename}', CURRENT_TIMESTAMP, 4000, 2664, 5000000, 'Sony', 'A7R V', 'completed', CURRENT_TIMESTAMP)"
            )
        
        insert_lines.append("SELECT * FROM dual")
        
        batches.append({
            'batch_num': len(batches) + 1,
            'start_id': start_id,
            'count': len(batch_filenames),
            'sql': '\n'.join(insert_lines)
        })
        
        start_id += batch_size
    
    return batches, data

def main():
    """Generate batch import statements."""
    print("📸 Generating Enneagram Photo Import Batches")
    
    batches, data = generate_batch_inserts()
    
    print(f"Generated {len(batches)} batches for importing remaining photos")
    print(f"Total photos to import: {len(data['results']) - 10}")  # Minus 10 already imported
    
    # Write batch files
    for batch in batches:
        filename = f"batch_{batch['batch_num']:02d}.sql"
        with open(filename, 'w') as f:
            f.write(f"-- Batch {batch['batch_num']}: {batch['count']} photos starting from ID {batch['start_id']}\n")
            f.write(batch['sql'])
            f.write(";\n")
        
        print(f"  Batch {batch['batch_num']:2d}: {batch['count']:2d} photos -> {filename}")
    
    print(f"\n✅ Created {len(batches)} batch files")
    print("Execute each batch file with the Oracle MCP to import photos")

if __name__ == "__main__":
    main()