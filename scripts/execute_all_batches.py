#!/usr/bin/env python3
"""
Execute all remaining photo import batches by generating the exact SQL statements.
"""

import json
from pathlib import Path

def load_analysis_results():
    """Load the enneagram analysis results."""
    results_file = Path("enneagram_analysis_output/enneagram_analysis_results.json")
    with open(results_file, 'r') as f:
        return json.load(f)

def generate_all_remaining_inserts():
    """Generate all remaining INSERT statements for batches 3-7."""
    data = load_analysis_results()
    filenames = list(data['results'].keys())
    
    # We've imported 0-49 (batches 1-2), need to import 50-146
    remaining_batches = []
    
    # Batch 3: 2050-2069 (DSC04829.jpg onwards)
    for batch_num in range(3, 8):  # Batches 3, 4, 5, 6, 7
        start_idx = (batch_num - 1) * 20  # Batch 3 starts at index 40, etc.
        end_idx = min(start_idx + 20, len(filenames))
        batch_filenames = filenames[start_idx:end_idx]
        
        if not batch_filenames:
            break
            
        start_id = 2000 + start_idx
        
        # Generate INSERT ALL statement
        insert_lines = ["INSERT ALL"]
        
        for i, filename in enumerate(batch_filenames):
            photo_id = start_id + i
            insert_lines.append(
                f"    INTO PHOTOSIGHT.PHOTOS (id, file_path, capture_date, width, height, file_size, camera_make, camera_model, processing_status, created_at) "
                f"VALUES ({photo_id}, '/enneagram/{filename}', CURRENT_TIMESTAMP, 4000, 2664, 5000000, 'Sony', 'A7R V', 'completed', CURRENT_TIMESTAMP)"
            )
        
        insert_lines.append("SELECT * FROM dual")
        
        sql = '\n'.join(insert_lines)
        remaining_batches.append({
            'batch_num': batch_num,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'count': len(batch_filenames),
            'sql': sql
        })
    
    return remaining_batches

def main():
    """Generate and print all remaining batch SQL."""
    print("📸 Generating All Remaining Photo Import Batches\n")
    
    batches = generate_all_remaining_inserts()
    
    for batch in batches:
        print(f"=== BATCH {batch['batch_num']} ===")
        print(f"Photos: {batch['count']} (indices {batch['start_idx']}-{batch['end_idx']-1})")
        print("SQL:")
        print(batch['sql'])
        print()
    
    # Write a consolidated file
    with open('all_remaining_batches.sql', 'w') as f:
        for batch in batches:
            f.write(f"-- Batch {batch['batch_num']}: {batch['count']} photos\n")
            f.write(batch['sql'])
            f.write(";\n\n")
    
    print(f"✅ Generated {len(batches)} remaining batches")
    print("Copy and execute each SQL block with the Oracle MCP")

if __name__ == "__main__":
    main()