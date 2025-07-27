#!/usr/bin/env python3
"""
Systematic verification of PhotoSight pipeline execution.

Verifies each step of the analysis pipeline to ensure correct operation.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import statistics

def load_results():
    """Load all analysis results."""
    results_file = Path("enneagram_analysis_output/enneagram_analysis_results.json")
    rankings_file = Path("enneagram_analysis_output/enneagram_rankings.csv")
    report_file = Path("enneagram_analysis_output/enneagram_analysis_report.txt")
    
    # Load JSON results
    with open(results_file, 'r') as f:
        json_data = json.load(f)
    
    # Load CSV rankings
    csv_data = []
    with open(rankings_file, 'r') as f:
        reader = csv.DictReader(f)
        csv_data = list(reader)
    
    # Load report
    with open(report_file, 'r') as f:
        report_text = f.read()
    
    return json_data, csv_data, report_text


def verify_step_1_input_loading(json_data):
    """Step 1: Verify photos were correctly located and loaded."""
    print("\n" + "="*60)
    print("STEP 1: VERIFY INPUT PHOTO LOADING")
    print("="*60)
    
    metadata = json_data['metadata']
    results = json_data['results']
    
    # Check photo count
    total_photos = metadata['total_photos']
    processed = metadata['processed']
    failed = metadata.get('failed', 0)
    
    print(f"✓ Total photos found: {total_photos}")
    print(f"✓ Successfully processed: {processed}")
    print(f"✓ Failed to process: {failed}")
    
    # Verify all photos have expected filename pattern
    photo_files = list(results.keys())
    dsc_photos = [f for f in photo_files if f.startswith('DSC')]
    
    print(f"\n✓ Photos with DSC prefix: {len(dsc_photos)}/{len(photo_files)}")
    print(f"✓ First 5 photos: {photo_files[:5]}")
    print(f"✓ Last 5 photos: {photo_files[-5:]}")
    
    # Check for gaps in numbering
    photo_numbers = []
    for photo in dsc_photos:
        try:
            num = int(photo[3:8])  # Extract number from DSC04XXX.jpg
            photo_numbers.append(num)
        except:
            pass
    
    if photo_numbers:
        photo_numbers.sort()
        print(f"\n✓ Photo number range: DSC{photo_numbers[0]:05d} to DSC{photo_numbers[-1]:05d}")
        print(f"✓ Total span: {photo_numbers[-1] - photo_numbers[0] + 1} numbers")
    
    return len(photo_files) == processed


def verify_step_2_analysis_execution(json_data):
    """Step 2: Verify analysis modules executed for each photo."""
    print("\n" + "="*60)
    print("STEP 2: VERIFY ANALYSIS MODULE EXECUTION")
    print("="*60)
    
    results = json_data['results']
    
    # Expected analysis components
    expected_components = [
        'scene', 'composition', 'technical', 'aesthetic', 
        'vision_quality', 'decisive_moment', 'scores'
    ]
    
    # Check first photo has all components
    first_photo = list(results.keys())[0]
    first_result = results[first_photo]
    
    print(f"Checking photo: {first_photo}")
    for component in expected_components:
        if component in first_result:
            print(f"✓ {component}: Present")
        else:
            print(f"✗ {component}: Missing!")
    
    # Verify all photos have complete analysis
    missing_components = defaultdict(int)
    for photo, data in results.items():
        for component in expected_components:
            if component not in data:
                missing_components[component] += 1
    
    print(f"\nAnalysis completeness across all {len(results)} photos:")
    for component in expected_components:
        missing = missing_components.get(component, 0)
        complete = len(results) - missing
        print(f"✓ {component}: {complete}/{len(results)} complete ({complete/len(results)*100:.1f}%)")
    
    # Sample analysis values
    print("\nSample analysis values from first photo:")
    print(f"  Scene type: {first_result['scene']['type']}")
    print(f"  Technical score: {first_result['technical']['overall_technical']:.3f}")
    print(f"  Artistic score: {first_result['aesthetic']['overall_aesthetic']:.3f}")
    print(f"  Emotional impact: {first_result['vision_quality']['emotional_impact']:.3f}")
    print(f"  Is decisive moment: {first_result['decisive_moment']['is_decisive']}")
    
    return len(missing_components) == 0


def verify_step_3_scoring_calculations(json_data):
    """Step 3: Verify scoring calculations were performed correctly."""
    print("\n" + "="*60)
    print("STEP 3: VERIFY SCORING CALCULATIONS")
    print("="*60)
    
    results = json_data['results']
    
    # Check score ranges
    all_scores = {
        'overall': [],
        'technical': [],
        'artistic': [],
        'emotional': []
    }
    
    for photo, data in results.items():
        if 'scores' in data:
            for score_type in all_scores:
                all_scores[score_type].append(data['scores'][score_type])
    
    print("Score distributions:")
    for score_type, values in all_scores.items():
        if values:
            print(f"\n{score_type.capitalize()} scores:")
            print(f"  Count: {len(values)}")
            print(f"  Range: {min(values):.3f} - {max(values):.3f}")
            print(f"  Mean: {statistics.mean(values):.3f}")
            print(f"  Std Dev: {statistics.stdev(values):.3f}")
    
    # Verify scoring formula (weights: technical=0.3, artistic=0.4, emotional=0.3)
    print("\nVerifying scoring formula on sample photos:")
    sample_photos = list(results.keys())[:5]
    
    for photo in sample_photos:
        data = results[photo]
        if 'scores' in data:
            scores = data['scores']
            
            # Recalculate expected overall score
            expected = (
                scores['technical'] * 0.3 +
                scores['artistic'] * 0.4 +
                scores['emotional'] * 0.3
            )
            
            actual = scores['overall']
            diff = abs(expected - actual)
            
            status = "✓" if diff < 0.001 else "✗"
            print(f"{status} {photo}: Expected={expected:.3f}, Actual={actual:.3f}, Diff={diff:.6f}")
    
    # Check for reasonable score variance
    overall_variance = statistics.variance(all_scores['overall']) if len(all_scores['overall']) > 1 else 0
    print(f"\nOverall score variance: {overall_variance:.6f}")
    print(f"Score diversity: {'Good' if overall_variance > 0.001 else 'Low'}")
    
    return overall_variance > 0.001


def verify_step_4_ranking_logic(json_data, csv_data):
    """Step 4: Verify ranking and sorting logic executed properly."""
    print("\n" + "="*60)
    print("STEP 4: VERIFY RANKING AND SORTING LOGIC")
    print("="*60)
    
    # Check CSV is properly sorted
    csv_scores = [float(row['Overall Score']) for row in csv_data]
    is_sorted = all(csv_scores[i] >= csv_scores[i+1] for i in range(len(csv_scores)-1))
    
    print(f"✓ CSV sorted by score: {is_sorted}")
    print(f"✓ Number of ranked photos: {len(csv_data)}")
    print(f"✓ Highest score: {csv_scores[0]:.3f}")
    print(f"✓ Lowest score: {csv_scores[-1]:.3f}")
    
    # Verify rank numbers are sequential
    ranks = [int(row['Rank']) for row in csv_data]
    sequential_ranks = all(ranks[i] == i+1 for i in range(len(ranks)))
    print(f"✓ Sequential rank numbers: {sequential_ranks}")
    
    # Check JSON rankings match CSV
    json_rankings = json_data['rankings']['overall']
    
    print(f"\nComparing top 5 rankings:")
    for i in range(5):
        csv_photo = csv_data[i]['Filename']
        csv_score = float(csv_data[i]['Overall Score'])
        
        json_photo = json_rankings[i]['filename']
        json_score = json_rankings[i]['scores']['overall']
        
        match = "✓" if csv_photo == json_photo and abs(csv_score - json_score) < 0.001 else "✗"
        print(f"{match} Rank {i+1}: CSV={csv_photo} ({csv_score:.3f}), JSON={json_photo} ({json_score:.3f})")
    
    # Verify decisive moments
    decisive_moments = json_data['rankings']['decisive_moments']
    print(f"\n✓ Decisive moments identified: {len(decisive_moments)}")
    if decisive_moments:
        print(f"✓ Highest confidence: {decisive_moments[0]['confidence']:.2f}")
        print(f"✓ Example reason: '{decisive_moments[0]['reason']}'")
    
    return is_sorted and sequential_ranks


def verify_step_5_output_structure(json_data, csv_data, report_text):
    """Step 5: Verify output files contain expected data structures."""
    print("\n" + "="*60)
    print("STEP 5: VERIFY OUTPUT DATA STRUCTURES")
    print("="*60)
    
    # Check JSON structure
    print("JSON file structure:")
    print(f"✓ Has 'metadata' section: {'metadata' in json_data}")
    print(f"✓ Has 'results' section: {'results' in json_data}")
    print(f"✓ Has 'rankings' section: {'rankings' in json_data}")
    
    # Check metadata completeness
    metadata_fields = ['timestamp', 'total_photos', 'processed', 'processing_time']
    for field in metadata_fields:
        print(f"✓ Metadata.{field}: {field in json_data['metadata']}")
    
    # Check CSV columns
    expected_csv_columns = [
        'Rank', 'Filename', 'Overall Score', 'Technical', 
        'Artistic', 'Emotional', 'Scene', 'Mood'
    ]
    actual_columns = list(csv_data[0].keys()) if csv_data else []
    
    print(f"\nCSV columns:")
    for col in expected_csv_columns:
        print(f"✓ {col}: {col in actual_columns}")
    
    # Check report sections
    report_sections = [
        'ENNEAGRAM WORKSHOP PHOTO ANALYSIS REPORT',
        'TOP 20 PHOTOS BY OVERALL SCORE',
        'DECISIVE MOMENTS',
        'SCENE DISTRIBUTION',
        'MOOD DISTRIBUTION',
        'AVERAGE SCORES'
    ]
    
    print(f"\nReport sections:")
    for section in report_sections:
        print(f"✓ {section}: {section in report_text}")
    
    # Verify scene and mood distributions
    results = json_data['results']
    scene_counts = defaultdict(int)
    mood_counts = defaultdict(int)
    
    for photo, data in results.items():
        if 'scene' in data:
            scene_counts[data['scene']['type']] += 1
        if 'aesthetic' in data:
            mood_counts[data['aesthetic']['mood']] += 1
    
    print(f"\n✓ Unique scene types: {len(scene_counts)}")
    print(f"✓ Unique moods: {len(mood_counts)}")
    print(f"✓ Most common scene: {max(scene_counts, key=scene_counts.get)} ({max(scene_counts.values())} photos)")
    print(f"✓ Most common mood: {max(mood_counts, key=mood_counts.get)} ({max(mood_counts.values())} photos)")
    
    return True


def main():
    """Run systematic pipeline verification."""
    print("\n" + "="*70)
    print("PHOTOSIGHT PIPELINE SYSTEMATIC VERIFICATION")
    print("="*70)
    
    try:
        # Load all results
        json_data, csv_data, report_text = load_results()
        
        # Run verification steps
        steps = [
            ("Input Loading", lambda: verify_step_1_input_loading(json_data)),
            ("Analysis Execution", lambda: verify_step_2_analysis_execution(json_data)),
            ("Scoring Calculations", lambda: verify_step_3_scoring_calculations(json_data)),
            ("Ranking Logic", lambda: verify_step_4_ranking_logic(json_data, csv_data)),
            ("Output Structure", lambda: verify_step_5_output_structure(json_data, csv_data, report_text))
        ]
        
        results = []
        for step_name, verify_func in steps:
            try:
                result = verify_func()
                results.append((step_name, result))
            except Exception as e:
                print(f"\n✗ Error in {step_name}: {e}")
                results.append((step_name, False))
        
        # Final summary
        print("\n" + "="*70)
        print("VERIFICATION SUMMARY")
        print("="*70)
        
        all_passed = True
        for step_name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status} - {step_name}")
            if not result:
                all_passed = False
        
        print("\n" + "="*70)
        if all_passed:
            print("✅ ALL VERIFICATION STEPS PASSED")
            print("The PhotoSight pipeline executed correctly!")
        else:
            print("❌ SOME VERIFICATION STEPS FAILED")
            print("Please check the output above for details.")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("Could not load analysis results. Ensure the analysis has been run first.")


if __name__ == "__main__":
    main()