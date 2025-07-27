#!/usr/bin/env python3
"""
Display the top-ranked enneagram photos with their analysis details.
"""

import json
from pathlib import Path
from datetime import datetime

def load_rankings():
    """Load the analysis results."""
    results_file = Path("enneagram_analysis_output/enneagram_analysis_results.json")
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def display_top_photos(data, count=20):
    """Display top photos with detailed information."""
    rankings = data['rankings']['overall']
    results = data['results']
    
    print("\n" + "="*80)
    print(f"TOP {count} ENNEAGRAM WORKSHOP PHOTOS")
    print("="*80)
    
    for i, entry in enumerate(rankings[:count], 1):
        filename = entry['filename']
        scores = entry['scores']
        full_data = results[filename]
        
        print(f"\n{'─'*80}")
        print(f"RANK #{i}: {filename}")
        print(f"{'─'*80}")
        
        # Overall scores
        print(f"📊 SCORES:")
        print(f"   Overall:    {scores['overall']:.3f} ⭐")
        print(f"   Technical:  {scores['technical']:.3f}")
        print(f"   Artistic:   {scores['artistic']:.3f}")
        print(f"   Emotional:  {scores['emotional']:.3f}")
        
        # Scene and mood
        print(f"\n🎬 SCENE ANALYSIS:")
        print(f"   Type: {full_data['scene']['type']}")
        print(f"   Confidence: {full_data['scene']['confidence']:.2f}")
        print(f"   Mood: {full_data['aesthetic']['mood']}")
        
        # Composition details
        print(f"\n📐 COMPOSITION:")
        print(f"   Balance: {full_data['composition']['balance_score']:.2f}")
        print(f"   Rule of Thirds: {full_data['composition']['rule_of_thirds']:.2f}")
        print(f"   Leading Lines: {'Yes' if full_data['composition']['leading_lines'] else 'No'}")
        
        # Technical details
        print(f"\n🔧 TECHNICAL QUALITY:")
        print(f"   Sharpness: {full_data['technical']['sharpness']:.2f}")
        print(f"   Exposure: {full_data['technical']['exposure']:.2f}")
        print(f"   Noise Level: {full_data['technical']['noise']:.2f}")
        
        # Aesthetic details
        print(f"\n🎨 AESTHETIC QUALITY:")
        print(f"   Color Harmony: {full_data['aesthetic']['color_harmony']:.2f}")
        print(f"   Visual Appeal: {full_data['aesthetic']['visual_appeal']:.2f}")
        
        # Decisive moment
        if full_data['decisive_moment']['is_decisive']:
            print(f"\n⚡ DECISIVE MOMENT:")
            print(f"   Confidence: {full_data['decisive_moment']['confidence']:.2f}")
            print(f"   Reason: {full_data['decisive_moment']['reason']}")
    
    # Show decisive moments summary
    print(f"\n{'='*80}")
    print("DECISIVE MOMENTS SUMMARY")
    print("="*80)
    
    decisive_moments = data['rankings']['decisive_moments']
    print(f"\nTotal Decisive Moments: {len(decisive_moments)}")
    
    if decisive_moments:
        print("\nTop 10 Decisive Moments:")
        for i, moment in enumerate(decisive_moments[:10], 1):
            print(f"{i:2d}. {moment['filename']} - Confidence: {moment['confidence']:.2f}")

def show_photo_groups(data):
    """Show photos grouped by scene type and mood."""
    results = data['results']
    
    # Group by scene
    scene_groups = {}
    mood_groups = {}
    
    for filename, photo_data in results.items():
        scene = photo_data['scene']['type']
        mood = photo_data['aesthetic']['mood']
        score = photo_data['scores']['overall']
        
        if scene not in scene_groups:
            scene_groups[scene] = []
        scene_groups[scene].append((filename, score))
        
        if mood not in mood_groups:
            mood_groups[mood] = []
        mood_groups[mood].append((filename, score))
    
    # Sort each group by score
    for group in scene_groups.values():
        group.sort(key=lambda x: x[1], reverse=True)
    for group in mood_groups.values():
        group.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*80}")
    print("TOP PHOTOS BY SCENE TYPE")
    print("="*80)
    
    for scene, photos in sorted(scene_groups.items()):
        print(f"\n{scene.upper()} (Top 5):")
        for filename, score in photos[:5]:
            print(f"  - {filename}: {score:.3f}")
    
    print(f"\n{'='*80}")
    print("TOP PHOTOS BY MOOD")
    print("="*80)
    
    for mood, photos in sorted(mood_groups.items()):
        print(f"\n{mood.upper()} (Top 5):")
        for filename, score in photos[:5]:
            print(f"  - {filename}: {score:.3f}")

def create_selection_report(data, output_file="enneagram_selection.txt"):
    """Create a detailed selection report."""
    with open(output_file, 'w') as f:
        f.write("PHOTOSIGHT ENNEAGRAM WORKSHOP - PHOTO SELECTION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Top 30 for portfolio
        f.write("TOP 30 PHOTOS FOR PORTFOLIO\n")
        f.write("-" * 60 + "\n\n")
        
        rankings = data['rankings']['overall']
        for i, entry in enumerate(rankings[:30], 1):
            filename = entry['filename']
            scores = entry['scores']
            scene = entry['scene']
            mood = entry['mood']
            
            f.write(f"{i:2d}. {filename}\n")
            f.write(f"    Score: {scores['overall']:.3f} ")
            f.write(f"(T:{scores['technical']:.2f} A:{scores['artistic']:.2f} E:{scores['emotional']:.2f})\n")
            f.write(f"    Scene: {scene} | Mood: {mood}\n\n")
        
        # Decisive moments
        f.write("\nDECISIVE MOMENTS (ALL)\n")
        f.write("-" * 60 + "\n\n")
        
        decisive_moments = data['rankings']['decisive_moments']
        for i, moment in enumerate(decisive_moments, 1):
            f.write(f"{i:2d}. {moment['filename']} (Confidence: {moment['confidence']:.2f})\n")
        
        # Category winners
        f.write("\n\nCATEGORY WINNERS\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("HIGHEST TECHNICAL QUALITY:\n")
        tech_sorted = sorted(rankings, key=lambda x: x['scores']['technical'], reverse=True)
        for entry in tech_sorted[:5]:
            f.write(f"  - {entry['filename']}: {entry['scores']['technical']:.3f}\n")
        
        f.write("\nMOST ARTISTIC:\n")
        art_sorted = sorted(rankings, key=lambda x: x['scores']['artistic'], reverse=True)
        for entry in art_sorted[:5]:
            f.write(f"  - {entry['filename']}: {entry['scores']['artistic']:.3f}\n")
        
        f.write("\nSTRONGEST EMOTIONAL IMPACT:\n")
        emo_sorted = sorted(rankings, key=lambda x: x['scores']['emotional'], reverse=True)
        for entry in emo_sorted[:5]:
            f.write(f"  - {entry['filename']}: {entry['scores']['emotional']:.3f}\n")
    
    print(f"\n✅ Selection report saved to: {output_file}")

def main():
    """Show the selected photos."""
    # Load rankings
    data = load_rankings()
    
    # Display top 20 photos with details
    display_top_photos(data, count=20)
    
    # Show grouped views
    show_photo_groups(data)
    
    # Create selection report
    create_selection_report(data)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SELECTION SUMMARY")
    print("="*80)
    
    metadata = data['metadata']
    rankings = data['rankings']['overall']
    
    print(f"\nTotal Photos Analyzed: {metadata['total_photos']}")
    print(f"Processing Time: {metadata['processing_time']:.1f} seconds")
    
    # Score thresholds
    excellent = len([r for r in rankings if r['scores']['overall'] >= 0.8])
    very_good = len([r for r in rankings if 0.75 <= r['scores']['overall'] < 0.8])
    good = len([r for r in rankings if 0.7 <= r['scores']['overall'] < 0.75])
    
    print(f"\nQuality Distribution:")
    print(f"  Excellent (≥0.800): {excellent} photos")
    print(f"  Very Good (0.750-0.799): {very_good} photos")
    print(f"  Good (0.700-0.749): {good} photos")
    print(f"  Total High Quality: {excellent + very_good + good} photos")
    
    print(f"\nRecommended for portfolio: Top {min(30, len(rankings))} photos")
    print(f"Decisive moments to highlight: {len(data['rankings']['decisive_moments'])} photos")

if __name__ == "__main__":
    main()