#!/usr/bin/env python3
"""
Test Vision LLM on Enneagram workshop photos.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from photosight.analysis.vision_llm_analyzer import VisionLLMAnalyzer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_enneagram_photos():
    """Test vision LLM on enneagram workshop photos."""
    # Initialize provider and analyzer
    print("Initializing Gemini Vision analyzer...")
    
    # Create config matching PhotoSight's structure
    config = {
        'vision_llm': {
            'enabled': True,
            'provider': 'gemini',
            'gemini': {
                'model': 'gemini-1.5-flash',
                'temperature': 0.4,
                'max_output_tokens': 2048,
                'safety_settings': 'medium'
            },
            'processing': {
                'use_previews': True,
                'preview_size': 1024,
                'cache_results': True,
                'timeout': 30
            },
            'privacy': {
                'strip_exif': True,
                'blur_faces_before_analysis': False  # For workshop photos, faces are important
            }
        }
    }
    
    analyzer = VisionLLMAnalyzer(config)
    
    # Get sample photos
    enneagram_dir = Path("/Users/sam/Desktop/photosight_output/enneagram_workshop/accepted")
    photos = list(enneagram_dir.glob("*.jpg"))[:1]  # Test first photo only
    
    if not photos:
        print("No photos found!")
        return
    
    print(f"\nFound {len(photos)} photos to analyze")
    
    # Test each photo
    for photo_path in photos:
        print(f"\n{'='*60}")
        print(f"Analyzing: {photo_path.name}")
        print(f"{'='*60}")
        
        try:
            # Scene analysis
            print("\n🎨 Scene Analysis:")
            scene_result = analyzer.analyze_scene(str(photo_path))
            print(f"  Scene Type: {scene_result.get('scene_type', 'Unknown')}")
            print(f"  Environment: {scene_result.get('environment', 'Unknown')}")
            print(f"  Time of Day: {scene_result.get('time_of_day', 'Unknown')}")
            print(f"  Confidence: {scene_result.get('confidence', 0):.2f}")
            
            # Composition analysis
            print("\n📐 Composition Analysis:")
            comp_result = analyzer.analyze_composition(str(photo_path))
            print(f"  Balance: {comp_result.get('balance_score', 0):.2f}")
            print(f"  Leading Lines: {comp_result.get('leading_lines', False)}")
            print(f"  Depth: {comp_result.get('depth_perception', 'Unknown')}")
            print(f"  Visual Weight: {comp_result.get('visual_weight_distribution', 'Unknown')}")
            
            # Quality assessment
            print("\n⭐ Quality Assessment:")
            quality_result = analyzer.assess_quality(str(photo_path))
            print(f"  Technical Score: {quality_result.get('technical_score', 0):.2f}")
            print(f"  Artistic Score: {quality_result.get('artistic_score', 0):.2f}")
            print(f"  Emotional Impact: {quality_result.get('emotional_impact', 0):.2f}")
            print(f"  Overall Score: {quality_result.get('overall_score', 0):.2f}")
            
            # Decisive moment detection
            print("\n⚡ Decisive Moment:")
            moment_result = analyzer.detect_moment(str(photo_path))
            print(f"  Is Decisive: {moment_result.get('is_decisive_moment', False)}")
            print(f"  Reason: {moment_result.get('reason', 'N/A')}")
            print(f"  Confidence: {moment_result.get('confidence', 0):.2f}")
            
            # Suggest intelligent crops
            print("\n✂️ Crop Suggestions:")
            crop_result = analyzer.suggest_crops(str(photo_path))
            if crop_result.get('suggestions'):
                for i, crop in enumerate(crop_result['suggestions'][:2]):
                    print(f"  Crop {i+1}: {crop.get('description', 'N/A')}")
            else:
                print("  No crop suggestions available")
            
        except Exception as e:
            print(f"❌ Error analyzing {photo_path.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_enneagram_photos()