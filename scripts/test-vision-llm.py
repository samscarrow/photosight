#!/usr/bin/env python3
"""
Test script for Vision LLM integration in PhotoSight.

Tests the vision-capable LLM providers with sample images.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from photosight.analysis.vision_llm_analyzer import VisionLLMAnalyzer
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_vision_analyzer():
    """Test the vision LLM analyzer with a sample image."""
    
    # Configuration for testing
    config = {
        'vision_llm': {
            'enabled': True,
            'provider': 'gemini',
            'gemini': {
                'model': 'gemini-pro-vision',
                'api_key': os.environ.get('GEMINI_API_KEY'),
                'temperature': 0.4,
                'safety_settings': 'medium'
            }
        }
    }
    
    # Initialize analyzer
    try:
        analyzer = VisionLLMAnalyzer(config)
        print("✅ Vision LLM Analyzer initialized successfully")
        print(f"   Provider: {config['vision_llm']['provider']}")
        print(f"   Enabled: {analyzer.enabled}")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    # Test with a sample image
    # For testing, we'll create a simple test image
    test_image_path = Path("/tmp/photosight_vision_test.jpg")
    
    if not test_image_path.exists():
        print("⚠️  No test image found. Creating a sample image...")
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a test image
        img = Image.new('RGB', (800, 600), color='skyblue')
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes
        draw.rectangle([100, 100, 300, 300], fill='green', outline='darkgreen', width=3)
        draw.ellipse([400, 200, 600, 400], fill='yellow', outline='orange', width=3)
        
        # Add text
        try:
            # Try to use a nice font, fall back to default if not available
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except:
            font = ImageFont.load_default()
        
        draw.text((200, 50), "PhotoSight Vision Test", fill='white', font=font)
        draw.text((250, 500), "Sample Scene", fill='black', font=font)
        
        # Save test image
        img.save(test_image_path, 'JPEG')
        print(f"✅ Created test image: {test_image_path}")
    
    # Test different analysis types
    print("\n🔍 Testing Vision Analysis Functions...\n")
    
    # 1. Scene Classification
    print("1. Scene Classification:")
    scene_result = analyzer.analyze_scene(test_image_path)
    if 'error' not in scene_result:
        print("   ✅ Scene analysis successful")
        if 'parsed' in scene_result:
            print(f"   Scene type: {scene_result['parsed'].get('scene_type', 'N/A')}")
            print(f"   Mood: {scene_result['parsed'].get('mood', 'N/A')}")
        print(f"   Raw response preview: {scene_result.get('content', '')[:200]}...")
    else:
        print(f"   ❌ Scene analysis failed: {scene_result['error']}")
    
    # 2. Composition Analysis
    print("\n2. Composition Analysis:")
    comp_result = analyzer.analyze_composition(test_image_path)
    if 'error' not in comp_result:
        print("   ✅ Composition analysis successful")
        if 'parsed' in comp_result:
            print(f"   Focal point: {comp_result['parsed'].get('focal_point', 'N/A')}")
            print(f"   Balance: {comp_result['parsed'].get('balance', 'N/A')}")
    else:
        print(f"   ❌ Composition analysis failed: {comp_result['error']}")
    
    # 3. Quality Assessment
    print("\n3. Quality Assessment:")
    quality_result = analyzer.assess_quality(test_image_path)
    if 'error' not in quality_result:
        print("   ✅ Quality assessment successful")
        if 'parsed' in quality_result:
            print(f"   Rating: {quality_result['parsed'].get('rating', 'N/A')}/10")
            print(f"   Technical quality: {quality_result['parsed'].get('technical_quality', 'N/A')}")
    else:
        print(f"   ❌ Quality assessment failed: {quality_result['error']}")
    
    # 4. Crop Suggestions
    print("\n4. Crop Suggestions:")
    crop_result = analyzer.suggest_crops(test_image_path)
    if 'error' not in crop_result:
        print("   ✅ Crop suggestion successful")
        if 'parsed' in crop_result:
            crops = crop_result['parsed'].get('crop_options', [])
            print(f"   Number of crop suggestions: {len(crops) if isinstance(crops, list) else 'N/A'}")
    else:
        print(f"   ❌ Crop suggestion failed: {crop_result['error']}")
    
    # Print usage statistics if available
    print("\n📊 Usage Statistics:")
    total_tokens = 0
    for result in [scene_result, comp_result, quality_result, crop_result]:
        if 'usage' in result:
            usage = result['usage']
            total_tokens += usage.get('total_tokens', 0)
            print(f"   {result.get('provider', 'Unknown')}: {usage.get('total_tokens', 0)} tokens")
    
    print(f"\n   Total tokens used: {total_tokens}")
    
    # Test caching
    print("\n🔄 Testing Cache:")
    print("   Re-analyzing scene (should use cache)...")
    cached_result = analyzer.analyze_scene(test_image_path)
    print(f"   Cache hit: {'Yes' if cached_result == scene_result else 'No'}")
    
    # Clear cache
    analyzer.clear_cache()
    print("   ✅ Cache cleared")
    
    print("\n✅ Vision LLM testing complete!")
    
    # Optionally save results
    results_file = Path("/tmp/photosight_vision_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'scene': scene_result,
            'composition': comp_result,
            'quality': quality_result,
            'crop': crop_result
        }, f, indent=2)
    print(f"\n💾 Full results saved to: {results_file}")


def main():
    """Main test execution."""
    print("🚀 PhotoSight Vision LLM Integration Test\n")
    
    # Check for API key
    if not os.environ.get('GEMINI_API_KEY'):
        print("⚠️  GEMINI_API_KEY not set in environment")
        print("💡 Set it with: export GEMINI_API_KEY='your-api-key'")
        print("💡 Or use: op read \"op://Personal/Google AI API/credential\"")
        return
    
    try:
        test_vision_analyzer()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Test failed with error: {e}")


if __name__ == "__main__":
    main()