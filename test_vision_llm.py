#!/usr/bin/env python3
"""
Test script for Vision LLM enhanced emotional impact analysis.

Tests the complete integration of:
- VisionLLMAnalyzer emotional impact analysis
- AestheticAnalyzer with unified mood/emotional scoring
- QualityRanker orchestration
- Fallback behavior
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from photosight.analysis.vision_llm_analyzer import VisionLLMAnalyzer
from photosight.analysis.aesthetic_analyzer import AestheticAnalyzer
from photosight.ranking.quality_ranker import QualityRanker
from photosight.processing.analysis_context import AnalysisContext
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisionLLMTester:
    """Test harness for Vision LLM analysis modules."""
    
    def __init__(self, config_path: str = None):
        """Initialize tester with configuration."""
        # Load config
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default test configuration
            self.config = {
                'vision_llm': {
                    'enabled': True,
                    'provider': 'gemini',
                    'gemini': {
                        'api_key': os.getenv('GEMINI_API_KEY', ''),
                        'model': 'gemini-1.5-flash',
                        'temperature': 0.3
                    }
                },
                'ranking': {
                    'weights': {
                        'technical': 0.3,
                        'composition': 0.25,
                        'aesthetic': 0.25,
                        'subject': 0.2
                    }
                }
            }
        
        # Initialize components
        self.vision_analyzer = VisionLLMAnalyzer(self.config)
        self.aesthetic_analyzer = AestheticAnalyzer(self.config)
        self.quality_ranker = QualityRanker(self.config)
        
    def test_emotional_impact_analysis(self, image_path: Path) -> Dict:
        """Test emotional impact analysis on a single image."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {image_path.name}")
        logger.info(f"{'='*60}")
        
        results = {}
        
        # 1. Test Vision LLM emotional impact directly
        logger.info("\n1. Testing Vision LLM Emotional Impact Analysis:")
        try:
            vision_result = self.vision_analyzer.analyze_emotional_impact(image_path)
            parsed = vision_result.get('parsed', {})
            
            if parsed:
                results['vision_llm'] = {
                    'emotion_score': parsed.get('emotion_score', 0),
                    'dominant_emotion': parsed.get('dominant_emotion', 'unknown'),
                    'emotional_elements': parsed.get('emotional_elements', []),
                    'visual_contributors': parsed.get('visual_contributors', [])
                }
                
                logger.info(f"  ✓ Emotion Score: {results['vision_llm']['emotion_score']:.3f}")
                logger.info(f"  ✓ Dominant Emotion: {results['vision_llm']['dominant_emotion']}")
                logger.info(f"  ✓ Emotional Elements: {results['vision_llm']['emotional_elements']}")
            else:
                logger.warning("  ✗ No parsed results from Vision LLM")
                results['vision_llm'] = {'error': 'No parsed results'}
                
        except Exception as e:
            logger.error(f"  ✗ Vision LLM analysis failed: {e}")
            results['vision_llm'] = {'error': str(e)}
        
        # 2. Test Aesthetic Analysis with emotional impact
        logger.info("\n2. Testing Aesthetic Analysis Integration:")
        try:
            # Load image and create context
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)
            
            # Create analysis context
            context = AnalysisContext(image_array, str(image_path))
            
            # Set emotional impact score if we got one from Vision LLM
            if 'vision_llm' in results and 'emotion_score' in results['vision_llm']:
                context.emotional_impact_score = results['vision_llm']['emotion_score']
                logger.info(f"  → Set context.emotional_impact_score: {context.emotional_impact_score:.3f}")
            
            # Run aesthetic analysis
            aesthetic_results = self.aesthetic_analyzer.analyze_aesthetics(context)
            
            results['aesthetic'] = {
                'mood_score': aesthetic_results.get('mood_score', 0),
                'emotional_impact': aesthetic_results.get('emotional_impact', 0),
                'mood_characteristics': aesthetic_results.get('mood_characteristics', {}),
                'overall_appeal': aesthetic_results.get('overall_appeal', 0)
            }
            
            logger.info(f"  ✓ Mood Score: {results['aesthetic']['mood_score']:.3f}")
            logger.info(f"  ✓ Emotional Impact: {results['aesthetic']['emotional_impact']:.3f}")
            logger.info(f"  ✓ Overall Appeal: {results['aesthetic']['overall_appeal']:.3f}")
            
        except Exception as e:
            logger.error(f"  ✗ Aesthetic analysis failed: {e}")
            results['aesthetic'] = {'error': str(e)}
        
        # 3. Test Quality Ranker orchestration
        logger.info("\n3. Testing Quality Ranker Orchestration:")
        try:
            quality_score = self.quality_ranker.rank_photo(image_path)
            results['quality_score'] = quality_score
            logger.info(f"  ✓ Overall Quality Score: {quality_score:.3f}")
            
            # Get detailed analysis
            detailed = self.quality_ranker.get_detailed_analysis(image_path)
            results['detailed_scores'] = detailed.get('component_scores', {})
            
            logger.info(f"  ✓ Technical: {results['detailed_scores'].get('technical', 0):.3f}")
            logger.info(f"  ✓ Composition: {results['detailed_scores'].get('composition', 0):.3f}")
            logger.info(f"  ✓ Aesthetic: {results['detailed_scores'].get('aesthetic', 0):.3f}")
            logger.info(f"  ✓ Subject: {results['detailed_scores'].get('subject', 0):.3f}")
            
        except Exception as e:
            logger.error(f"  ✗ Quality ranking failed: {e}")
            results['quality_score'] = 0
            results['error'] = str(e)
        
        return results
    
    def test_fallback_behavior(self, image_path: Path) -> Dict:
        """Test fallback behavior when Vision LLM is disabled."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing Fallback Behavior: {image_path.name}")
        logger.info(f"{'='*60}")
        
        # Temporarily disable Vision LLM
        original_enabled = self.vision_analyzer.enabled
        self.vision_analyzer.enabled = False
        
        results = {}
        
        try:
            # Load image and create context
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)
            context = AnalysisContext(image_array, str(image_path))
            
            # Run aesthetic analysis without Vision LLM
            aesthetic_results = self.aesthetic_analyzer.analyze_aesthetics(context)
            
            results['fallback'] = {
                'mood_score': aesthetic_results.get('mood_score', 0),
                'emotional_impact': aesthetic_results.get('emotional_impact', 0),
                'mood_characteristics': aesthetic_results.get('mood_characteristics', {})
            }
            
            logger.info(f"  ✓ Fallback Mood Score: {results['fallback']['mood_score']:.3f}")
            logger.info(f"  ✓ Fallback Emotional Impact: {results['fallback']['emotional_impact']:.3f}")
            logger.info(f"  ✓ Mood Characteristics: {list(results['fallback']['mood_characteristics'].keys())}")
            
        except Exception as e:
            logger.error(f"  ✗ Fallback analysis failed: {e}")
            results['error'] = str(e)
        finally:
            # Restore Vision LLM state
            self.vision_analyzer.enabled = original_enabled
        
        return results
    
    def run_comprehensive_test(self, test_images: List[Path]) -> Dict:
        """Run comprehensive tests on multiple images."""
        all_results = {}
        
        for image_path in test_images:
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Test with Vision LLM enabled
            results_enabled = self.test_emotional_impact_analysis(image_path)
            
            # Test fallback behavior
            results_fallback = self.test_fallback_behavior(image_path)
            
            # Compare results
            logger.info(f"\n4. Comparison for {image_path.name}:")
            if 'vision_llm' in results_enabled and 'fallback' in results_fallback:
                vision_score = results_enabled['vision_llm'].get('emotion_score', 0)
                fallback_score = results_fallback['fallback'].get('mood_score', 0)
                difference = abs(vision_score - fallback_score)
                
                logger.info(f"  → Vision LLM Score: {vision_score:.3f}")
                logger.info(f"  → Fallback Score: {fallback_score:.3f}")
                logger.info(f"  → Difference: {difference:.3f}")
                
                if results_enabled['vision_llm'].get('dominant_emotion'):
                    logger.info(f"  → Dominant Emotion: {results_enabled['vision_llm']['dominant_emotion']}")
            
            all_results[str(image_path)] = {
                'with_vision_llm': results_enabled,
                'fallback': results_fallback
            }
        
        return all_results


def main():
    """Main test function."""
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        logger.warning("GEMINI_API_KEY not set. Vision LLM tests may fail.")
        logger.info("Set it with: export GEMINI_API_KEY='your-api-key'")
    
    # Initialize tester
    tester = VisionLLMTester()
    
    # Find test images
    test_dir = Path("test_photos")
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
    else:
        # Try to find some sample images in the project
        logger.info("Looking for sample images...")
        test_images = []
        for pattern in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            test_images.extend(Path(".").glob(f"**/{pattern}"))
        test_images = test_images[:5]  # Limit to 5 images for testing
    
    if not test_images:
        logger.error("No test images found. Please add some images to test_photos/ directory.")
        return
    
    logger.info(f"Found {len(test_images)} test images")
    
    # Run tests
    results = tester.run_comprehensive_test(test_images)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    successful_tests = sum(1 for r in results.values() if 'error' not in r.get('with_vision_llm', {}))
    logger.info(f"✓ Successful Vision LLM tests: {successful_tests}/{len(results)}")
    
    # Save results
    results_file = Path("vision_llm_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()