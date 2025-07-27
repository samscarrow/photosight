#!/usr/bin/env python3
"""
Simple test of Vision LLM emotional impact analysis.
"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from photosight.analysis.vision_llm_analyzer import VisionLLMAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_single_image():
    """Test emotional impact analysis on a single image."""
    
    # Check API key
    if not os.getenv('GEMINI_API_KEY'):
        logger.error("GEMINI_API_KEY not set!")
        logger.info("Please run: export GEMINI_API_KEY='your-api-key'")
        return
    
    # Configuration
    config = {
        'vision_llm': {
            'enabled': True,
            'provider': 'gemini',
            'gemini': {
                'api_key': os.getenv('GEMINI_API_KEY'),
                'model': 'gemini-1.5-flash',
                'temperature': 0.3
            }
        }
    }
    
    # Initialize analyzer
    analyzer = VisionLLMAnalyzer(config)
    
    # Find a test image
    test_image = Path("test_images/test_001.jpg")
    if not test_image.exists():
        # Try other locations
        for path in ["test_output/accepted/photo3.jpeg", "demo_mask_types.png"]:
            test_image = Path(path)
            if test_image.exists():
                break
        else:
            logger.error("No test images found!")
            return
    
    logger.info(f"Testing image: {test_image}")
    logger.info("-" * 60)
    
    # Test emotional impact analysis
    result = analyzer.analyze_emotional_impact(test_image)
    
    if 'error' in result:
        logger.error(f"Analysis failed: {result['error']}")
        return
    
    # Display results
    parsed = result.get('parsed', {})
    if parsed:
        logger.info(f"✓ Emotion Score: {parsed.get('emotion_score', 'N/A')}")
        logger.info(f"✓ Dominant Emotion: {parsed.get('dominant_emotion', 'N/A')}")
        logger.info(f"✓ Emotional Elements: {parsed.get('emotional_elements', 'N/A')}")
        logger.info(f"✓ Visual Contributors: {parsed.get('visual_contributors', 'N/A')}")
        logger.info(f"✓ Composition Impact: {parsed.get('composition_impact', 'N/A')}")
    else:
        logger.warning("No parsed results returned")
        if 'content' in result:
            logger.info(f"Raw response: {result['content'][:200]}...")

if __name__ == "__main__":
    test_single_image()