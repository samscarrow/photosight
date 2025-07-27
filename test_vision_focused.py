#!/usr/bin/env python3
"""
Focused test of Vision LLM without complex dependencies.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_vision_llm_complete():
    """Test the complete vision LLM pipeline with minimal dependencies."""
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("GEMINI_API_KEY not set!")
        return
    
    # Configuration
    config = {
        'vision_llm': {
            'enabled': True,
            'provider': 'gemini',
            'gemini': {
                'api_key': api_key,
                'model': 'gemini-1.5-flash',
                'temperature': 0.3
            }
        },
        'aesthetic_analysis': {},
        'ranking': {
            'weights': {
                'technical': 0.3,
                'composition': 0.25,
                'aesthetic': 0.25,
                'subject': 0.2
            }
        }
    }
    
    # Find test images
    test_images = []
    for pattern in ["test_images/*.jpg", "test_output/accepted/*.jpeg"]:
        test_images.extend(Path(".").glob(pattern))
    
    if not test_images:
        logger.error("No test images found!")
        return
    
    logger.info(f"Found {len(test_images)} test images")
    
    # Test each component
    for image_path in test_images[:3]:  # Test first 3 images
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {image_path.name}")
        logger.info(f"{'='*60}")
        
        # 1. Test Vision LLM directly
        test_vision_llm_direct(config, image_path)
        
        # 2. Test integrated flow
        test_integrated_flow(config, image_path)
        
        # 3. Test fallback behavior
        test_fallback_behavior(config, image_path)

def test_vision_llm_direct(config, image_path):
    """Test Vision LLM emotional impact analysis directly."""
    logger.info("\n1. Direct Vision LLM Test:")
    
    try:
        # Import only what we need
        from photosight.analysis.vision_llm_analyzer import VisionLLMAnalyzer
        
        analyzer = VisionLLMAnalyzer(config)
        result = analyzer.analyze_emotional_impact(image_path)
        
        if 'error' in result:
            logger.error(f"  ✗ Error: {result['error']}")
            return
        
        parsed = result.get('parsed', {})
        if parsed:
            logger.info(f"  ✓ Emotion Score: {parsed.get('emotion_score', 'N/A')}")
            logger.info(f"  ✓ Dominant Emotion: {parsed.get('dominant_emotion', 'N/A')}")
            logger.info(f"  ✓ Emotional Elements: {parsed.get('emotional_elements', [])[:3]}...")
            
            # Also test other methods
            scene_result = analyzer.analyze_scene(image_path)
            if 'parsed' in scene_result:
                logger.info(f"  ✓ Scene Type: {scene_result['parsed'].get('scene_type', 'N/A')}")
                logger.info(f"  ✓ Mood: {scene_result['parsed'].get('mood', 'N/A')}")
        else:
            logger.warning("  ✗ No parsed results")
            
    except Exception as e:
        logger.error(f"  ✗ Vision LLM test failed: {e}")

def test_integrated_flow(config, image_path):
    """Test the integrated flow with minimal dependencies."""
    logger.info("\n2. Integrated Flow Test:")
    
    try:
        # Create a minimal analysis context
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image, dtype=np.uint8)
        
        # Manual context creation to avoid complex imports
        class SimpleContext:
            def __init__(self, image_array, path):
                self.standard_image = image_array.astype(np.float32) / 255.0
                self.uint8_image = image_array
                self.photo_path = str(path)
                self.emotional_impact_score = None
                self.width = image_array.shape[1]
                self.height = image_array.shape[0]
                self.channels = image_array.shape[2] if len(image_array.shape) > 2 else 1
                
            @property
            def hsv_image(self):
                import cv2
                return cv2.cvtColor(self.uint8_image, cv2.COLOR_RGB2HSV)
                
            @property
            def grayscale_uint8(self):
                import cv2
                return cv2.cvtColor(self.uint8_image, cv2.COLOR_RGB2GRAY)
        
        context = SimpleContext(image_array, image_path)
        
        # Test Vision LLM -> Context flow
        from photosight.analysis.vision_llm_analyzer import VisionLLMAnalyzer
        vision_analyzer = VisionLLMAnalyzer(config)
        
        vision_result = vision_analyzer.analyze_emotional_impact(image_path)
        if 'parsed' in vision_result:
            emotion_score = vision_result['parsed'].get('emotion_score', 0.5)
            context.emotional_impact_score = emotion_score
            logger.info(f"  ✓ Set emotional impact in context: {emotion_score}")
        
        # Test Aesthetic Analyzer using context
        from photosight.analysis.aesthetic_analyzer import AestheticAnalyzer
        aesthetic_analyzer = AestheticAnalyzer(config)
        
        aesthetic_results = aesthetic_analyzer.analyze_aesthetics(context)
        logger.info(f"  ✓ Mood Score: {aesthetic_results.get('mood_score', 'N/A')}")
        logger.info(f"  ✓ Emotional Impact: {aesthetic_results.get('emotional_impact', 'N/A')}")
        logger.info(f"  ✓ Overall Appeal: {aesthetic_results.get('overall_appeal', 'N/A')}")
        
    except Exception as e:
        logger.error(f"  ✗ Integrated flow test failed: {e}")
        import traceback
        traceback.print_exc()

def test_fallback_behavior(config, image_path):
    """Test fallback behavior when Vision LLM is disabled."""
    logger.info("\n3. Fallback Behavior Test:")
    
    try:
        # Create context without Vision LLM
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image, dtype=np.uint8)
        
        # Import AnalysisContext carefully
        from photosight.processing.analysis_context import AnalysisContext
        context = AnalysisContext(image_array, str(image_path))
        
        # Don't set emotional_impact_score to test fallback
        from photosight.analysis.aesthetic_analyzer import AestheticAnalyzer
        aesthetic_analyzer = AestheticAnalyzer(config)
        
        aesthetic_results = aesthetic_analyzer.analyze_aesthetics(context)
        logger.info(f"  ✓ Fallback Mood Score: {aesthetic_results.get('mood_score', 'N/A')}")
        logger.info(f"  ✓ Fallback Emotional Impact: {aesthetic_results.get('emotional_impact', 'N/A')}")
        logger.info(f"  ✓ Mood Characteristics: {list(aesthetic_results.get('mood_characteristics', {}).keys())}")
        
    except Exception as e:
        logger.error(f"  ✗ Fallback test failed: {e}")

if __name__ == "__main__":
    test_vision_llm_complete()