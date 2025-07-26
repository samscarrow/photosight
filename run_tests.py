#!/usr/bin/env python3
"""
Simple test runner to verify dependency injection works without external dependencies.
"""

import sys
import os
from typing import List, Dict, Any, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the external dependencies to avoid import errors
class MockConfig:
    pass

class MockPhoto:
    pass

class MockAnalysisResult:
    pass

# Replace problematic imports with mocks
sys.modules['yaml'] = MockConfig()
sys.modules['photosight.config'] = MockConfig()
sys.modules['photosight.db.models'] = type('MockModels', (), {
    'Photo': MockPhoto,
    'AnalysisResult': MockAnalysisResult
})()

def test_dependency_injection():
    """Test that dependency injection pattern works correctly."""
    print("🧪 Testing Dependency Injection Pattern...")
    
    # Now we can import our module
    from photosight.selection.quality_ranker import (
        PhotoQualityRanker,
        RankingWeights,
        PhotoDataProvider,
        DefaultPhotoDataProvider
    )
    
    print("✅ Successfully imported modules with dependency injection")
    
    # Test 1: Default provider injection
    ranker = PhotoQualityRanker()
    assert ranker.data_provider is not None
    assert isinstance(ranker.data_provider, DefaultPhotoDataProvider)
    print("✅ Default provider injection works")
    
    # Test 2: Custom provider injection
    class TestProvider:
        def get_photos_for_selection(self, project_name=None, limit=2000):
            return [{'ID': 1, 'FILENAME': 'test.jpg'}]
    
    test_provider = TestProvider()
    ranker_with_custom = PhotoQualityRanker(data_provider=test_provider)
    assert ranker_with_custom.data_provider is test_provider
    print("✅ Custom provider injection works")
    
    # Test 3: RankingWeights defaults
    weights = RankingWeights()
    total = (
        weights.sharpness + weights.exposure + weights.composition +
        weights.face_quality + weights.emotional_impact + weights.technical_excellence
    )
    assert abs(total - 1.0) < 0.001
    print("✅ RankingWeights defaults sum to 1.0")
    
    # Test 4: RankingWeights from dict
    partial_dict = {'sharpness': 0.5, 'composition': 0.3}
    custom_weights = RankingWeights.from_dict(partial_dict)
    assert custom_weights.sharpness == 0.5
    assert custom_weights.composition == 0.3
    assert custom_weights.exposure == 0.15  # Default value
    print("✅ RankingWeights.from_dict works with partial values")
    
    # Test 5: Mock photo scoring
    mock_photo = {
        'SHARPNESS_SCORE': 0.8,
        'EXPOSURE_QUALITY': 0.7,
        'COMPOSITION_SCORE': 0.9,
        'FACE_QUALITY_SCORE': 0.6,
        'EMOTIONAL_IMPACT': 0.5,
        'TECHNICAL_EXCELLENCE': 0.7,
        'PERSON_DETECTED': True,
        'HIGHLIGHT_CLIPPING': 0.02,
        'SHADOW_CLIPPING': 0.01
    }
    
    score = ranker.calculate_composite_score_from_dict(mock_photo)
    assert 0.0 <= score <= 1.0
    print(f"✅ Composite score calculation works: {score:.3f}")
    
    # Test 6: Burst detection with mock data
    mock_photos = [
        {
            'ID': 1,
            'DATE_TAKEN': '2024-01-01T10:00:00Z',
            'ISO': 100,
            'APERTURE': 2.8,
            'FOCAL_LENGTH': 50
        },
        {
            'ID': 2, 
            'DATE_TAKEN': '2024-01-01T10:00:02Z',
            'ISO': 100,
            'APERTURE': 2.8,
            'FOCAL_LENGTH': 50
        },
        {
            'ID': 3,
            'DATE_TAKEN': '2024-01-01T10:00:30Z',  # Different time
            'ISO': 200,
            'APERTURE': 4.0,
            'FOCAL_LENGTH': 85
        }
    ]
    
    burst_groups = ranker.detect_burst_groups_from_dicts(mock_photos)
    assert len(burst_groups) == 2  # One burst, one single
    assert len(burst_groups[0]) == 2  # Burst of 2 photos
    assert len(burst_groups[1]) == 1  # Single photo
    print("✅ Burst detection works")
    
    print("\n🎉 All dependency injection tests passed!")
    return True

def test_architecture_patterns():
    """Test that the architecture follows good patterns."""
    print("\n🏗️  Testing Architecture Patterns...")
    
    from photosight.selection.quality_ranker import PhotoQualityRanker, RankingWeights
    
    # Test separation of concerns
    ranker = PhotoQualityRanker()
    
    # Data provider is separate concern
    assert hasattr(ranker, 'data_provider')
    
    # Weights are separate dataclass
    assert isinstance(ranker.weights, RankingWeights)
    
    # Methods are focused and testable
    methods = ['calculate_composite_score_from_dict', 'detect_burst_groups_from_dicts', 
               'rank_photos_from_dicts', 'select_best_photos', 'get_selection_report']
    
    for method in methods:
        assert hasattr(ranker, method)
        assert callable(getattr(ranker, method))
    
    print("✅ Good separation of concerns")
    print("✅ Testable method interfaces")
    print("✅ Dependency injection pattern implemented")
    
    print("\n🎉 All architecture tests passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_dependency_injection() and test_architecture_patterns()
        if success:
            print(f"\n{'='*60}")
            print("🚀 ALL TESTS PASSED - DEPENDENCY INJECTION WORKING!")
            print(f"{'='*60}")
            sys.exit(0)
        else:
            print("❌ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)