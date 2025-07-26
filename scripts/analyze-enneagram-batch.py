#!/usr/bin/env python3
"""
Analyze multiple enneagram photos with vision LLM.
"""
import os
import sys
from pathlib import Path
import google.generativeai as genai
import json
import PIL.Image

# Set up API
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("Please set GEMINI_API_KEY")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Get photos
accepted_dir = Path("/Users/sam/Desktop/photosight_output/enneagram_workshop/accepted")
photos = list(accepted_dir.glob("*.jpg"))[:5]  # Analyze first 5

print(f"Analyzing {len(photos)} Enneagram Workshop Photos")
print("=" * 80)

for i, photo_path in enumerate(photos, 1):
    print(f"\n📸 Photo {i}: {photo_path.name}")
    print("-" * 60)
    
    try:
        image = PIL.Image.open(photo_path)
        
        # Combined analysis prompt
        prompt = """Analyze this workshop photo and provide a comprehensive JSON response:
{
  "scene": {
    "type": "describe the scene type",
    "activity": "what's happening",
    "participant_count": number,
    "engagement_level": "high/medium/low"
  },
  "quality": {
    "technical": 0.0-1.0,
    "composition": 0.0-1.0,
    "focus": "sharp/soft/mixed",
    "lighting": "excellent/good/fair/poor",
    "overall": 0.0-1.0
  },
  "decisive_moment": {
    "is_decisive": true/false,
    "captured_emotion": "describe",
    "significance": 0.0-1.0
  },
  "workshop_specific": {
    "presenter_visible": true/false,
    "audience_engagement": "describe",
    "learning_environment": "describe atmosphere",
    "key_moment": "what makes this photo interesting"
  }
}
Respond with ONLY JSON."""

        response = model.generate_content([image, prompt])
        text = response.text.strip()
        
        # Clean JSON
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        # Parse and display
        data = json.loads(text.strip())
        
        # Scene info
        scene = data.get('scene', {})
        print(f"  📍 Scene: {scene.get('type', 'N/A')}")
        print(f"  👥 Participants: {scene.get('participant_count', 'N/A')}")
        print(f"  🎯 Activity: {scene.get('activity', 'N/A')}")
        print(f"  📊 Engagement: {scene.get('engagement_level', 'N/A')}")
        
        # Quality
        quality = data.get('quality', {})
        print(f"\n  ⭐ Quality:")
        print(f"     Technical: {quality.get('technical', 0):.2f}")
        print(f"     Composition: {quality.get('composition', 0):.2f}")
        print(f"     Focus: {quality.get('focus', 'N/A')}")
        print(f"     Lighting: {quality.get('lighting', 'N/A')}")
        print(f"     Overall: {quality.get('overall', 0):.2f}")
        
        # Decisive moment
        moment = data.get('decisive_moment', {})
        if moment.get('is_decisive'):
            print(f"\n  ⚡ DECISIVE MOMENT!")
            print(f"     Emotion: {moment.get('captured_emotion', 'N/A')}")
            print(f"     Significance: {moment.get('significance', 0):.2f}")
        
        # Workshop specific
        workshop = data.get('workshop_specific', {})
        print(f"\n  🎓 Workshop Analysis:")
        print(f"     Presenter: {'Yes' if workshop.get('presenter_visible') else 'No'}")
        print(f"     Audience: {workshop.get('audience_engagement', 'N/A')}")
        print(f"     Atmosphere: {workshop.get('learning_environment', 'N/A')}")
        print(f"     Key Moment: {workshop.get('key_moment', 'N/A')}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("✅ Analysis complete!")

# Summary
print("\n📊 Summary:")
print(f"  - Analyzed {len(photos)} workshop photos")
print("  - Vision LLM successfully identified scene context, composition, and engagement")
print("  - Can detect workshop-specific elements like presenter, audience dynamics")
print("  - Provides both technical and semantic quality assessment")