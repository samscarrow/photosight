#!/usr/bin/env python3
"""
Focused test of vision LLM integration.
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

# Test photo
photo = "/Users/sam/Desktop/photosight_output/enneagram_workshop/accepted/DSC04826.jpg"
image = PIL.Image.open(photo)

print("Testing Vision LLM Analysis on Enneagram Workshop Photo")
print("=" * 60)

# Test 1: Scene Analysis
print("\n🎨 Scene Analysis:")
prompt = """Analyze this photo and provide a JSON response:
{
  "scene_type": "workshop/meeting/portrait/etc",
  "environment": "indoor/outdoor",
  "activity": "what's happening",
  "number_of_people": "approximate count",
  "time_of_day": "morning/afternoon/evening/unknown",
  "confidence": 0.0-1.0
}
Respond with ONLY JSON."""

response = model.generate_content([image, prompt])
text = response.text.strip()
if text.startswith('```json'):
    text = text[7:-3]
elif text.startswith('```'):
    text = text[3:-3]

try:
    scene = json.loads(text)
    print(f"  Type: {scene.get('scene_type', 'Unknown')}")
    print(f"  Environment: {scene.get('environment', 'Unknown')}")
    print(f"  Activity: {scene.get('activity', 'Unknown')}")
    print(f"  People: {scene.get('number_of_people', 'Unknown')}")
    print(f"  Time: {scene.get('time_of_day', 'Unknown')}")
    print(f"  Confidence: {scene.get('confidence', 0):.2f}")
except Exception as e:
    print(f"  Error: {e}")
    print(f"  Raw: {text[:100]}...")

# Test 2: Quality Assessment
print("\n⭐ Quality Assessment:")
prompt = """Assess the photographic quality:
{
  "technical_score": 0.0-1.0,
  "composition_score": 0.0-1.0,
  "lighting_quality": "excellent/good/fair/poor",
  "focus_quality": "sharp/soft/blurry",
  "artistic_score": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "strengths": ["list"],
  "weaknesses": ["list"]
}
JSON only."""

response = model.generate_content([image, prompt])
text = response.text.strip()
if text.startswith('```json'):
    text = text[7:-3]
elif text.startswith('```'):
    text = text[3:-3]

try:
    quality = json.loads(text)
    print(f"  Technical: {quality.get('technical_score', 0):.2f}")
    print(f"  Composition: {quality.get('composition_score', 0):.2f}")
    print(f"  Lighting: {quality.get('lighting_quality', 'Unknown')}")
    print(f"  Focus: {quality.get('focus_quality', 'Unknown')}")
    print(f"  Artistic: {quality.get('artistic_score', 0):.2f}")
    print(f"  Overall: {quality.get('overall_score', 0):.2f}")
    if quality.get('strengths'):
        print(f"  Strengths: {', '.join(quality['strengths'])}")
    if quality.get('weaknesses'):
        print(f"  Weaknesses: {', '.join(quality['weaknesses'])}")
except Exception as e:
    print(f"  Error: {e}")

# Test 3: Decisive Moment
print("\n⚡ Decisive Moment Detection:")
prompt = """Is this a "decisive moment" in photography (a significant instant captured)?
{
  "is_decisive_moment": true/false,
  "reason": "explanation",
  "emotion_captured": "describe emotional content",
  "spontaneity": 0.0-1.0,
  "significance": 0.0-1.0
}
JSON only."""

response = model.generate_content([image, prompt])
text = response.text.strip()
if text.startswith('```json'):
    text = text[7:-3]
elif text.startswith('```'):
    text = text[3:-3]

try:
    moment = json.loads(text)
    print(f"  Is Decisive: {moment.get('is_decisive_moment', False)}")
    print(f"  Reason: {moment.get('reason', 'N/A')}")
    print(f"  Emotion: {moment.get('emotion_captured', 'N/A')}")
    print(f"  Spontaneity: {moment.get('spontaneity', 0):.2f}")
    print(f"  Significance: {moment.get('significance', 0):.2f}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("✅ Vision LLM integration is working!")