#!/usr/bin/env python3
"""
Simple test of Gemini vision API to debug issues.
"""
import os
import google.generativeai as genai
from pathlib import Path
import PIL.Image

# Configure API
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("Please set GEMINI_API_KEY environment variable")
    exit(1)

genai.configure(api_key=api_key)

# Initialize model
model = genai.GenerativeModel('gemini-1.5-flash')

# Test with a simple image
photo_path = "/Users/sam/Desktop/photosight_output/enneagram_workshop/accepted/DSC04826.jpg"

print(f"Testing with: {photo_path}")

try:
    # Load image
    image = PIL.Image.open(photo_path)
    
    # Simple text prompt first
    print("\n1. Testing simple description...")
    response = model.generate_content([
        image,
        "Describe this image in one sentence."
    ])
    print(f"Response: {response.text}")
    
    # Test JSON response
    print("\n2. Testing JSON response...")
    response = model.generate_content([
        image,
        """Analyze this image and provide a JSON response with the following structure:
{
  "scene_type": "indoor/outdoor/portrait/landscape/etc",
  "main_subjects": ["list", "of", "subjects"],
  "mood": "describe the mood",
  "technical_quality": "excellent/good/fair/poor"
}

Respond with ONLY the JSON, no other text."""
    ])
    print(f"Raw response: {response.text}")
    
    # Try to parse JSON
    import json
    try:
        parsed = json.loads(response.text)
        print(f"Parsed successfully: {parsed}")
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response text: '{response.text}'")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()