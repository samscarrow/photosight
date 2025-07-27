#!/usr/bin/env python3
"""Test clean authentication without warnings"""

from photosight.storage.gdrive_manager import GoogleDriveManager
import logging

# Suppress all but critical messages
logging.basicConfig(level=logging.CRITICAL)

manager = GoogleDriveManager()
if manager.authenticate():
    print("✅ Authentication successful - no file_cache warnings!")
else:
    print("❌ Authentication failed")