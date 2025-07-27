#!/usr/bin/env python3
"""
Share Google Drive folders with PhotoSight service account

This script helps share specific folders from sscarrow@gmail.com 
with the PhotoSight service account for access.
"""

import os
import sys
from pathlib import Path

# Add photosight to path
sys.path.insert(0, str(Path(__file__).parent))

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_EMAIL = "photosight-storage@home-dev-sam.iam.gserviceaccount.com"

def authenticate_user_account():
    """Authenticate as sscarrow@gmail.com using application default credentials."""
    try:
        # Use the application default credentials we set up earlier
        from google.auth import default
        credentials, project = default(scopes=['https://www.googleapis.com/auth/drive'])
        service = build('drive', 'v3', credentials=credentials)
        
        # Test authentication
        result = service.about().get(fields="user").execute()
        logger.info(f"Authenticated as: {result.get('user', {}).get('emailAddress', 'Unknown')}")
        return service
        
    except Exception as e:
        logger.error(f"User authentication failed: {e}")
        return None

def list_user_folders(service):
    """List folders accessible to the user account."""
    try:
        query = "mimeType='application/vnd.google-apps.folder'"
        results = service.files().list(
            q=query,
            fields="files(id,name,parents)",
            pageSize=50
        ).execute()
        
        return results.get('files', [])
        
    except HttpError as e:
        logger.error(f"Error listing folders: {e}")
        return []

def share_folder_with_service_account(service, folder_id: str, folder_name: str):
    """Share a folder with the PhotoSight service account."""
    try:
        permission = {
            'type': 'user',
            'role': 'writer',  # Give write access so service account can upload
            'emailAddress': SERVICE_ACCOUNT_EMAIL
        }
        
        service.permissions().create(
            fileId=folder_id,
            body=permission
        ).execute()
        
        logger.info(f"✅ Shared '{folder_name}' with {SERVICE_ACCOUNT_EMAIL}")
        return True
        
    except HttpError as e:
        logger.error(f"❌ Failed to share '{folder_name}': {e}")
        return False

def main():
    print("🔐 PhotoSight Folder Sharing Tool")
    print("=" * 50)
    print(f"📧 Service Account: {SERVICE_ACCOUNT_EMAIL}")
    print()
    
    # Authenticate as user
    service = authenticate_user_account()
    if not service:
        print("❌ Failed to authenticate. Make sure you ran:")
        print("   gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive")
        return 1
    
    # List folders
    print("📁 Available folders:")
    folders = list_user_folders(service)
    
    if not folders:
        print("No folders found")
        return 1
    
    for i, folder in enumerate(folders, 1):
        print(f"  {i:2d}. {folder['name']} ({folder['id']})")
    
    print()
    
    # Get folder selection
    try:
        selection = input("Enter folder numbers to share (comma-separated, or 'all'): ").strip()
        
        if selection.lower() == 'all':
            selected_folders = folders
        else:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_folders = [folders[i] for i in indices if 0 <= i < len(folders)]
        
        if not selected_folders:
            print("No valid folders selected")
            return 1
        
        print(f"\n🔄 Sharing {len(selected_folders)} folder(s)...")
        
        success_count = 0
        for folder in selected_folders:
            if share_folder_with_service_account(service, folder['id'], folder['name']):
                success_count += 1
        
        print(f"\n✅ Successfully shared {success_count}/{len(selected_folders)} folders")
        print(f"🤖 Service account {SERVICE_ACCOUNT_EMAIL} now has access")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Cancelled")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())