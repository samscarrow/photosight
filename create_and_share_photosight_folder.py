#!/usr/bin/env python3
"""
Create PhotoSight folder in sscarrow@gmail.com's Google Drive and share with service account

This script:
1. Creates a new "PhotoSight" folder in sscarrow@gmail.com's Google Drive
2. Shares it with the PhotoSight service account with writer permissions
3. Creates subfolders for organization
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

def create_folder(service, name: str, parent_id: str = None):
    """Create a folder in Google Drive."""
    try:
        folder_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_id:
            folder_metadata['parents'] = [parent_id]
        
        folder = service.files().create(body=folder_metadata, fields='id,name').execute()
        logger.info(f"✅ Created folder '{name}' with ID: {folder.get('id')}")
        return folder.get('id')
        
    except HttpError as e:
        logger.error(f"❌ Failed to create folder '{name}': {e}")
        return None

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
    print("📁 PhotoSight Folder Creation Tool")
    print("=" * 50)
    print(f"📧 Service Account: {SERVICE_ACCOUNT_EMAIL}")
    print()
    
    # Authenticate as user
    service = authenticate_user_account()
    if not service:
        print("❌ Failed to authenticate. Make sure you ran:")
        print("   gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive")
        return 1
    
    print("🔄 Creating PhotoSight folder structure...")
    
    # Create main PhotoSight folder
    photosight_folder_id = create_folder(service, "PhotoSight")
    if not photosight_folder_id:
        print("❌ Failed to create main PhotoSight folder")
        return 1
    
    # Create subfolders
    subfolders = {
        "RAW Photos": create_folder(service, "RAW Photos", photosight_folder_id),
        "Processed Photos": create_folder(service, "Processed Photos", photosight_folder_id),
        "Thumbnails": create_folder(service, "Thumbnails", photosight_folder_id),
        "Enneagram": create_folder(service, "Enneagram", photosight_folder_id)
    }
    
    # Share main folder with service account
    print(f"\n🔄 Sharing PhotoSight folder with service account...")
    if share_folder_with_service_account(service, photosight_folder_id, "PhotoSight"):
        print("✅ Successfully shared PhotoSight folder!")
    else:
        print("❌ Failed to share PhotoSight folder")
        return 1
    
    # Print folder structure
    print(f"\n📋 Created folder structure:")
    print(f"   📁 PhotoSight ({photosight_folder_id})")
    for name, folder_id in subfolders.items():
        if folder_id:
            print(f"      📁 {name} ({folder_id})")
    
    print(f"\n🎉 Setup complete!")
    print(f"📧 Service account {SERVICE_ACCOUNT_EMAIL} now has writer access")
    print(f"🔗 View folder: https://drive.google.com/drive/folders/{photosight_folder_id}")
    
    # Save folder IDs to a config file
    config_content = f"""# PhotoSight Google Drive Folder Configuration
PHOTOSIGHT_MAIN_FOLDER_ID={photosight_folder_id}
PHOTOSIGHT_RAW_FOLDER_ID={subfolders.get('RAW Photos', '')}
PHOTOSIGHT_PROCESSED_FOLDER_ID={subfolders.get('Processed Photos', '')}
PHOTOSIGHT_THUMBNAILS_FOLDER_ID={subfolders.get('Thumbnails', '')}
PHOTOSIGHT_ENNEAGRAM_FOLDER_ID={subfolders.get('Enneagram', '')}
PHOTOSIGHT_SERVICE_ACCOUNT_EMAIL={SERVICE_ACCOUNT_EMAIL}
"""
    
    with open('photosight_gdrive_config.env', 'w') as f:
        f.write(config_content)
    
    print(f"💾 Saved configuration to: photosight_gdrive_config.env")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())