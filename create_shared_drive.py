#!/usr/bin/env python3
"""
Create a Google Shared Drive for PhotoSight and share with service account

This fixes the storage quota issue by using Shared Drives instead of personal Drive.
"""

import os
import sys
from pathlib import Path

# Add photosight to path
sys.path.insert(0, str(Path(__file__).parent))

from google.auth import default
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
        credentials, project = default(scopes=['https://www.googleapis.com/auth/drive'])
        service = build('drive', 'v3', credentials=credentials)
        
        # Test authentication
        result = service.about().get(fields="user").execute()
        logger.info(f"Authenticated as: {result.get('user', {}).get('emailAddress', 'Unknown')}")
        return service
        
    except Exception as e:
        logger.error(f"User authentication failed: {e}")
        return None

def create_shared_drive(service, name: str):
    """Create a new Shared Drive."""
    try:
        drive_metadata = {
            'name': name
        }
        
        # Create shared drive with a request ID (required)
        import uuid
        request_id = str(uuid.uuid4())
        
        drive = service.drives().create(
            body=drive_metadata,
            requestId=request_id,
            fields='id,name'
        ).execute()
        
        logger.info(f"✅ Created Shared Drive '{name}' with ID: {drive.get('id')}")
        return drive.get('id')
        
    except HttpError as e:
        if 'insufficient_permissions' in str(e) or 'permission' in str(e).lower():
            logger.error(f"❌ Insufficient permissions to create Shared Drive. You need Google Workspace admin privileges.")
            logger.error("   Alternative: Ask your Google Workspace admin to create the Shared Drive")
            return None
        else:
            logger.error(f"❌ Failed to create Shared Drive '{name}': {e}")
            return None

def add_service_account_to_shared_drive(service, drive_id: str, email: str):
    """Add service account as organizer to the Shared Drive."""
    try:
        permission = {
            'type': 'user',
            'role': 'organizer',  # Full access to manage the shared drive
            'emailAddress': email
        }
        
        service.permissions().create(
            fileId=drive_id,
            body=permission,
            supportsAllDrives=True
        ).execute()
        
        logger.info(f"✅ Added {email} as organizer to Shared Drive")
        return True
        
    except HttpError as e:
        logger.error(f"❌ Failed to add service account to Shared Drive: {e}")
        return False

def create_folder_structure_in_shared_drive(service, drive_id: str):
    """Create folder structure in the Shared Drive."""
    try:
        folders = {}
        
        folder_names = ['RAW Photos', 'Processed Photos', 'Thumbnails', 'Enneagram']
        
        for folder_name in folder_names:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [drive_id]
            }
            
            folder = service.files().create(
                body=folder_metadata,
                fields='id,name',
                supportsAllDrives=True
            ).execute()
            
            folders[folder_name.lower().replace(' ', '_')] = folder.get('id')
            logger.info(f"✅ Created folder '{folder_name}' with ID: {folder.get('id')}")
        
        return folders
        
    except HttpError as e:
        logger.error(f"❌ Failed to create folder structure: {e}")
        return {}

def main():
    print("🚀 PhotoSight Shared Drive Setup")
    print("=" * 50)
    print(f"📧 Service Account: {SERVICE_ACCOUNT_EMAIL}")
    print()
    
    # Authenticate as user
    service = authenticate_user_account()
    if not service:
        print("❌ Failed to authenticate")
        return 1
    
    print("🔄 Creating PhotoSight Shared Drive...")
    
    # Create shared drive
    drive_id = create_shared_drive(service, "PhotoSight")
    if not drive_id:
        print("❌ Failed to create Shared Drive")
        print("💡 Try these alternatives:")
        print("   1. Ask your Google Workspace admin to create the Shared Drive")
        print("   2. Use domain-wide delegation")
        print("   3. Use OAuth instead of service account")
        return 1
    
    # Add service account to shared drive
    print(f"\n🔄 Adding service account to Shared Drive...")
    if not add_service_account_to_shared_drive(service, drive_id, SERVICE_ACCOUNT_EMAIL):
        print("❌ Failed to add service account to Shared Drive")
        return 1
    
    # Create folder structure
    print(f"\n🔄 Creating folder structure...")
    folders = create_folder_structure_in_shared_drive(service, drive_id)
    
    if not folders:
        print("❌ Failed to create folder structure")
        return 1
    
    # Print results
    print(f"\n📋 Created Shared Drive structure:")
    print(f"   🚗 PhotoSight Shared Drive ({drive_id})")
    for folder_type, folder_id in folders.items():
        print(f"      📁 {folder_type.replace('_', ' ').title()} ({folder_id})")
    
    # Save configuration
    config_content = f"""# PhotoSight Google Shared Drive Configuration
PHOTOSIGHT_SHARED_DRIVE_ID={drive_id}
PHOTOSIGHT_RAW_FOLDER_ID={folders.get('raw_photos', '')}
PHOTOSIGHT_PROCESSED_FOLDER_ID={folders.get('processed_photos', '')}
PHOTOSIGHT_THUMBNAILS_FOLDER_ID={folders.get('thumbnails', '')}
PHOTOSIGHT_ENNEAGRAM_FOLDER_ID={folders.get('enneagram', '')}
PHOTOSIGHT_SERVICE_ACCOUNT_EMAIL={SERVICE_ACCOUNT_EMAIL}
"""
    
    with open('photosight_shared_drive_config.env', 'w') as f:
        f.write(config_content)
    
    print(f"\n💾 Saved configuration to: photosight_shared_drive_config.env")
    print(f"🎉 Setup complete! Service account can now upload to Shared Drive.")
    print(f"🔗 View Shared Drive: https://drive.google.com/drive/folders/{drive_id}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())