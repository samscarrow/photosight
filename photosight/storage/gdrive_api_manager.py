"""
Google Drive API Manager for PhotoSight (Non-Interactive)

Uses service account authentication for fully automated photo management.
"""

import os
import logging
import json
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


@dataclass
class DrivePhoto:
    """Represents a photo stored in Google Drive."""
    file_id: str
    name: str
    web_view_link: str
    download_link: str
    size: int
    created_time: str
    md5_checksum: str
    parent_folders: List[str]


class GoogleDriveAPIManager:
    """Manages PhotoSight photos in Google Drive using API and service account."""
    
    def __init__(self, credentials_path: str = None):
        self.credentials_path = credentials_path or os.path.expanduser("~/.config/photosight-credentials.json")
        self.service = None
        self.project_folder_id = None
        self.photos_folder_id = None
        self.enneagram_folder_id = None
        
        # Initialize the service
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate using service account."""
        try:
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(f"Service account credentials not found: {self.credentials_path}")
            
            # Load service account credentials
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            
            # Build the Drive service
            self.service = build('drive', 'v3', credentials=credentials)
            logger.info("Successfully authenticated with Google Drive API")
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    def create_folder(self, name: str, parent_id: str = None) -> str:
        """Create a folder in Google Drive and return its ID."""
        try:
            folder_metadata = {
                'name': name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                folder_metadata['parents'] = [parent_id]
            
            folder = self.service.files().create(body=folder_metadata, fields='id').execute()
            folder_id = folder.get('id')
            
            logger.info(f"Created folder '{name}' with ID: {folder_id}")
            return folder_id
            
        except HttpError as e:
            logger.error(f"Failed to create folder '{name}': {e}")
            raise
    
    def setup_folder_structure(self) -> Dict[str, str]:
        """Create the PhotoSight folder structure in Google Drive."""
        try:
            folders = {}
            
            # Create main PhotoSight folder
            self.project_folder_id = self.create_folder("PhotoSight")
            folders['photosight'] = self.project_folder_id
            
            # Create Photos subfolder
            self.photos_folder_id = self.create_folder("Photos", self.project_folder_id)
            folders['photos'] = self.photos_folder_id
            
            # Create Enneagram subfolder
            self.enneagram_folder_id = self.create_folder("Enneagram", self.photos_folder_id)
            folders['enneagram'] = self.enneagram_folder_id
            
            logger.info("Folder structure created successfully")
            return folders
            
        except Exception as e:
            logger.error(f"Failed to create folder structure: {e}")
            raise
    
    def upload_file(self, file_path: str, folder_id: str) -> Optional[DrivePhoto]:
        """Upload a file to Google Drive."""
        try:
            filename = Path(file_path).name
            
            # Read file content
            with open(file_path, 'rb') as file:
                file_content = file.read()
            
            # Create media upload
            media = MediaIoBaseUpload(
                io.BytesIO(file_content),
                mimetype='application/octet-stream',
                resumable=True
            )
            
            # File metadata
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            
            # Upload file
            file_obj = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,size,createdTime,md5Checksum,webViewLink,webContentLink'
            ).execute()
            
            # Create DrivePhoto object
            drive_photo = DrivePhoto(
                file_id=file_obj['id'],
                name=file_obj['name'],
                web_view_link=file_obj.get('webViewLink', ''),
                download_link=file_obj.get('webContentLink', ''),
                size=int(file_obj.get('size', 0)),
                created_time=file_obj.get('createdTime', ''),
                md5_checksum=file_obj.get('md5Checksum', ''),
                parent_folders=[folder_id]
            )
            
            logger.info(f"Uploaded {filename} -> {file_obj['id']}")
            return drive_photo
            
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            return None
    
    def upload_enneagram_photos(self, local_directory: str) -> List[DrivePhoto]:
        """Upload all enneagram ARW photos to Google Drive."""
        uploaded_photos = []
        
        # Setup folder structure
        folders = self.setup_folder_structure()
        if not self.enneagram_folder_id:
            logger.error("Failed to create folder structure")
            return []
        
        # Find all ARW files
        arw_files = list(Path(local_directory).glob("*.ARW"))
        logger.info(f"Found {len(arw_files)} ARW files to upload")
        
        for i, arw_file in enumerate(arw_files, 1):
            logger.info(f"Uploading {i}/{len(arw_files)}: {arw_file.name}")
            
            drive_photo = self.upload_file(str(arw_file), self.enneagram_folder_id)
            if drive_photo:
                uploaded_photos.append(drive_photo)
            else:
                logger.warning(f"Failed to upload {arw_file.name}")
        
        logger.info(f"Successfully uploaded {len(uploaded_photos)}/{len(arw_files)} photos")
        return uploaded_photos
    
    def download_file_content(self, file_id: str) -> Optional[bytes]:
        """Download file content by ID."""
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            return file_content.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            return None
    
    def get_file_info(self, file_id: str) -> Optional[Dict]:
        """Get file information from Google Drive."""
        try:
            file_info = self.service.files().get(
                fileId=file_id,
                fields='id,name,size,createdTime,md5Checksum,webViewLink,webContentLink,parents'
            ).execute()
            
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_id}: {e}")
            return None
    
    def share_with_user(self, file_id: str, email: str, role: str = 'reader'):
        """Share a file with a specific user."""
        try:
            permission = {
                'type': 'user',
                'role': role,
                'emailAddress': email
            }
            
            self.service.permissions().create(
                fileId=file_id,
                body=permission
            ).execute()
            
            logger.info(f"Shared file {file_id} with {email} as {role}")
            
        except Exception as e:
            logger.error(f"Failed to share file {file_id} with {email}: {e}")
    
    def share_folder_with_user(self, folder_id: str, email: str, role: str = 'reader'):
        """Share a folder (and all contents) with a specific user."""
        try:
            # Share the folder
            self.share_with_user(folder_id, email, role)
            
            # List all files in the folder and share them too
            results = self.service.files().list(
                q=f"'{folder_id}' in parents",
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            for file in files:
                self.share_with_user(file['id'], email, role)
            
            logger.info(f"Shared folder {folder_id} and {len(files)} files with {email}")
            
        except Exception as e:
            logger.error(f"Failed to share folder {folder_id} with {email}: {e}")


def create_drive_photo_mapping(local_directory: str, share_with_email: str = "sscarrow@gmail.com") -> Dict[str, str]:
    """Upload photos and create mapping, then share with main account."""
    try:
        drive_manager = GoogleDriveAPIManager()
        
        # Upload photos
        uploaded_photos = drive_manager.upload_enneagram_photos(local_directory)
        
        if not uploaded_photos:
            logger.error("No photos were uploaded")
            return {}
        
        # Share the enneagram folder with the main account
        if drive_manager.enneagram_folder_id and share_with_email:
            logger.info(f"Sharing folder with {share_with_email}...")
            drive_manager.share_folder_with_user(
                drive_manager.enneagram_folder_id, 
                share_with_email, 
                'reader'
            )
        
        # Create mapping: filename -> file_id
        mapping = {}
        for photo in uploaded_photos:
            filename_base = Path(photo.name).stem  # DSC04778
            mapping[filename_base] = photo.file_id
        
        logger.info(f"Created mapping for {len(mapping)} photos")
        return mapping
        
    except Exception as e:
        logger.error(f"Failed to create photo mapping: {e}")
        return {}


if __name__ == "__main__":
    # Test the Google Drive API integration
    logging.basicConfig(level=logging.INFO)
    
    test_dir = "/Users/sam/Library/CloudStorage/GoogleDrive-sscarrow@gmail.com/Other computers/USB and External Devices/Untitled"
    
    if os.path.exists(test_dir):
        print(f"Testing upload from: {test_dir}")
        mapping = create_drive_photo_mapping(test_dir)
        print(f"Created mapping for {len(mapping)} photos")
        if mapping:
            print("Sample mapping:", dict(list(mapping.items())[:3]))
    else:
        print(f"Test directory not found: {test_dir}")
        print("Please update the path to your ARW files")