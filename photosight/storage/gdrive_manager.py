"""
Google Drive Storage Manager for PhotoSight

Handles uploading, organizing, and serving photos from Google Drive using service account authentication.
"""

import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import tempfile
import io

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError
from googleapiclient.discovery_cache.base import Cache

# Disable file cache to avoid warnings
class NoCache(Cache):
    def get(self, url):
        return None
    def set(self, url, content):
        pass

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


class GoogleDriveManager:
    """Manages PhotoSight photos in Google Drive using service account authentication."""
    
    def __init__(self, credentials_file: str = None):
        # Default to service account credentials in project root
        if credentials_file is None:
            credentials_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "photosight-service-account-key.json"
            )
        
        self.credentials_file = credentials_file
        self.service = None
        
        # Load folder IDs from environment or config
        self.folder_ids = self._load_folder_config()
        
    def _load_folder_config(self) -> Dict[str, str]:
        """Load Google Drive folder IDs from environment variables or config file."""
        folder_ids = {}
        
        # Try environment variables first
        folder_ids['main'] = os.getenv('PHOTOSIGHT_MAIN_FOLDER_ID')
        folder_ids['raw'] = os.getenv('PHOTOSIGHT_RAW_FOLDER_ID')
        folder_ids['processed'] = os.getenv('PHOTOSIGHT_PROCESSED_FOLDER_ID')
        folder_ids['thumbnails'] = os.getenv('PHOTOSIGHT_THUMBNAILS_FOLDER_ID')
        folder_ids['enneagram'] = os.getenv('PHOTOSIGHT_ENNEAGRAM_FOLDER_ID')
        
        # If not in environment, try loading from config file
        if not folder_ids['main']:
            config_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "photosight_gdrive_config.env"
            )
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            if key == 'PHOTOSIGHT_MAIN_FOLDER_ID':
                                folder_ids['main'] = value
                            elif key == 'PHOTOSIGHT_RAW_FOLDER_ID':
                                folder_ids['raw'] = value
                            elif key == 'PHOTOSIGHT_PROCESSED_FOLDER_ID':
                                folder_ids['processed'] = value
                            elif key == 'PHOTOSIGHT_THUMBNAILS_FOLDER_ID':
                                folder_ids['thumbnails'] = value
                            elif key == 'PHOTOSIGHT_ENNEAGRAM_FOLDER_ID':
                                folder_ids['enneagram'] = value
        
        return folder_ids
        
    def authenticate(self) -> bool:
        """Authenticate with Google Drive using service account."""
        try:
            if not os.path.exists(self.credentials_file):
                logger.error(f"Service account credentials not found: {self.credentials_file}")
                return False
            
            # Load service account credentials
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_file,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            
            # Build the Drive service with no cache to avoid warnings
            self.service = build('drive', 'v3', credentials=credentials, cache=NoCache())
            
            # Test authentication by making a simple API call
            result = self.service.about().get(fields="user").execute()
            logger.info(f"Authenticated as service account: {result.get('user', {}).get('emailAddress', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def setup_folder_structure(self) -> Dict[str, str]:
        """Create folder structure in Google Drive and return folder IDs."""
        try:
            if not self.service:
                logger.error("Drive service not authenticated")
                return {}
            
            folders = {}
            
            # Create main PhotoSight folder
            folder_metadata = {
                'name': self.project_folder,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.service.files().create(body=folder_metadata, fields='id').execute()
            folders['photosight'] = folder.get('id')
            logger.info(f"Created PhotoSight folder: {folders['photosight']}")
            
            # Create Photos subfolder
            folder_metadata = {
                'name': 'Photos',
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [folders['photosight']]
            }
            folder = self.service.files().create(body=folder_metadata, fields='id').execute()
            folders['photos'] = folder.get('id')
            logger.info(f"Created Photos folder: {folders['photos']}")
            
            # Create Enneagram subfolder
            folder_metadata = {
                'name': 'Enneagram',
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [folders['photos']]
            }
            folder = self.service.files().create(body=folder_metadata, fields='id').execute()
            folders['enneagram'] = folder.get('id')
            logger.info(f"Created Enneagram folder: {folders['enneagram']}")
            
            return folders
            
        except HttpError as e:
            logger.error(f"Failed to create folder structure: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to create folder structure: {e}")
            return {}
    
    def upload_photo(self, local_path: str, drive_folder_id: str) -> Optional[DrivePhoto]:
        """Upload a single photo to Google Drive."""
        try:
            if not self.service:
                logger.error("Drive service not authenticated")
                return None
                
            filename = Path(local_path).name
            
            # File metadata
            file_metadata = {
                'name': filename,
                'parents': [drive_folder_id]
            }
            
            # Media upload
            media = MediaFileUpload(local_path, resumable=True)
            
            # Upload the file
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            if file_id:
                # Get file details
                details = self._get_file_details(file_id)
                if details:
                    logger.info(f"Uploaded {filename} -> {file_id}")
                    return details
            
            logger.error(f"Failed to upload {filename}")
            return None
            
        except HttpError as e:
            logger.error(f"HTTP error uploading {local_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error uploading {local_path}: {e}")
            return None
    
    def upload_enneagram_photos(self, local_directory: str) -> List[DrivePhoto]:
        """Upload all enneagram photos from local directory to Google Drive."""
        uploaded_photos = []
        
        # Authenticate first
        if not self.authenticate():
            logger.error("Failed to authenticate with Google Drive")
            return []
        
        # Get enneagram folder ID from config
        enneagram_folder_id = self.folder_ids.get('enneagram')
        if not enneagram_folder_id:
            logger.error("Enneagram folder ID not configured")
            return []
        
        # Find all ARW files
        arw_files = list(Path(local_directory).glob("*.ARW"))
        logger.info(f"Found {len(arw_files)} ARW files to upload to folder {enneagram_folder_id}")
        
        for arw_file in arw_files:
            logger.info(f"Uploading {arw_file.name}...")
            
            drive_photo = self.upload_photo(str(arw_file), enneagram_folder_id)
            if drive_photo:
                uploaded_photos.append(drive_photo)
            else:
                logger.warning(f"Failed to upload {arw_file.name}")
        
        logger.info(f"Successfully uploaded {len(uploaded_photos)} photos")
        return uploaded_photos
    
    def upload_raw_photo(self, local_path: str) -> Optional[DrivePhoto]:
        """Upload a raw photo to the RAW Photos folder."""
        if not self.authenticate():
            logger.error("Failed to authenticate with Google Drive")
            return None
            
        raw_folder_id = self.folder_ids.get('raw')
        if not raw_folder_id:
            logger.error("RAW Photos folder ID not configured")
            return None
            
        return self.upload_photo(local_path, raw_folder_id)
    
    def upload_processed_photo(self, local_path: str) -> Optional[DrivePhoto]:
        """Upload a processed photo to the Processed Photos folder."""
        if not self.authenticate():
            logger.error("Failed to authenticate with Google Drive")
            return None
            
        processed_folder_id = self.folder_ids.get('processed')
        if not processed_folder_id:
            logger.error("Processed Photos folder ID not configured")
            return None
            
        return self.upload_photo(local_path, processed_folder_id)
    
    def upload_thumbnail(self, local_path: str) -> Optional[DrivePhoto]:
        """Upload a thumbnail to the Thumbnails folder."""
        if not self.authenticate():
            logger.error("Failed to authenticate with Google Drive")
            return None
            
        thumbnails_folder_id = self.folder_ids.get('thumbnails')
        if not thumbnails_folder_id:
            logger.error("Thumbnails folder ID not configured")
            return None
            
        return self.upload_photo(local_path, thumbnails_folder_id)
    
    def get_download_link(self, file_id: str) -> Optional[str]:
        """Get direct download link for a file."""
        try:
            if not self.service:
                logger.error("Drive service not authenticated")
                return None
                
            file = self.service.files().get(fileId=file_id, fields='webContentLink').execute()
            return file.get('webContentLink')
            
        except HttpError as e:
            logger.error(f"HTTP error getting download link for {file_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting download link for {file_id}: {e}")
            return None
    
    def download_photo_stream(self, file_id: str) -> Optional[bytes]:
        """Download photo content as bytes for thumbnail generation."""
        try:
            if not self.service:
                logger.error("Drive service not authenticated")
                return None
                
            request = self.service.files().get_media(fileId=file_id)
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                
            return file_io.getvalue()
                
        except HttpError as e:
            logger.error(f"HTTP error downloading {file_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {file_id}: {e}")
            return None
    
    def _get_file_details(self, file_id: str) -> Optional[DrivePhoto]:
        """Get detailed file information from Google Drive."""
        try:
            if not self.service:
                logger.error("Drive service not authenticated")
                return None
                
            file = self.service.files().get(
                fileId=file_id,
                fields='id,name,webViewLink,webContentLink,size,createdTime,md5Checksum'
            ).execute()
            
            return DrivePhoto(
                file_id=file_id,
                name=file.get('name', ''),
                web_view_link=file.get('webViewLink', ''),
                download_link=file.get('webContentLink', ''),
                size=int(file.get('size', 0)),
                created_time=file.get('createdTime', ''),
                md5_checksum=file.get('md5Checksum', '')
            )
            
        except HttpError as e:
            logger.error(f"HTTP error getting file details for {file_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting file details for {file_id}: {e}")
            return None
    
    def list_folders(self, folder_name: str = None) -> List[Dict[str, str]]:
        """List folders in Google Drive, optionally filtered by name."""
        try:
            if not self.service:
                logger.error("Drive service not authenticated")
                return []
            
            query = "mimeType='application/vnd.google-apps.folder'"
            if folder_name:
                query += f" and name='{folder_name}'"
            
            results = self.service.files().list(
                q=query,
                fields="files(id,name,parents)"
            ).execute()
            
            return results.get('files', [])
            
        except HttpError as e:
            logger.error(f"HTTP error listing folders: {e}")
            return []
        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            return []
    
    def share_with_user(self, file_id: str, email: str, role: str = 'reader') -> bool:
        """Share a file or folder with a user."""
        try:
            if not self.service:
                logger.error("Drive service not authenticated")
                return False
            
            permission = {
                'type': 'user',
                'role': role,
                'emailAddress': email
            }
            
            self.service.permissions().create(
                fileId=file_id,
                body=permission
            ).execute()
            
            logger.info(f"Shared {file_id} with {email} as {role}")
            return True
            
        except HttpError as e:
            logger.error(f"HTTP error sharing {file_id} with {email}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sharing {file_id} with {email}: {e}")
            return False


def create_drive_photo_mapping(local_directory: str) -> Dict[str, str]:
    """Create a mapping of local filenames to Google Drive file IDs."""
    drive_manager = GoogleDriveManager()
    uploaded_photos = drive_manager.upload_enneagram_photos(local_directory)
    
    # Create mapping: filename -> file_id
    mapping = {}
    for photo in uploaded_photos:
        # Extract DSC number from filename (e.g., DSC04778.ARW -> 04778)
        filename_base = Path(photo.name).stem  # DSC04778
        mapping[filename_base] = photo.file_id
    
    return mapping


if __name__ == "__main__":
    # Test the Google Drive integration
    logging.basicConfig(level=logging.INFO)
    
    # Test directory (use the Google Drive location you mentioned)
    test_dir = "/Users/sam/Library/CloudStorage/GoogleDrive-sscarrow@gmail.com/Other computers/USB and External Devices/Untitled"
    
    if os.path.exists(test_dir):
        print(f"Testing upload from: {test_dir}")
        mapping = create_drive_photo_mapping(test_dir)
        print(f"Created mapping for {len(mapping)} photos")
        print("Sample mapping:", dict(list(mapping.items())[:3]))
    else:
        print(f"Test directory not found: {test_dir}")
        print("Please update the path to your ARW files")