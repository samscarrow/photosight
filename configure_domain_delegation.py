#!/usr/bin/env python3
"""
Configure domain-wide delegation programmatically using Google Admin SDK
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

SERVICE_ACCOUNT_CLIENT_ID = "110517673134127106176"
OAUTH_SCOPES = "https://www.googleapis.com/auth/drive"

def authenticate_admin():
    """Authenticate with Google Workspace Admin SDK."""
    try:
        credentials, project = default(scopes=[
            'https://www.googleapis.com/auth/admin.directory.domain',
            'https://www.googleapis.com/auth/cloud-platform'
        ])
        
        admin_service = build('admin', 'directory_v1', credentials=credentials)
        
        # Test authentication
        try:
            result = admin_service.customers().get(customerKey='my_customer').execute()
            logger.info(f"Authenticated with Admin SDK for domain: {result.get('customerDomain', 'Unknown')}")
            return admin_service
        except HttpError as e:
            if 'forbidden' in str(e).lower():
                logger.error("Access denied. You need Google Workspace Admin privileges.")
                return None
            raise
        
    except Exception as e:
        logger.error(f"Admin SDK authentication failed: {e}")
        return None

def configure_domain_wide_delegation(admin_service):
    """Configure domain-wide delegation for the PhotoSight service account."""
    try:
        # There's no direct API for domain-wide delegation configuration
        # We need to use the OAuth2 API or manual configuration
        
        logger.info("Checking existing domain-wide delegation settings...")
        
        # Try to list existing authorizations (if available)
        try:
            # This endpoint may not exist or may require different authentication
            result = admin_service.tokens().list(userKey='all').execute()
            logger.info(f"Found {len(result.get('items', []))} existing tokens")
        except Exception as e:
            logger.warning(f"Could not list existing tokens: {e}")
        
        logger.error("❌ Direct API configuration not available")
        logger.info("💡 Domain-wide delegation must be configured manually in Admin Console")
        
        return False
        
    except HttpError as e:
        logger.error(f"HTTP error configuring delegation: {e}")
        return False
    except Exception as e:
        logger.error(f"Error configuring delegation: {e}")
        return False

def check_workspace_admin_access():
    """Check if current user has Google Workspace admin access."""
    try:
        credentials, project = default()
        admin_service = build('admin', 'directory_v1', credentials=credentials)
        
        # Try to access admin info
        result = admin_service.customers().get(customerKey='my_customer').execute()
        logger.info(f"✅ Google Workspace Admin access confirmed")
        logger.info(f"   Domain: {result.get('customerDomain', 'Unknown')}")
        logger.info(f"   Customer ID: {result.get('id', 'Unknown')}")
        return True
        
    except HttpError as e:
        if 'forbidden' in str(e).lower() or 'access_denied' in str(e).lower():
            logger.error("❌ No Google Workspace Admin access")
            logger.error("   You need Super Admin privileges to configure domain-wide delegation")
            return False
        else:
            logger.error(f"❌ Admin SDK error: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Authentication error: {e}")
        return False

def provide_manual_instructions():
    """Provide manual configuration instructions."""
    print("\n" + "=" * 60)
    print("📋 MANUAL CONFIGURATION REQUIRED")
    print("=" * 60)
    print()
    print("🔧 Google doesn't provide a direct API for domain-wide delegation.")
    print("   You need to configure it manually in Google Workspace Admin Console.")
    print()
    print("📍 Follow these steps:")
    print()
    print("1️⃣  Go to Google Workspace Admin Console:")
    print("   https://admin.google.com")
    print()
    print("2️⃣  Navigate to Security → API Controls → Domain-wide Delegation")
    print()
    print("3️⃣  Click 'Add new' and enter:")
    print(f"   📋 Client ID: {SERVICE_ACCOUNT_CLIENT_ID}")
    print(f"   📋 OAuth Scopes: {OAUTH_SCOPES}")
    print("   📋 Description: PhotoSight service account for photo storage")
    print()
    print("4️⃣  Click 'Authorize'")
    print()
    print("5️⃣  Test the configuration:")
    print("   python test_delegation_upload.py")
    print()
    print("💡 Alternative: Use a service account with Shared Drives if available")

def main():
    print("🔧 PhotoSight Domain-Wide Delegation Configuration")
    print("=" * 60)
    
    # Check if we have admin access
    print("🔐 Checking Google Workspace Admin access...")
    
    if not check_workspace_admin_access():
        print("\n❌ Cannot configure domain-wide delegation programmatically")
        provide_manual_instructions()
        return 1
    
    print("\n🔄 Attempting programmatic configuration...")
    admin_service = authenticate_admin()
    
    if not admin_service:
        print("❌ Failed to authenticate with Admin SDK")
        provide_manual_instructions()
        return 1
    
    success = configure_domain_wide_delegation(admin_service)
    
    if success:
        print("✅ Domain-wide delegation configured successfully!")
        print("🧪 Run test: python test_delegation_upload.py")
        return 0
    else:
        provide_manual_instructions()
        return 1

if __name__ == "__main__":
    sys.exit(main())