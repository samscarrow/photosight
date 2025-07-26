#!/usr/bin/env python3
"""
Comprehensive API Test Suite for PhotoSight

Tests all API endpoints, WebSocket functionality, and integration scenarios.
"""

import pytest
import asyncio
import json
import io
from pathlib import Path
from datetime import datetime
import tempfile
import numpy as np
from PIL import Image
import socketio
from unittest.mock import Mock, patch, MagicMock

# Import API components
from photosight.api.app import create_app
from photosight.api.models_pydantic import (
    ProcessingRecipe, Photo, BatchJob, Session,
    User, LoginRequest, TokenResponse
)
from photosight.api.auth import create_access_token, verify_token
from photosight.api.celery_app import celery_app
from photosight.db import get_session, init_db


# Test fixtures
@pytest.fixture
def app():
    """Create test Flask app"""
    app = create_app()
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret-key'
    app.config['JWT_SECRET_KEY'] = 'test-jwt-secret'
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def auth_headers():
    """Create authenticated headers"""
    token = create_access_token(user_id=1, email='test@example.com')
    return {'Authorization': f'Bearer {token}'}


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    img = Image.new('RGB', (100, 100), color='red')
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    return img_io


@pytest.fixture
def sample_raw_file():
    """Create a mock RAW file"""
    # In real tests, use actual RAW file
    with tempfile.NamedTemporaryFile(suffix='.CR2', delete=False) as f:
        f.write(b'MOCK_RAW_DATA')
        return f.name


class TestAuthenticationAPI:
    """Test authentication endpoints"""
    
    def test_register_user(self, client):
        """Test user registration"""
        response = client.post('/api/v1/auth/register', json={
            'email': 'newuser@example.com',
            'password': 'SecurePass123!',
            'name': 'Test User'
        })
        
        assert response.status_code == 201
        data = response.get_json()
        assert data['status'] == 'success'
        assert 'user_id' in data['data']
    
    def test_login_valid_credentials(self, client):
        """Test login with valid credentials"""
        # First register
        client.post('/api/v1/auth/register', json={
            'email': 'testlogin@example.com',
            'password': 'TestPass123!'
        })
        
        # Then login
        response = client.post('/api/v1/auth/login', json={
            'email': 'testlogin@example.com',
            'password': 'TestPass123!'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'access_token' in data['data']
        assert 'refresh_token' in data['data']
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        response = client.post('/api/v1/auth/login', json={
            'email': 'nonexistent@example.com',
            'password': 'WrongPass'
        })
        
        assert response.status_code == 401
    
    def test_token_refresh(self, client):
        """Test token refresh"""
        # Login first
        login_response = client.post('/api/v1/auth/login', json={
            'email': 'test@example.com',
            'password': 'password'
        })
        refresh_token = login_response.get_json()['data']['refresh_token']
        
        # Refresh token
        response = client.post('/api/v1/auth/refresh', json={
            'refresh_token': refresh_token
        })
        
        assert response.status_code == 200
        assert 'access_token' in response.get_json()['data']


class TestPhotoAPI:
    """Test photo management endpoints"""
    
    def test_upload_photo(self, client, auth_headers, sample_image):
        """Test photo upload"""
        response = client.post(
            '/api/v1/photos/upload',
            headers=auth_headers,
            data={
                'file': (sample_image, 'test.jpg'),
                'auto_process': 'true'
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 201
        data = response.get_json()
        assert data['data']['filename'] == 'test.jpg'
        assert 'photo_id' in data['data']
    
    def test_upload_raw_photo(self, client, auth_headers, sample_raw_file):
        """Test RAW photo upload"""
        with open(sample_raw_file, 'rb') as f:
            response = client.post(
                '/api/v1/photos/upload',
                headers=auth_headers,
                data={
                    'file': (f, 'test.CR2'),
                    'project_id': 'test-project'
                },
                content_type='multipart/form-data'
            )
        
        assert response.status_code == 201
        data = response.get_json()
        assert data['data']['filename'] == 'test.CR2'
    
    def test_list_photos(self, client, auth_headers):
        """Test listing photos"""
        response = client.get('/api/v1/photos', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'photos' in data['data']
        assert 'total' in data['data']
    
    def test_get_photo_details(self, client, auth_headers):
        """Test getting photo details"""
        # Upload a photo first
        upload_response = client.post(
            '/api/v1/photos/upload',
            headers=auth_headers,
            data={'file': (sample_image, 'test.jpg')},
            content_type='multipart/form-data'
        )
        photo_id = upload_response.get_json()['data']['photo_id']
        
        # Get details
        response = client.get(f'/api/v1/photos/{photo_id}', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['data']['photo_id'] == photo_id
    
    def test_delete_photo(self, client, auth_headers):
        """Test photo deletion"""
        # Upload a photo first
        upload_response = client.post(
            '/api/v1/photos/upload',
            headers=auth_headers,
            data={'file': (sample_image, 'test.jpg')},
            content_type='multipart/form-data'
        )
        photo_id = upload_response.get_json()['data']['photo_id']
        
        # Delete it
        response = client.delete(f'/api/v1/photos/{photo_id}', headers=auth_headers)
        
        assert response.status_code == 200
        
        # Verify it's gone
        get_response = client.get(f'/api/v1/photos/{photo_id}', headers=auth_headers)
        assert get_response.status_code == 404


class TestProcessingAPI:
    """Test photo processing endpoints"""
    
    def test_process_photo_with_recipe(self, client, auth_headers):
        """Test processing with custom recipe"""
        # Mock photo exists
        photo_id = 'test-photo-123'
        
        recipe = {
            'exposure': 0.5,
            'contrast': 10,
            'vibrance': 15,
            'white_balance': {
                'temperature': 5600,
                'tint': 5
            }
        }
        
        response = client.post(
            f'/api/v1/photos/{photo_id}/process',
            headers=auth_headers,
            json={'recipe': recipe}
        )
        
        assert response.status_code == 202
        data = response.get_json()
        assert 'task_id' in data['data']
    
    def test_apply_preset(self, client, auth_headers):
        """Test applying processing preset"""
        photo_id = 'test-photo-123'
        
        response = client.post(
            f'/api/v1/photos/{photo_id}/process',
            headers=auth_headers,
            json={'recipe': {'preset': 'wedding_classic'}}
        )
        
        assert response.status_code == 202
    
    def test_get_processing_status(self, client, auth_headers):
        """Test getting processing status"""
        task_id = 'test-task-123'
        
        response = client.get(
            f'/api/v1/processing/status/{task_id}',
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'status' in data['data']


class TestBatchAPI:
    """Test batch processing endpoints"""
    
    def test_create_batch_job(self, client, auth_headers):
        """Test creating batch processing job"""
        batch_data = {
            'photo_ids': ['photo1', 'photo2', 'photo3'],
            'recipe': {
                'exposure': 0.3,
                'vibrance': 10
            },
            'priority': 'high'
        }
        
        response = client.post(
            '/api/v1/batch',
            headers=auth_headers,
            json=batch_data
        )
        
        assert response.status_code == 202
        data = response.get_json()
        assert 'job_id' in data['data']
    
    def test_get_batch_status(self, client, auth_headers):
        """Test getting batch job status"""
        job_id = 'test-batch-123'
        
        response = client.get(
            f'/api/v1/batch/{job_id}',
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'status' in data['data']
        assert 'progress' in data['data']
    
    def test_cancel_batch_job(self, client, auth_headers):
        """Test canceling batch job"""
        job_id = 'test-batch-123'
        
        response = client.delete(
            f'/api/v1/batch/{job_id}',
            headers=auth_headers
        )
        
        assert response.status_code == 200


class TestSessionAPI:
    """Test editing session endpoints"""
    
    def test_create_session(self, client, auth_headers):
        """Test creating editing session"""
        response = client.post(
            '/api/v1/sessions',
            headers=auth_headers,
            json={'photo_id': 'test-photo-123'}
        )
        
        assert response.status_code == 201
        data = response.get_json()
        assert 'session_id' in data['data']
        assert 'preview_url' in data['data']
    
    def test_update_session(self, client, auth_headers):
        """Test updating session with new recipe"""
        session_id = 'test-session-123'
        
        update_data = {
            'recipe': {
                'exposure': 0.7,
                'shadows': 25
            }
        }
        
        response = client.put(
            f'/api/v1/sessions/{session_id}',
            headers=auth_headers,
            json=update_data
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'preview_url' in data['data']
    
    def test_session_undo(self, client, auth_headers):
        """Test undo operation in session"""
        session_id = 'test-session-123'
        
        response = client.post(
            f'/api/v1/sessions/{session_id}/undo',
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    def test_save_session(self, client, auth_headers):
        """Test saving session changes"""
        session_id = 'test-session-123'
        
        response = client.post(
            f'/api/v1/sessions/{session_id}/save',
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['data']['saved'] == True


class TestExportAPI:
    """Test export endpoints"""
    
    def test_export_photo(self, client, auth_headers):
        """Test exporting processed photo"""
        photo_id = 'test-photo-123'
        
        export_data = {
            'format': 'jpeg',
            'quality': 95,
            'resize': {
                'enabled': True,
                'width': 2048
            }
        }
        
        response = client.post(
            f'/api/v1/photos/{photo_id}/export',
            headers=auth_headers,
            json=export_data
        )
        
        assert response.status_code == 202
        data = response.get_json()
        assert 'export_id' in data['data']
    
    def test_download_export(self, client, auth_headers):
        """Test downloading exported file"""
        export_id = 'test-export-123'
        
        response = client.get(
            f'/api/v1/exports/{export_id}/download',
            headers=auth_headers
        )
        
        # Should redirect or return file
        assert response.status_code in [200, 302]


class TestSystemAPI:
    """Test system monitoring endpoints"""
    
    def test_get_queue_status(self, client, auth_headers):
        """Test getting queue status"""
        response = client.get(
            '/api/v1/system/queues',
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'queues' in data['data']
    
    def test_get_system_stats(self, client, auth_headers):
        """Test getting system statistics"""
        response = client.get(
            '/api/v1/system/stats',
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'processing_stats' in data['data']
        assert 'storage_stats' in data['data']


class TestWebSocketAPI:
    """Test WebSocket functionality"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, app):
        """Test WebSocket connection"""
        client = socketio.AsyncClient()
        
        await client.connect(
            'http://localhost:5000',
            auth={'token': 'test-token'}
        )
        
        assert client.connected
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_join_session_room(self, app):
        """Test joining session room"""
        client = socketio.AsyncClient()
        
        await client.connect(
            'http://localhost:5000',
            auth={'token': 'test-token'}
        )
        
        # Join session
        await client.emit('join_session', {
            'session_id': 'test-session-123'
        })
        
        # Should receive confirmation
        received = []
        
        @client.on('session_joined')
        def on_join(data):
            received.append(data)
        
        await asyncio.sleep(0.1)
        assert len(received) > 0
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_preview_updates(self, app):
        """Test receiving preview updates"""
        client = socketio.AsyncClient()
        updates = []
        
        @client.on('preview_update')
        def on_update(data):
            updates.append(data)
        
        await client.connect(
            'http://localhost:5000',
            auth={'token': 'test-token'}
        )
        
        # Simulate preview update
        await client.emit('update_preview', {
            'session_id': 'test-session-123',
            'recipe': {'exposure': 0.5}
        })
        
        await asyncio.sleep(0.1)
        assert len(updates) > 0
        assert 'preview_url' in updates[0]
        
        await client.disconnect()


class TestChunkedUploadAPI:
    """Test chunked upload functionality"""
    
    def test_init_chunked_upload(self, client, auth_headers):
        """Test initializing chunked upload"""
        response = client.post(
            '/api/v1/upload/chunked/init',
            headers=auth_headers,
            json={
                'filename': 'large_photo.CR2',
                'file_size': 50 * 1024 * 1024,  # 50MB
                'chunk_size': 5 * 1024 * 1024    # 5MB chunks
            }
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'upload_id' in data['data']
        assert 'total_chunks' in data['data']
    
    def test_upload_chunk(self, client, auth_headers):
        """Test uploading a chunk"""
        upload_id = 'test-upload-123'
        chunk_data = b'x' * (5 * 1024 * 1024)  # 5MB chunk
        
        response = client.post(
            f'/api/v1/upload/chunked/{upload_id}/chunk/0',
            headers=auth_headers,
            data=chunk_data,
            content_type='application/octet-stream'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['data']['chunk_received'] == 0
    
    def test_complete_chunked_upload(self, client, auth_headers):
        """Test completing chunked upload"""
        upload_id = 'test-upload-123'
        
        response = client.post(
            f'/api/v1/upload/chunked/{upload_id}/complete',
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'photo_id' in data['data']
    
    def test_resume_chunked_upload(self, client, auth_headers):
        """Test resuming interrupted upload"""
        upload_id = 'test-upload-123'
        
        response = client.get(
            f'/api/v1/upload/chunked/{upload_id}/status',
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'received_chunks' in data['data']
        assert 'missing_chunks' in data['data']


class TestValidation:
    """Test input validation"""
    
    def test_invalid_recipe_parameters(self, client, auth_headers):
        """Test validation of recipe parameters"""
        invalid_recipe = {
            'exposure': 10,  # Out of range
            'contrast': 200,  # Out of range
            'invalid_param': 'test'
        }
        
        response = client.post(
            '/api/v1/photos/test-photo/process',
            headers=auth_headers,
            json={'recipe': invalid_recipe}
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error_details' in data
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test missing required fields"""
        response = client.post(
            '/api/v1/batch',
            headers=auth_headers,
            json={'recipe': {}}  # Missing photo_ids
        )
        
        assert response.status_code == 400


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_404_not_found(self, client, auth_headers):
        """Test 404 responses"""
        response = client.get(
            '/api/v1/photos/nonexistent-photo',
            headers=auth_headers
        )
        
        assert response.status_code == 404
        data = response.get_json()
        assert data['status'] == 'error'
        assert 'message' in data
    
    def test_unauthorized_access(self, client):
        """Test unauthorized access"""
        response = client.get('/api/v1/photos')
        
        assert response.status_code == 401
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting"""
        # Make many requests quickly
        for _ in range(105):  # Assuming 100 req/min limit
            response = client.get('/api/v1/photos', headers=auth_headers)
        
        # Should get rate limited
        assert response.status_code == 429
        assert 'Retry-After' in response.headers


class TestIntegration:
    """Test integration scenarios"""
    
    @pytest.mark.integration
    def test_full_workflow(self, client, auth_headers, sample_raw_file):
        """Test complete workflow from upload to export"""
        # 1. Upload RAW photo
        with open(sample_raw_file, 'rb') as f:
            upload_response = client.post(
                '/api/v1/photos/upload',
                headers=auth_headers,
                data={
                    'file': (f, 'wedding_001.CR2'),
                    'auto_process': 'false'
                },
                content_type='multipart/form-data'
            )
        
        assert upload_response.status_code == 201
        photo_id = upload_response.get_json()['data']['photo_id']
        
        # 2. Create editing session
        session_response = client.post(
            '/api/v1/sessions',
            headers=auth_headers,
            json={'photo_id': photo_id}
        )
        
        assert session_response.status_code == 201
        session_id = session_response.get_json()['data']['session_id']
        
        # 3. Apply adjustments
        update_response = client.put(
            f'/api/v1/sessions/{session_id}',
            headers=auth_headers,
            json={
                'recipe': {
                    'preset': 'wedding_classic',
                    'exposure': 0.3,
                    'vibrance': 15
                }
            }
        )
        
        assert update_response.status_code == 200
        
        # 4. Save session
        save_response = client.post(
            f'/api/v1/sessions/{session_id}/save',
            headers=auth_headers
        )
        
        assert save_response.status_code == 200
        
        # 5. Export final image
        export_response = client.post(
            f'/api/v1/photos/{photo_id}/export',
            headers=auth_headers,
            json={
                'format': 'jpeg',
                'quality': 95,
                'resize': {'enabled': True, 'width': 2048}
            }
        )
        
        assert export_response.status_code == 202
        export_id = export_response.get_json()['data']['export_id']
        
        # 6. Check export status
        status_response = client.get(
            f'/api/v1/exports/{export_id}/status',
            headers=auth_headers
        )
        
        assert status_response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])