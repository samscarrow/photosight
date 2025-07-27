"""
Security configuration and validation for PhotoSight.

This module ensures that security-critical environment variables
are properly configured before the application starts, preventing
production deployments with insecure defaults.
"""

import os
import logging
import secrets
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    jwt_refresh_expiration_days: int = 30
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    csrf_enabled: bool = True
    secure_cookies: bool = True
    session_timeout_minutes: int = 60


class SecurityValidator:
    """Validates security configuration and environment setup."""
    
    # Critical environment variables that must be set in production
    CRITICAL_ENV_VARS = {
        'PHOTOSIGHT_SECRET_KEY': {
            'required': True,
            'min_length': 32,
            'description': 'Application secret key for JWT and session security'
        },
        'DATABASE_URL': {
            'required': True,
            'min_length': 10,
            'description': 'Database connection string'
        },
        'PHOTOSIGHT_ENVIRONMENT': {
            'required': True,
            'allowed_values': ['development', 'staging', 'production'],
            'description': 'Application environment'
        }
    }
    
    # Optional but recommended environment variables
    RECOMMENDED_ENV_VARS = {
        'REDIS_URL': {
            'description': 'Redis connection string for caching and rate limiting'
        },
        'SENTRY_DSN': {
            'description': 'Sentry DSN for error tracking'
        },
        'LOG_LEVEL': {
            'description': 'Logging level (DEBUG, INFO, WARNING, ERROR)',
            'default': 'INFO'
        }
    }
    
    # Insecure default values that must not be used in production
    INSECURE_DEFAULTS = [
        'dev-secret-key',
        'development-key',
        'test-key',
        'changeme',
        'secret',
        'password',
        '123456',
        'default'
    ]

    def __init__(self):
        self.environment = os.getenv('PHOTOSIGHT_ENVIRONMENT', 'development')
        self.is_production = self.environment == 'production'
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []

    def validate_environment(self) -> bool:
        """
        Validate the environment configuration.
        
        Returns:
            True if validation passes, False otherwise
        """
        logger.info(f"Validating security configuration for environment: {self.environment}")
        
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Validate critical environment variables
        self._validate_critical_env_vars()
        
        # Validate secret key security
        self._validate_secret_key()
        
        # Validate database configuration
        self._validate_database_config()
        
        # Check for recommended variables
        self._check_recommended_env_vars()
        
        # Production-specific validations
        if self.is_production:
            self._validate_production_requirements()
        
        # Log results
        self._log_validation_results()
        
        return len(self.validation_errors) == 0

    def _validate_critical_env_vars(self) -> None:
        """Validate that all critical environment variables are set."""
        for var_name, config in self.CRITICAL_ENV_VARS.items():
            value = os.getenv(var_name)
            
            if not value:
                if config['required']:
                    self.validation_errors.append(
                        f"Missing required environment variable: {var_name} "
                        f"({config['description']})"
                    )
                continue
            
            # Check minimum length
            if 'min_length' in config and len(value) < config['min_length']:
                self.validation_errors.append(
                    f"Environment variable {var_name} is too short "
                    f"(minimum {config['min_length']} characters)"
                )
            
            # Check allowed values
            if 'allowed_values' in config and value not in config['allowed_values']:
                self.validation_errors.append(
                    f"Environment variable {var_name} has invalid value '{value}'. "
                    f"Allowed values: {', '.join(config['allowed_values'])}"
                )

    def _validate_secret_key(self) -> None:
        """Validate the application secret key."""
        secret_key = os.getenv('PHOTOSIGHT_SECRET_KEY')
        
        if not secret_key:
            return  # Already handled in _validate_critical_env_vars
        
        # Check for insecure defaults
        if secret_key.lower() in [default.lower() for default in self.INSECURE_DEFAULTS]:
            self.validation_errors.append(
                f"PHOTOSIGHT_SECRET_KEY uses an insecure default value. "
                f"Generate a secure random key for production."
            )
        
        # Check entropy
        if len(set(secret_key)) < 10:
            self.validation_warnings.append(
                "PHOTOSIGHT_SECRET_KEY has low entropy. Consider using a more random key."
            )
        
        # Production-specific checks
        if self.is_production:
            if len(secret_key) < 64:
                self.validation_warnings.append(
                    "For production, PHOTOSIGHT_SECRET_KEY should be at least 64 characters long."
                )

    def _validate_database_config(self) -> None:
        """Validate database configuration."""
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            return  # Already handled in _validate_critical_env_vars
        
        # Check for insecure database configurations
        if 'password=' not in database_url and 'localhost' not in database_url:
            self.validation_warnings.append(
                "Database URL appears to be missing authentication. "
                "Ensure proper credentials are configured."
            )
        
        # Production-specific database checks
        if self.is_production:
            if 'localhost' in database_url or '127.0.0.1' in database_url:
                self.validation_warnings.append(
                    "Production environment is using localhost database. "
                    "Consider using a dedicated database server."
                )
            
            if 'ssl=false' in database_url.lower():
                self.validation_errors.append(
                    "Production database connections must use SSL encryption."
                )

    def _check_recommended_env_vars(self) -> None:
        """Check for recommended but optional environment variables."""
        for var_name, config in self.RECOMMENDED_ENV_VARS.items():
            value = os.getenv(var_name)
            
            if not value:
                self.validation_warnings.append(
                    f"Recommended environment variable not set: {var_name} "
                    f"({config['description']})"
                )

    def _validate_production_requirements(self) -> None:
        """Additional validations specific to production environment."""
        
        # Check for debug settings
        if os.getenv('FLASK_DEBUG', '').lower() in ['true', '1', 'yes']:
            self.validation_errors.append(
                "FLASK_DEBUG must not be enabled in production environment."
            )
        
        # Check for development-specific settings
        if os.getenv('PHOTOSIGHT_DEV_MODE', '').lower() in ['true', '1', 'yes']:
            self.validation_errors.append(
                "PHOTOSIGHT_DEV_MODE must not be enabled in production environment."
            )
        
        # Validate Redis configuration for production
        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            self.validation_errors.append(
                "REDIS_URL is required in production for rate limiting and caching."
            )

    def _log_validation_results(self) -> None:
        """Log validation results."""
        if self.validation_errors:
            logger.error("Security validation failed:")
            for error in self.validation_errors:
                logger.error(f"  ❌ {error}")
        
        if self.validation_warnings:
            logger.warning("Security validation warnings:")
            for warning in self.validation_warnings:
                logger.warning(f"  ⚠️  {warning}")
        
        if not self.validation_errors and not self.validation_warnings:
            logger.info("✅ Security validation passed with no issues.")

    def get_security_config(self) -> SecurityConfig:
        """
        Get security configuration based on environment variables.
        
        Returns:
            SecurityConfig instance with validated settings
        """
        secret_key = os.getenv('PHOTOSIGHT_SECRET_KEY')
        if not secret_key:
            raise ValueError("PHOTOSIGHT_SECRET_KEY environment variable is required")
        
        return SecurityConfig(
            secret_key=secret_key,
            jwt_algorithm=os.getenv('JWT_ALGORITHM', 'HS256'),
            jwt_expiration_hours=int(os.getenv('JWT_EXPIRATION_HOURS', '24')),
            jwt_refresh_expiration_days=int(os.getenv('JWT_REFRESH_EXPIRATION_DAYS', '30')),
            password_min_length=int(os.getenv('PASSWORD_MIN_LENGTH', '8')),
            max_login_attempts=int(os.getenv('MAX_LOGIN_ATTEMPTS', '5')),
            lockout_duration_minutes=int(os.getenv('LOCKOUT_DURATION_MINUTES', '15')),
            csrf_enabled=os.getenv('CSRF_ENABLED', 'true').lower() == 'true',
            secure_cookies=self.is_production,  # Always true in production
            session_timeout_minutes=int(os.getenv('SESSION_TIMEOUT_MINUTES', '60'))
        )


def validate_production_environment() -> bool:
    """
    Validate that the production environment is properly configured.
    
    This function should be called during application startup to ensure
    that all security-critical environment variables are properly set.
    
    Returns:
        True if validation passes, False otherwise
        
    Raises:
        SystemExit: If validation fails in production environment
    """
    validator = SecurityValidator()
    
    if not validator.validate_environment():
        if validator.is_production:
            logger.critical(
                "Security validation failed in production environment. "
                "Application startup aborted."
            )
            raise SystemExit(1)
        else:
            logger.warning(
                "Security validation failed in development environment. "
                "This would prevent startup in production."
            )
            return False
    
    return True


def generate_secure_secret_key() -> str:
    """
    Generate a cryptographically secure secret key.
    
    Returns:
        A 64-character random string suitable for use as PHOTOSIGHT_SECRET_KEY
    """
    return secrets.token_urlsafe(48)  # 48 bytes = 64 URL-safe characters


def get_security_headers() -> Dict[str, str]:
    """
    Get security headers that should be applied to all HTTP responses.
    
    Returns:
        Dictionary of security headers
    """
    return {
        # Prevent MIME type sniffing
        'X-Content-Type-Options': 'nosniff',
        
        # Enable XSS protection
        'X-XSS-Protection': '1; mode=block',
        
        # Prevent clickjacking
        'X-Frame-Options': 'DENY',
        
        # Enforce HTTPS in production
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains' if 
            os.getenv('PHOTOSIGHT_ENVIRONMENT') == 'production' else '',
        
        # Content Security Policy (basic - should be customized)
        'Content-Security-Policy': (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'"
        ),
        
        # Referrer policy
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        
        # Remove server information
        'Server': 'PhotoSight'
    }


def check_file_upload_security(filename: str, content_type: str, file_size: int) -> Dict[str, Any]:
    """
    Validate file upload security.
    
    Args:
        filename: Uploaded filename
        content_type: MIME content type
        file_size: File size in bytes
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []
    
    # Maximum file size (100MB default)
    max_file_size = int(os.getenv('MAX_UPLOAD_SIZE', str(100 * 1024 * 1024)))
    
    # Allowed extensions for photo files
    allowed_extensions = {
        '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp',
        '.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.orf', '.rw2'
    }
    
    # Check file size
    if file_size > max_file_size:
        issues.append(f"File size ({file_size:,} bytes) exceeds maximum allowed ({max_file_size:,} bytes)")
    
    # Check filename for path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        issues.append("Filename contains potentially dangerous path components")
    
    # Check for null bytes
    if '\x00' in filename:
        issues.append("Filename contains null bytes")
    
    # Check file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in allowed_extensions:
        issues.append(f"File extension '{file_ext}' is not allowed")
    
    # Check for double extensions
    if filename.count('.') > 1:
        warnings.append("Filename has multiple extensions - verify this is intentional")
    
    # Check content type alignment with extension
    expected_content_types = {
        '.jpg': ['image/jpeg'],
        '.jpeg': ['image/jpeg'],
        '.png': ['image/png'],
        '.tiff': ['image/tiff'],
        '.tif': ['image/tiff'],
        '.webp': ['image/webp']
        # RAW files often have application/octet-stream
    }
    
    if file_ext in expected_content_types:
        if content_type not in expected_content_types[file_ext]:
            warnings.append(f"Content type '{content_type}' doesn't match extension '{file_ext}'")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'sanitized_filename': Path(filename).name  # Remove any path components
    }


if __name__ == '__main__':
    # Command-line tool for testing security configuration
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--generate-key':
        print("Generated secure secret key:")
        print(generate_secure_secret_key())
        sys.exit(0)
    
    print("PhotoSight Security Configuration Validator")
    print("=" * 50)
    
    validator = SecurityValidator()
    success = validator.validate_environment()
    
    if success:
        print("\n✅ Security validation passed!")
        config = validator.get_security_config()
        print(f"\nSecurity configuration loaded for environment: {validator.environment}")
    else:
        print("\n❌ Security validation failed!")
        if validator.is_production:
            print("This would prevent application startup in production.")
        
        print("\nTo generate a secure secret key, run:")
        print("python -m photosight.config.security --generate-key")
    
    sys.exit(0 if success else 1)