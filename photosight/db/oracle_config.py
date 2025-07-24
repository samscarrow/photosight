"""
Oracle Database Configuration for PhotoSight
Updated: 2025-07-24 - Using verified working configuration
"""

import os
import oracledb
from pathlib import Path
from dotenv import load_dotenv

# Load Oracle environment
env_path = Path(__file__).parent.parent.parent / '.env.oracle'
load_dotenv(env_path)

class OracleConfig:
    """Oracle database configuration using verified working settings"""
    
    # Connection parameters
    USER = os.getenv('ORACLE_USER', 'ADMIN')
    PASSWORD = os.getenv('ORACLE_PASSWORD', 'hdv!gwm6DPQ.ycu.jua')
    TNS_NAME = os.getenv('ORACLE_TNS_NAME', 'photosightdb_high')
    
    # Wallet configuration
    WALLET_PATH = os.getenv('ORACLE_WALLET_PATH', '/Users/sam/oracle-wallets/photosight')
    TNS_ADMIN = os.getenv('TNS_ADMIN', '/Users/sam/oracle-wallets/photosight')
    
    # Oracle Instant Client
    CLIENT_PATH = os.getenv('ORACLE_CLIENT_PATH', '/opt/oracle/instantclient_19_16')
    
    # SQLAlchemy URL
    DATABASE_URL = os.getenv('PHOTOSIGHT_DATABASE_URL', 
                            f'oracle+oracledb://{USER}:{PASSWORD}@{TNS_NAME}')
    
    @classmethod
    def init_oracle_client(cls):
        """Initialize Oracle client in thick mode for wallet support"""
        try:
            oracledb.init_oracle_client(
                lib_dir=cls.CLIENT_PATH,
                config_dir=cls.TNS_ADMIN
            )
            return True
        except Exception as e:
            print(f"Warning: Could not initialize Oracle thick mode: {e}")
            return False
    
    @classmethod
    def get_connection(cls):
        """Get Oracle database connection using wallet"""
        # Ensure thick mode is initialized
        if not hasattr(cls, '_thick_mode_initialized'):
            cls._thick_mode_initialized = cls.init_oracle_client()
        
        # Ensure TNS_ADMIN is set in environment
        os.environ['TNS_ADMIN'] = cls.TNS_ADMIN
        
        return oracledb.connect(
            user=cls.USER,
            password=cls.PASSWORD,
            dsn=cls.TNS_NAME,
            config_dir=cls.TNS_ADMIN,
            wallet_location=cls.WALLET_PATH,
            wallet_password=cls.PASSWORD
        )
    
    @classmethod
    def get_sqlalchemy_engine(cls):
        """Get SQLAlchemy engine for Oracle"""
        from sqlalchemy import create_engine
        
        # Initialize thick mode first
        cls.init_oracle_client()
        
        # Create engine with thick mode parameters
        return create_engine(
            cls.DATABASE_URL,
            thick_mode={
                "lib_dir": cls.CLIENT_PATH,
                "config_dir": cls.TNS_ADMIN
            }
        )