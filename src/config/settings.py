"""
NIC Chat System Configuration Management

This module provides centralized configuration management for the NIC Chat application,
supporting environment variables, validation, and extensible settings for all components.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from pydantic import validator, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamlitSettings(BaseSettings):
    """Streamlit-specific configuration settings"""
    
    page_title: str = Field(default="NIC Chat", description="Application page title")
    page_icon: str = Field(default="💬", description="Application page icon")
    layout: str = Field(default="wide", description="Streamlit page layout")
    initial_sidebar_state: str = Field(default="expanded", description="Initial sidebar state")
    
    # Server Configuration
    server_port: int = Field(default=8000, description="Default server port")
    server_address: str = Field(default="0.0.0.0", description="Server address")
    
    # UI Configuration
    max_messages_display: int = Field(default=100, description="Maximum messages to display")
    message_chunk_size: int = Field(default=50, description="Messages loaded per chunk")
    auto_scroll: bool = Field(default=True, description="Auto-scroll to new messages")
    
    class Config:
        env_prefix = "STREAMLIT_"


class GitLabSettings(BaseSettings):
    """GitLab integration configuration"""
    
    url: Optional[str] = Field(default=None, description="GitLab instance URL")
    private_token: Optional[str] = Field(default=None, description="GitLab API token")
    default_branch: str = Field(default="main", description="Default branch for operations")
    timeout: int = Field(default=30, description="API request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum API retry attempts")
    
    # Authentication settings
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    check_permissions: bool = Field(default=True, description="Verify token permissions on connection")
    cache_auth_results: bool = Field(default=True, description="Cache authentication results")
    auth_cache_ttl: int = Field(default=300, description="Authentication cache TTL in seconds")
    
    # Connection settings
    connect_timeout: int = Field(default=10, description="Connection timeout in seconds")
    read_timeout: int = Field(default=30, description="Read timeout in seconds")
    retry_backoff_factor: float = Field(default=0.3, description="Exponential backoff factor")
    connection_pool_size: int = Field(default=10, description="HTTP connection pool size")
    
    # Rate limiting
    requests_per_second: float = Field(default=10.0, description="Rate limit requests per second")
    burst_limit: int = Field(default=20, description="Burst request limit")
    
    # Search configuration
    max_search_results: int = Field(default=50, description="Maximum search results")
    search_timeout: int = Field(default=10, description="Search timeout in seconds")
    search_file_extensions: str = Field(
        default="md,txt,py,js,ts,json,yaml,yml",
        description="Default file extensions to search (comma-separated)"
    )
    
    # API settings
    api_version: str = Field(default="v4", description="GitLab API version")
    per_page: int = Field(default=100, description="Default pagination size")
    max_pages: int = Field(default=10, description="Maximum pages to fetch")
    
    # Project access
    accessible_projects: Optional[List[int]] = Field(default=None, description="Specific project IDs to access")
    accessible_groups: Optional[List[int]] = Field(default=None, description="Specific group IDs to access")
    search_scope: str = Field(default="projects", description="Search scope: projects, groups, or all")
    
    @validator('url')
    def validate_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('GitLab URL must start with http:// or https://')
        return v
    
    @validator('search_scope')
    def validate_search_scope(cls, v):
        valid_scopes = ['projects', 'groups', 'all']
        if v not in valid_scopes:
            raise ValueError(f'Search scope must be one of: {valid_scopes}')
        return v
    
    @validator('search_file_extensions')
    def validate_file_extensions(cls, v):
        if isinstance(v, str):
            # Parse comma-separated string
            extensions = [ext.strip().lstrip('.').lower() for ext in v.split(',') if ext.strip()]
        elif isinstance(v, list):
            # Handle list input
            extensions = [ext.lstrip('.').lower() for ext in v]
        else:
            extensions = []
        return extensions
    
    class Config:
        env_prefix = "GITLAB_"


class GroqSettings(BaseSettings):
    """Groq API configuration"""
    
    api_key: Optional[str] = Field(default=None, description="Groq API key")
    model: str = Field(default="llama-3.1-8b-instant", description="Default model")
    max_tokens: int = Field(default=4096, description="Maximum tokens per response")
    temperature: float = Field(default=0.7, description="Response creativity")
    timeout: int = Field(default=30, description="API timeout in seconds")
    
    # Rate limiting
    requests_per_minute: int = Field(default=30, description="Rate limit")
    
    class Config:
        env_prefix = "GROQ_"


class DocumentSettings(BaseSettings):
    """Document generation configuration"""
    
    default_format: str = Field(default="docx", description="Default document format")
    max_document_size_mb: int = Field(default=50, description="Maximum document size")
    image_quality: int = Field(default=85, description="Image compression quality")
    
    # Template configuration
    template_directory: str = Field(default="templates", description="Template directory")
    default_template: str = Field(default="default", description="Default template name")
    
    class Config:
        env_prefix = "DOCUMENT_"


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size_mb: int = Field(default=10, description="Max log file size")
    backup_count: int = Field(default=5, description="Number of backup files")
    
    class Config:
        env_prefix = "LOG_"


class AppSettings(BaseSettings):
    """Main application settings"""
    
    # Application metadata
    app_name: str = Field(default="NIC Chat", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Environment
    environment: str = Field(default="development", description="Deployment environment")
    
    # Component settings
    streamlit: StreamlitSettings = StreamlitSettings()
    gitlab: GitLabSettings = GitLabSettings()
    groq: GroqSettings = GroqSettings()
    document: DocumentSettings = DocumentSettings()
    logging: LoggingSettings = LoggingSettings()
    
    # Session configuration
    session_timeout_minutes: int = Field(default=60, description="Session timeout")
    max_concurrent_sessions: int = Field(default=100, description="Max concurrent sessions")
    
    # Security
    secret_key: Optional[str] = Field(default=None, description="Application secret key")
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_environments = ['development', 'testing', 'staging', 'production']
        if v not in valid_environments:
            raise ValueError(f'Environment must be one of: {valid_environments}')
        return v
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure application logging"""
        log_level = getattr(logging, self.logging.level.upper())
        logging.basicConfig(
            level=log_level,
            format=self.logging.format
        )
        
        if self.logging.file_path:
            # Add file handler if specified
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            logging.getLogger().addHandler(file_handler)
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled"""
        return self.debug and not self.is_production()
    
    def get_gitlab_config(self) -> Dict[str, Any]:
        """Get GitLab configuration as dictionary"""
        return self.gitlab.dict()
    
    def get_groq_config(self) -> Dict[str, Any]:
        """Get Groq configuration as dictionary"""
        return self.groq.dict()
    
    def validate_required_settings(self) -> List[str]:
        """Validate that required settings are present"""
        missing = []
        
        # Check GitLab configuration if not in development
        if not self.is_debug():
            if not self.gitlab.url:
                missing.append("GITLAB_URL")
            if not self.gitlab.private_token:
                missing.append("GITLAB_PRIVATE_TOKEN")
        
        # Check Groq configuration
        if not self.groq.api_key:
            missing.append("GROQ_API_KEY")
        
        return missing
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra environment variables


# Global settings instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get application settings singleton"""
    global _settings
    if _settings is None:
        _settings = AppSettings()
        logger.info(f"Initialized NIC Chat settings for environment: {_settings.environment}")
        
        # Validate required settings
        missing = _settings.validate_required_settings()
        if missing:
            logger.warning(f"Missing required configuration: {', '.join(missing)}")
    
    return _settings


def reload_settings() -> AppSettings:
    """Reload settings from environment"""
    global _settings
    _settings = None
    return get_settings()


# Convenience functions
def is_production() -> bool:
    """Check if running in production"""
    return get_settings().is_production()


def is_debug() -> bool:
    """Check if debug mode is enabled"""
    return get_settings().is_debug()


# Export main settings for easy importing
settings = get_settings()

if __name__ == "__main__":
    # Test configuration loading
    config = get_settings()
    print(f"App: {config.app_name} v{config.version}")
    print(f"Environment: {config.environment}")
    print(f"Debug: {config.is_debug()}")
    print(f"Streamlit Title: {config.streamlit.page_title}")
    
    # Check missing configuration
    missing = config.validate_required_settings()
    if missing:
        print(f"Missing configuration: {', '.join(missing)}")
    else:
        print("All required configuration present")