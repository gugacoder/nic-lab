"""
GitLab-Specific Configuration Management

This module provides specialized configuration for GitLab integration including
authentication, connection pooling, rate limiting, and API-specific settings.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from pydantic import validator, Field
from pydantic_settings import BaseSettings

from ..utils.secrets import SecureToken, get_env_token, validate_gitlab_token

logger = logging.getLogger(__name__)


class GitLabTokenType(Enum):
    """GitLab token types"""
    PERSONAL_ACCESS_TOKEN = "personal_access_token"
    DEPLOY_TOKEN = "deploy_token"
    OAUTH_TOKEN = "oauth_token"
    JOB_TOKEN = "job_token"


class GitLabPermissionLevel(Enum):
    """GitLab permission levels"""
    GUEST = 10
    REPORTER = 20
    DEVELOPER = 30
    MAINTAINER = 40
    OWNER = 50


@dataclass
class GitLabInstanceConfig:
    """Configuration for a single GitLab instance"""
    
    name: str
    url: str
    token: SecureToken
    token_type: GitLabTokenType = GitLabTokenType.PERSONAL_ACCESS_TOKEN
    default_branch: str = "main"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True
    
    # Project access configuration
    accessible_projects: Optional[List[int]] = None
    accessible_groups: Optional[List[int]] = None
    search_scope: str = "projects"  # 'projects', 'groups', 'all'
    
    # Performance settings
    connection_pool_size: int = 10
    max_connections: int = 20
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not validate_gitlab_token(self.token):
            logger.warning(f"Token validation failed for instance {self.name}")
    
    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return (
            bool(self.name and self.url and self.token) and
            self.url.startswith(('http://', 'https://')) and
            self.token.is_valid
        )


class GitLabAuthConfig(BaseSettings):
    """Enhanced GitLab authentication configuration"""
    
    # Primary instance configuration
    primary_instance: Optional[str] = Field(default=None, description="Primary GitLab instance name")
    
    # Connection settings
    connect_timeout: int = Field(default=10, description="Connection timeout in seconds")
    read_timeout: int = Field(default=30, description="Read timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_backoff_factor: float = Field(default=0.3, description="Exponential backoff factor")
    retry_statuses: List[int] = Field(
        default=[429, 500, 502, 503, 504], 
        description="HTTP status codes to retry"
    )
    
    # Rate limiting
    requests_per_second: float = Field(default=10.0, description="Rate limit requests per second")
    burst_limit: int = Field(default=20, description="Burst request limit")
    
    # Security settings
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    check_permissions: bool = Field(default=True, description="Verify token permissions on connection")
    
    # Cache settings
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=1000, description="Maximum cache entries")
    
    # API settings
    api_version: str = Field(default="v4", description="GitLab API version")
    per_page: int = Field(default=100, description="Default pagination size")
    max_pages: int = Field(default=10, description="Maximum pages to fetch")
    
    class Config:
        env_prefix = "GITLAB_AUTH_"


class GitLabConfigManager:
    """Manager for GitLab configuration and multiple instances"""
    
    def __init__(self):
        self._instances: Dict[str, GitLabInstanceConfig] = {}
        self._auth_config = GitLabAuthConfig()
        self._primary_instance: Optional[str] = None
    
    def add_instance(
        self,
        name: str,
        url: str,
        token: Union[str, SecureToken],
        token_type: GitLabTokenType = GitLabTokenType.PERSONAL_ACCESS_TOKEN,
        **kwargs
    ) -> GitLabInstanceConfig:
        """Add a GitLab instance configuration
        
        Args:
            name: Instance identifier
            url: GitLab instance URL
            token: API token (string or SecureToken)
            token_type: Type of token
            **kwargs: Additional configuration options
            
        Returns:
            GitLabInstanceConfig instance
        """
        # Convert string token to SecureToken if needed
        if isinstance(token, str):
            token = SecureToken(token, f"{name}_token")
        
        config = GitLabInstanceConfig(
            name=name,
            url=url,
            token=token,
            token_type=token_type,
            **kwargs
        )
        
        if not config.is_valid:
            raise ValueError(f"Invalid configuration for GitLab instance '{name}'")
        
        self._instances[name] = config
        
        # Set as primary if it's the first instance or explicitly requested
        if not self._primary_instance or kwargs.get('primary', False):
            self._primary_instance = name
        
        logger.info(f"Added GitLab instance: {name} at {url}")
        return config
    
    def get_instance(self, name: Optional[str] = None) -> Optional[GitLabInstanceConfig]:
        """Get GitLab instance configuration
        
        Args:
            name: Instance name, uses primary if None
            
        Returns:
            GitLabInstanceConfig if found, None otherwise
        """
        if name is None:
            name = self._primary_instance
        
        if name is None:
            logger.warning("No GitLab instance specified and no primary instance set")
            return None
        
        return self._instances.get(name)
    
    def remove_instance(self, name: str) -> bool:
        """Remove a GitLab instance
        
        Args:
            name: Instance name
            
        Returns:
            True if removed, False if not found
        """
        if name in self._instances:
            config = self._instances.pop(name)
            logger.info(f"Removed GitLab instance: {config.name}")
            
            # Update primary if we removed it
            if self._primary_instance == name:
                self._primary_instance = next(iter(self._instances.keys()), None)
                if self._primary_instance:
                    logger.info(f"Updated primary instance to: {self._primary_instance}")
            
            return True
        return False
    
    def list_instances(self) -> List[str]:
        """List all configured instances"""
        return list(self._instances.keys())
    
    def get_primary_instance(self) -> Optional[GitLabInstanceConfig]:
        """Get the primary GitLab instance"""
        return self.get_instance()
    
    def set_primary_instance(self, name: str) -> bool:
        """Set the primary GitLab instance
        
        Args:
            name: Instance name
            
        Returns:
            True if set successfully, False if instance not found
        """
        if name in self._instances:
            self._primary_instance = name
            logger.info(f"Set primary GitLab instance to: {name}")
            return True
        return False
    
    @property
    def auth_config(self) -> GitLabAuthConfig:
        """Get authentication configuration"""
        return self._auth_config
    
    def load_from_environment(self) -> bool:
        """Load GitLab configuration from environment variables
        
        Returns:
            True if at least one instance was loaded successfully
        """
        loaded = False
        
        # Try to load primary instance from environment
        gitlab_url = os.getenv("GITLAB_URL")
        gitlab_token = get_env_token("GITLAB_PRIVATE_TOKEN", "gitlab_primary")
        
        if gitlab_url and gitlab_token:
            try:
                self.add_instance(
                    name="primary",
                    url=gitlab_url,
                    token=gitlab_token,
                    primary=True
                )
                loaded = True
                logger.info("Loaded primary GitLab instance from environment")
            except Exception as e:
                logger.error(f"Failed to load primary GitLab instance: {e}")
        
        # Try to load additional instances (GITLAB_INSTANCE_1_URL, etc.)
        instance_num = 1
        while True:
            url_var = f"GITLAB_INSTANCE_{instance_num}_URL"
            token_var = f"GITLAB_INSTANCE_{instance_num}_TOKEN"
            name_var = f"GITLAB_INSTANCE_{instance_num}_NAME"
            
            url = os.getenv(url_var)
            token = get_env_token(token_var, f"instance_{instance_num}")
            name = os.getenv(name_var, f"instance_{instance_num}")
            
            if not url or not token:
                break
            
            try:
                self.add_instance(
                    name=name,
                    url=url,
                    token=token
                )
                loaded = True
                logger.info(f"Loaded GitLab instance '{name}' from environment")
            except Exception as e:
                logger.error(f"Failed to load GitLab instance '{name}': {e}")
            
            instance_num += 1
        
        return loaded
    
    def validate_all_instances(self) -> Dict[str, bool]:
        """Validate all configured instances
        
        Returns:
            Dictionary mapping instance names to validation results
        """
        results = {}
        for name, config in self._instances.items():
            results[name] = config.is_valid
            if not config.is_valid:
                logger.warning(f"Instance '{name}' configuration is invalid")
        return results


# Global configuration manager
_config_manager: Optional[GitLabConfigManager] = None


def get_gitlab_config() -> GitLabConfigManager:
    """Get the global GitLab configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = GitLabConfigManager()
        # Try to load from environment
        if _config_manager.load_from_environment():
            logger.info("GitLab configuration loaded from environment")
        else:
            logger.warning("No GitLab configuration found in environment")
    return _config_manager


def reset_gitlab_config():
    """Reset the global GitLab configuration manager"""
    global _config_manager
    _config_manager = None


if __name__ == "__main__":
    # Test GitLab configuration
    print("Testing GitLab configuration...")
    
    # Create test configuration
    config_manager = GitLabConfigManager()
    
    # Add test instance
    test_token = SecureToken("glpat-test-token-12345678", "test")
    config = config_manager.add_instance(
        name="test",
        url="https://gitlab.example.com",
        token=test_token,
        primary=True
    )
    
    print(f"Added instance: {config.name}")
    print(f"Is valid: {config.is_valid}")
    
    # Test environment loading
    os.environ["GITLAB_URL"] = "https://gitlab.test.com"
    os.environ["GITLAB_PRIVATE_TOKEN"] = "glpat-environment-token"
    
    env_config = GitLabConfigManager()
    if env_config.load_from_environment():
        print("Environment configuration loaded successfully")
        primary = env_config.get_primary_instance()
        if primary:
            print(f"Primary instance: {primary.name} at {primary.url}")
    
    print("GitLab configuration testing complete")