"""
GitLab Authentication Management

This module handles GitLab authentication, token validation, permission checking,
and connection management with retry logic and proper error handling.
"""

import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

import gitlab
from gitlab.exceptions import GitlabAuthenticationError, GitlabError

from ..config.gitlab_config import (
    GitLabConfigManager, 
    GitLabInstanceConfig, 
    GitLabTokenType,
    GitLabPermissionLevel,
    get_gitlab_config
)
from ..utils.secrets import SecureToken, secure_logging

logger = logging.getLogger(__name__)


class AuthenticationStatus(Enum):
    """Authentication status types"""
    SUCCESS = "success"
    INVALID_TOKEN = "invalid_token"
    INSUFFICIENT_PERMISSIONS = "insufficient_permissions"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class AuthenticationResult:
    """Result of authentication attempt"""
    
    status: AuthenticationStatus
    instance_name: str
    user_info: Optional[Dict[str, Any]] = None
    permissions: Optional[List[GitLabPermissionLevel]] = None
    projects_count: Optional[int] = None
    groups_count: Optional[int] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None
    
    @property
    def is_success(self) -> bool:
        """Check if authentication was successful"""
        return self.status == AuthenticationStatus.SUCCESS
    
    @property
    def can_read(self) -> bool:
        """Check if token has read permissions"""
        return self.is_success and self.user_info is not None
    
    @property
    def can_write(self) -> bool:
        """Check if token has write permissions"""
        if not self.permissions:
            return False
        
        # Check if user has at least developer permissions on some projects
        return any(perm.value >= GitLabPermissionLevel.DEVELOPER.value 
                  for perm in self.permissions)


class GitLabAuthenticator:
    """GitLab authentication manager with retry logic and caching"""
    
    def __init__(self, config_manager: Optional[GitLabConfigManager] = None):
        """Initialize authenticator
        
        Args:
            config_manager: GitLab configuration manager, uses global if None
        """
        self.config_manager = config_manager or get_gitlab_config()
        self._connection_cache: Dict[str, gitlab.Gitlab] = {}
        self._auth_cache: Dict[str, AuthenticationResult] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_auth_time: Dict[str, float] = {}
    
    def authenticate(
        self, 
        instance_name: Optional[str] = None,
        force_refresh: bool = False
    ) -> AuthenticationResult:
        """Authenticate with GitLab instance
        
        Args:
            instance_name: Instance to authenticate with, uses primary if None
            force_refresh: Force refresh even if cached result is available
            
        Returns:
            AuthenticationResult with status and details
        """
        config = self.config_manager.get_instance(instance_name)
        if not config:
            return AuthenticationResult(
                status=AuthenticationStatus.UNKNOWN_ERROR,
                instance_name=instance_name or "unknown",
                error_message="GitLab instance configuration not found"
            )
        
        # Check cache if not forcing refresh
        if not force_refresh and self._is_auth_cached(config.name):
            cached_result = self._auth_cache[config.name]
            logger.debug(f"Using cached authentication for {config.name}")
            return cached_result
        
        with secure_logging():
            logger.info(f"Authenticating with GitLab instance: {config.name}")
            
            start_time = time.time()
            
            try:
                # Create GitLab connection
                gl = self._create_gitlab_connection(config)
                
                # Test authentication by getting current user
                user = gl.auth()
                user_info = {
                    'id': user.id,
                    'username': user.username,
                    'name': user.name,
                    'email': user.email,
                    'state': user.state,
                    'is_admin': getattr(user, 'is_admin', False)
                }
                
                # Get additional information
                projects = gl.projects.list(owned=True, all=True)
                groups = gl.groups.list(owned=True, all=True)
                
                # Check permissions on accessible projects
                permissions = self._check_permissions(gl, config)
                
                response_time = (time.time() - start_time) * 1000
                
                result = AuthenticationResult(
                    status=AuthenticationStatus.SUCCESS,
                    instance_name=config.name,
                    user_info=user_info,
                    permissions=permissions,
                    projects_count=len(projects),
                    groups_count=len(groups),
                    response_time_ms=response_time
                )
                
                # Cache successful authentication
                self._cache_auth_result(config.name, result)
                
                # Store connection for reuse
                self._connection_cache[config.name] = gl
                
                logger.info(
                    f"Authentication successful for {config.name}: "
                    f"user={user_info['username']}, "
                    f"projects={len(projects)}, "
                    f"groups={len(groups)}, "
                    f"time={response_time:.0f}ms"
                )
                
                return result
                
            except GitlabAuthenticationError as e:
                logger.error(f"Authentication failed for {config.name}: {e}")
                return AuthenticationResult(
                    status=AuthenticationStatus.INVALID_TOKEN,
                    instance_name=config.name,
                    error_message=str(e),
                    response_time_ms=(time.time() - start_time) * 1000
                )
                
            except Exception as e:
                logger.error(f"Connection error for {config.name}: {e}")
                return AuthenticationResult(
                    status=AuthenticationStatus.CONNECTION_ERROR,
                    instance_name=config.name,
                    error_message=str(e),
                    response_time_ms=(time.time() - start_time) * 1000
                )
    
    def _create_gitlab_connection(self, config: GitLabInstanceConfig) -> gitlab.Gitlab:
        """Create GitLab connection with proper configuration
        
        Args:
            config: GitLab instance configuration
            
        Returns:
            Configured GitLab client instance
        """
        auth_config = self.config_manager.auth_config
        
        return gitlab.Gitlab(
            url=config.url,
            private_token=config.token.value,
            timeout=config.timeout,
            retry_transient_errors=True,
            ssl_verify=config.verify_ssl,
            api_version=auth_config.api_version,
            per_page=auth_config.per_page
        )
    
    def _check_permissions(
        self, 
        gl: gitlab.Gitlab, 
        config: GitLabInstanceConfig
    ) -> List[GitLabPermissionLevel]:
        """Check user permissions on configured projects
        
        Args:
            gl: GitLab client instance
            config: Instance configuration
            
        Returns:
            List of permission levels found
        """
        permissions = []
        
        try:
            if config.accessible_projects:
                # Check specific projects
                for project_id in config.accessible_projects:
                    try:
                        project = gl.projects.get(project_id)
                        member = project.members.get(gl.user.id, lazy=True)
                        if member.access_level:
                            level = GitLabPermissionLevel(member.access_level)
                            permissions.append(level)
                    except Exception as e:
                        logger.debug(f"Could not check permissions for project {project_id}: {e}")
            else:
                # Check permissions on owned projects
                projects = gl.projects.list(owned=True, all=True)[:10]  # Limit for performance
                for project in projects:
                    try:
                        # Owner has full permissions
                        permissions.append(GitLabPermissionLevel.OWNER)
                        break  # One is enough to confirm write access
                    except Exception:
                        continue
        
        except Exception as e:
            logger.warning(f"Error checking permissions: {e}")
        
        return permissions
    
    def _is_auth_cached(self, instance_name: str) -> bool:
        """Check if authentication result is cached and valid
        
        Args:
            instance_name: Instance name
            
        Returns:
            True if valid cached result exists
        """
        if instance_name not in self._auth_cache:
            return False
        
        last_auth = self._last_auth_time.get(instance_name, 0)
        return (time.time() - last_auth) < self._cache_ttl
    
    def _cache_auth_result(self, instance_name: str, result: AuthenticationResult):
        """Cache authentication result
        
        Args:
            instance_name: Instance name
            result: Authentication result to cache
        """
        self._auth_cache[instance_name] = result
        self._last_auth_time[instance_name] = time.time()
    
    def get_connection(self, instance_name: Optional[str] = None) -> Optional[gitlab.Gitlab]:
        """Get cached GitLab connection
        
        Args:
            instance_name: Instance name, uses primary if None
            
        Returns:
            GitLab client instance if authenticated, None otherwise
        """
        config = self.config_manager.get_instance(instance_name)
        if not config:
            return None
        
        # Check if we have a cached connection
        if config.name in self._connection_cache:
            return self._connection_cache[config.name]
        
        # Try to authenticate and get connection
        result = self.authenticate(config.name)
        if result.is_success:
            return self._connection_cache.get(config.name)
        
        return None
    
    def test_connection(
        self, 
        instance_name: Optional[str] = None,
        timeout: int = 10
    ) -> Tuple[bool, str]:
        """Test GitLab connection with timeout
        
        Args:
            instance_name: Instance to test, uses primary if None
            timeout: Connection timeout in seconds
            
        Returns:
            Tuple of (success, message)
        """
        config = self.config_manager.get_instance(instance_name)
        if not config:
            return False, "GitLab instance configuration not found"
        
        try:
            # Create connection with specified timeout
            gl = gitlab.Gitlab(
                url=config.url,
                private_token=config.token.value,
                timeout=timeout,
                ssl_verify=config.verify_ssl
            )
            
            # Test authentication
            user = gl.auth()
            return True, f"Connection successful as {user.username}"
            
        except GitlabAuthenticationError:
            return False, "Authentication failed - invalid token"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def check_permissions(
        self, 
        instance_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check detailed permissions for GitLab instance
        
        Args:
            instance_name: Instance to check, uses primary if None
            
        Returns:
            Dictionary with permission details
        """
        result = self.authenticate(instance_name)
        
        if not result.is_success:
            return {
                'authenticated': False,
                'error': result.error_message,
                'can_read': False,
                'can_write': False
            }
        
        return {
            'authenticated': True,
            'user': result.user_info,
            'can_read': result.can_read,
            'can_write': result.can_write,
            'projects_count': result.projects_count,
            'groups_count': result.groups_count,
            'permissions': [p.name for p in (result.permissions or [])],
            'response_time_ms': result.response_time_ms
        }
    
    def clear_cache(self, instance_name: Optional[str] = None):
        """Clear authentication cache
        
        Args:
            instance_name: Instance to clear, clears all if None
        """
        if instance_name:
            self._auth_cache.pop(instance_name, None)
            self._last_auth_time.pop(instance_name, None)
            self._connection_cache.pop(instance_name, None)
            logger.info(f"Cleared auth cache for {instance_name}")
        else:
            self._auth_cache.clear()
            self._last_auth_time.clear()
            self._connection_cache.clear()
            logger.info("Cleared all auth cache")


# Global authenticator instance
_authenticator: Optional[GitLabAuthenticator] = None


def get_gitlab_authenticator() -> GitLabAuthenticator:
    """Get global GitLab authenticator instance"""
    global _authenticator
    if _authenticator is None:
        _authenticator = GitLabAuthenticator()
    return _authenticator


def quick_auth_check(instance_name: Optional[str] = None) -> bool:
    """Quick authentication check
    
    Args:
        instance_name: Instance to check, uses primary if None
        
    Returns:
        True if authenticated successfully
    """
    authenticator = get_gitlab_authenticator()
    result = authenticator.authenticate(instance_name)
    return result.is_success


if __name__ == "__main__":
    # Test authentication functionality
    import sys
    
    print("Testing GitLab authentication...")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test with environment configuration
        authenticator = get_gitlab_authenticator()
        result = authenticator.authenticate()
        
        print(f"Authentication status: {result.status.value}")
        if result.is_success:
            print(f"User: {result.user_info['username']}")
            print(f"Projects: {result.projects_count}")
            print(f"Can write: {result.can_write}")
            print(f"Response time: {result.response_time_ms:.0f}ms")
        else:
            print(f"Error: {result.error_message}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "check-permissions":
        # Check permissions
        authenticator = get_gitlab_authenticator()
        permissions = authenticator.check_permissions()
        
        print("Permission check results:")
        for key, value in permissions.items():
            print(f"  {key}: {value}")
    
    else:
        print("Usage:")
        print("  python -m src.integrations.gitlab_auth test")
        print("  python -m src.integrations.gitlab_auth check-permissions")
    
    print("GitLab authentication testing complete")