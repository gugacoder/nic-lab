"""
Secure Token Handling Utilities

This module provides secure utilities for handling sensitive information like API tokens
and credentials, ensuring they are never exposed in logs or error messages.
"""

import os
import logging
import hashlib
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SecureToken:
    """Secure wrapper for sensitive tokens and credentials"""
    
    def __init__(self, token: str, name: str = "token"):
        """Initialize secure token wrapper
        
        Args:
            token: The actual token value
            name: Human-readable name for logging purposes
        """
        self._token = token
        self._name = name
        self._hash = self._calculate_hash(token)
    
    def _calculate_hash(self, token: str) -> str:
        """Calculate SHA-256 hash of token for safe logging"""
        return hashlib.sha256(token.encode()).hexdigest()[:8]
    
    @property
    def value(self) -> str:
        """Get the actual token value"""
        return self._token
    
    @property
    def safe_repr(self) -> str:
        """Get safe representation for logging"""
        if len(self._token) < 8:
            return f"{self._name}:***"
        return f"{self._name}:{self._token[:4]}***{self._hash}"
    
    @property
    def is_valid(self) -> bool:
        """Check if token appears to be valid (not empty)"""
        return bool(self._token and self._token.strip())
    
    def __str__(self) -> str:
        """String representation (safe for logging)"""
        return self.safe_repr
    
    def __repr__(self) -> str:
        """Representation (safe for debugging)"""
        return f"SecureToken({self.safe_repr})"
    
    def __eq__(self, other) -> bool:
        """Compare tokens securely"""
        if isinstance(other, SecureToken):
            return self._token == other._token
        return False
    
    def __bool__(self) -> bool:
        """Boolean check based on validity"""
        return self.is_valid


class TokenManager:
    """Manager for secure token storage and retrieval"""
    
    def __init__(self):
        self._tokens: Dict[str, SecureToken] = {}
    
    def add_token(self, name: str, token: str) -> SecureToken:
        """Add a token to secure storage
        
        Args:
            name: Token identifier
            token: Token value
            
        Returns:
            SecureToken instance
        """
        secure_token = SecureToken(token, name)
        self._tokens[name] = secure_token
        logger.info(f"Added token: {secure_token.safe_repr}")
        return secure_token
    
    def get_token(self, name: str) -> Optional[SecureToken]:
        """Retrieve a token by name
        
        Args:
            name: Token identifier
            
        Returns:
            SecureToken if found, None otherwise
        """
        return self._tokens.get(name)
    
    def remove_token(self, name: str) -> bool:
        """Remove a token from storage
        
        Args:
            name: Token identifier
            
        Returns:
            True if token was removed, False if not found
        """
        if name in self._tokens:
            token = self._tokens.pop(name)
            logger.info(f"Removed token: {token.safe_repr}")
            return True
        return False
    
    def clear_all(self):
        """Clear all stored tokens"""
        count = len(self._tokens)
        self._tokens.clear()
        logger.info(f"Cleared {count} tokens from secure storage")
    
    def list_tokens(self) -> Dict[str, str]:
        """List all stored tokens (safe representations only)
        
        Returns:
            Dictionary mapping token names to safe representations
        """
        return {name: token.safe_repr for name, token in self._tokens.items()}


def get_env_token(env_var: str, token_name: Optional[str] = None) -> Optional[SecureToken]:
    """Safely retrieve a token from environment variables
    
    Args:
        env_var: Environment variable name
        token_name: Optional human-readable name for the token
        
    Returns:
        SecureToken if found and valid, None otherwise
    """
    token_value = os.getenv(env_var)
    
    if not token_value:
        logger.warning(f"Environment variable {env_var} not found or empty")
        return None
    
    token_value = token_value.strip()
    if not token_value:
        logger.warning(f"Environment variable {env_var} is empty after stripping")
        return None
    
    name = token_name or env_var.lower()
    secure_token = SecureToken(token_value, name)
    
    logger.info(f"Retrieved token from environment: {secure_token.safe_repr}")
    return secure_token


def validate_gitlab_token(token: SecureToken) -> bool:
    """Validate GitLab token format
    
    Args:
        token: SecureToken instance
        
    Returns:
        True if token appears to be valid GitLab token format
    """
    if not token or not token.is_valid:
        return False
    
    token_value = token.value
    
    # GitLab personal access tokens are typically 20-26 characters
    # and contain alphanumeric characters and hyphens
    if len(token_value) < 20 or len(token_value) > 30:
        logger.warning(f"Token {token.safe_repr} has unusual length: {len(token_value)}")
        return False
    
    # Check for valid characters (alphanumeric, hyphens, underscores)
    if not all(c.isalnum() or c in '-_' for c in token_value):
        logger.warning(f"Token {token.safe_repr} contains invalid characters")
        return False
    
    return True


@contextmanager
def secure_logging():
    """Context manager to ensure no sensitive data is logged"""
    original_formatters = {}
    
    try:
        # Store original formatters
        for handler in logging.getLogger().handlers:
            original_formatters[handler] = handler.formatter
            
            # Create secure formatter that filters sensitive data
            class SecureFormatter(logging.Formatter):
                def format(self, record):
                    # Filter out potential tokens from log messages
                    if hasattr(record, 'msg') and isinstance(record.msg, str):
                        msg = record.msg
                        # Remove potential tokens (20+ character alphanumeric strings)
                        import re
                        msg = re.sub(r'\b[a-zA-Z0-9_-]{20,}\b', '[TOKEN]', msg)
                        record.msg = msg
                    
                    return super().format(record)
            
            handler.setFormatter(SecureFormatter(handler.formatter._fmt if handler.formatter else None))
        
        yield
        
    finally:
        # Restore original formatters
        for handler, formatter in original_formatters.items():
            handler.setFormatter(formatter)


# Global token manager instance
_token_manager: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    """Get global token manager instance"""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager


def clear_token_cache():
    """Clear the global token manager cache"""
    global _token_manager
    if _token_manager:
        _token_manager.clear_all()
    _token_manager = None


if __name__ == "__main__":
    # Test secure token handling
    print("Testing secure token handling...")
    
    # Test token creation
    test_token = SecureToken("glpat-xxxxxxxxxxxxxxxxxxxx", "test_gitlab_token")
    print(f"Token: {test_token}")
    print(f"Valid: {test_token.is_valid}")
    print(f"Validation: {validate_gitlab_token(test_token)}")
    
    # Test token manager
    manager = get_token_manager()
    manager.add_token("gitlab", "glpat-test-token-12345678")
    
    retrieved = manager.get_token("gitlab")
    if retrieved:
        print(f"Retrieved: {retrieved}")
    
    print(f"All tokens: {manager.list_tokens()}")
    
    # Test environment token retrieval
    os.environ["TEST_TOKEN"] = "test-token-value"
    env_token = get_env_token("TEST_TOKEN", "test")
    if env_token:
        print(f"Environment token: {env_token}")
    
    print("Secure token testing complete")