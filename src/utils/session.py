"""
Session State Management Utilities

This module provides utilities for managing Streamlit session state,
ensuring consistent data persistence across application reruns.
"""

import streamlit as st
import uuid
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = ""  # 'user' | 'assistant' | 'system'
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'id': self.id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create message from dictionary"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            role=data.get('role', ''),
            content=data.get('content', ''),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            metadata=data.get('metadata', {})
        )


@dataclass
class SessionState:
    """Represents the application session state"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Chat state
    messages: List[ChatMessage] = field(default_factory=list)
    is_processing: bool = False
    error_message: Optional[str] = None
    
    # UI state
    sidebar_expanded: bool = True
    current_page: str = "chat"
    
    # Settings
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # GitLab state
    gitlab_connected: bool = False
    gitlab_projects: List[Dict[str, Any]] = field(default_factory=list)
    
    # Document state
    current_document: Optional[Dict[str, Any]] = None
    document_drafts: List[Dict[str, Any]] = field(default_factory=list)


class SessionManager:
    """Manages Streamlit session state"""
    
    SESSION_TIMEOUT_MINUTES = 60
    
    @staticmethod
    def initialize_session() -> None:
        """Initialize session state with default values"""
        if 'session_initialized' not in st.session_state:
            # Create new session
            session = SessionState()
            
            # Store session data in st.session_state
            st.session_state.session_id = session.session_id
            st.session_state.created_at = session.created_at
            st.session_state.last_activity = session.last_activity
            st.session_state.messages = session.messages
            st.session_state.is_processing = session.is_processing
            st.session_state.error_message = session.error_message
            st.session_state.sidebar_expanded = session.sidebar_expanded
            st.session_state.current_page = session.current_page
            st.session_state.user_preferences = session.user_preferences
            st.session_state.gitlab_connected = session.gitlab_connected
            st.session_state.gitlab_projects = session.gitlab_projects
            st.session_state.current_document = session.current_document
            st.session_state.document_drafts = session.document_drafts
            
            # Mark as initialized
            st.session_state.session_initialized = True
            
            logger.info(f"Initialized new session: {session.session_id}")
        
        # Update last activity
        SessionManager.update_activity()
    
    @staticmethod
    def update_activity() -> None:
        """Update last activity timestamp"""
        st.session_state.last_activity = datetime.now()
    
    @staticmethod
    def is_session_expired() -> bool:
        """Check if session has expired"""
        if 'last_activity' not in st.session_state:
            return True
        
        timeout = timedelta(minutes=SessionManager.SESSION_TIMEOUT_MINUTES)
        return datetime.now() - st.session_state.last_activity > timeout
    
    @staticmethod
    def clear_session() -> None:
        """Clear all session state"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        logger.info("Session cleared")
    
    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        """Get session information"""
        return {
            'session_id': st.session_state.get('session_id', 'unknown'),
            'created_at': st.session_state.get('created_at'),
            'last_activity': st.session_state.get('last_activity'),
            'message_count': len(st.session_state.get('messages', [])),
            'is_processing': st.session_state.get('is_processing', False),
            'gitlab_connected': st.session_state.get('gitlab_connected', False)
        }


class ChatStateManager:
    """Manages chat-specific state"""
    
    @staticmethod
    def add_message(role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """Add a new message to the chat"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        st.session_state.messages.append(message)
        SessionManager.update_activity()
        
        logger.debug(f"Added {role} message: {content[:50]}...")
        return message
    
    @staticmethod
    def get_messages() -> List[ChatMessage]:
        """Get all chat messages"""
        return st.session_state.get('messages', [])
    
    @staticmethod
    def clear_messages() -> None:
        """Clear all chat messages"""
        st.session_state.messages = []
        logger.info("Chat messages cleared")
    
    @staticmethod
    def get_conversation_context(max_messages: int = 10) -> str:
        """Get recent conversation context as text"""
        messages = ChatStateManager.get_messages()
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        context_parts = []
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def set_processing(processing: bool) -> None:
        """Set processing state"""
        st.session_state.is_processing = processing
        SessionManager.update_activity()
    
    @staticmethod
    def is_processing() -> bool:
        """Check if currently processing"""
        return st.session_state.get('is_processing', False)
    
    @staticmethod
    def set_error(error: Optional[str]) -> None:
        """Set error message"""
        st.session_state.error_message = error
        if error:
            logger.error(f"Chat error: {error}")
    
    @staticmethod
    def get_error() -> Optional[str]:
        """Get current error message"""
        return st.session_state.get('error_message')
    
    @staticmethod
    def clear_error() -> None:
        """Clear error message"""
        st.session_state.error_message = None


class UIStateManager:
    """Manages UI-specific state"""
    
    @staticmethod
    def set_page(page: str) -> None:
        """Set current page"""
        st.session_state.current_page = page
        SessionManager.update_activity()
    
    @staticmethod
    def get_page() -> str:
        """Get current page"""
        return st.session_state.get('current_page', 'chat')
    
    @staticmethod
    def set_sidebar_expanded(expanded: bool) -> None:
        """Set sidebar state"""
        st.session_state.sidebar_expanded = expanded
    
    @staticmethod
    def is_sidebar_expanded() -> bool:
        """Check if sidebar is expanded"""
        return st.session_state.get('sidebar_expanded', True)
    
    @staticmethod
    def set_preference(key: str, value: Any) -> None:
        """Set user preference"""
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
        st.session_state.user_preferences[key] = value
        SessionManager.update_activity()
    
    @staticmethod
    def get_preference(key: str, default: Any = None) -> Any:
        """Get user preference"""
        preferences = st.session_state.get('user_preferences', {})
        return preferences.get(key, default)


def init_session_state() -> None:
    """Initialize session state - main entry point"""
    SessionManager.initialize_session()


def with_session_management(func: Callable) -> Callable:
    """Decorator to ensure session management"""
    def wrapper(*args, **kwargs):
        init_session_state()
        return func(*args, **kwargs)
    return wrapper


def get_session_debug_info() -> Dict[str, Any]:
    """Get debug information about current session"""
    return {
        'session_info': SessionManager.get_session_info(),
        'state_keys': list(st.session_state.keys()),
        'message_count': len(st.session_state.get('messages', [])),
        'processing': st.session_state.get('is_processing', False),
        'error': st.session_state.get('error_message'),
        'preferences': st.session_state.get('user_preferences', {})
    }


if __name__ == "__main__":
    # Test session utilities (outside of Streamlit context)
    message = ChatMessage(role="user", content="Hello, world!")
    print(f"Message: {message.to_dict()}")
    
    # Test message reconstruction
    reconstructed = ChatMessage.from_dict(message.to_dict())
    print(f"Reconstructed: {reconstructed.content}")
    
    session = SessionState()
    print(f"Session ID: {session.session_id}")
    print(f"Created: {session.created_at}")