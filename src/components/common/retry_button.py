"""
Retry Button Component

This module provides reusable retry button components with various styles and
functionality for handling failed operations across the NIC Chat system.
"""

import streamlit as st
from typing import Optional, Callable, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import time

from src.utils.error_handler import get_error_handler
from src.utils.state_recovery import get_recovery_manager


class RetryButtonStyle(Enum):
    """Retry button styles"""
    STANDARD = "standard"
    COMPACT = "compact"
    ICON_ONLY = "icon_only"
    TEXT_ONLY = "text_only"
    INLINE = "inline"


class RetryButtonState(Enum):
    """Retry button states"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    LOADING = "loading"
    COOLDOWN = "cooldown"


class RetryButton:
    """Reusable retry button component with various configurations"""
    
    def __init__(
        self,
        operation_id: str,
        retry_callback: Callable[[], bool],
        max_retries: int = 3,
        cooldown_seconds: int = 0,
        style: RetryButtonStyle = RetryButtonStyle.STANDARD,
        show_attempt_count: bool = True
    ):
        """
        Initialize retry button
        
        Args:
            operation_id: Unique identifier for the operation
            retry_callback: Function to call when retry is clicked (should return True on success)
            max_retries: Maximum number of retry attempts
            cooldown_seconds: Cooldown between retry attempts
            style: Visual style of the button
            show_attempt_count: Whether to show attempt count
        """
        self.operation_id = operation_id
        self.retry_callback = retry_callback
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self.style = style
        self.show_attempt_count = show_attempt_count
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state for this retry button
        self._initialize_state()
    
    def render(
        self,
        label: Optional[str] = None,
        help_text: Optional[str] = None,
        disabled: bool = False,
        key: Optional[str] = None
    ) -> bool:
        """
        Render the retry button
        
        Args:
            label: Custom button label
            help_text: Help text for the button
            disabled: Whether the button is disabled
            key: Streamlit widget key
            
        Returns:
            True if retry was clicked and callback succeeded
        """
        # Determine current state
        current_state = self._get_current_state()
        
        # Generate button properties
        button_props = self._get_button_properties(
            current_state, label, help_text, disabled
        )
        
        # Use provided key or generate one
        button_key = key or f"retry_btn_{self.operation_id}"
        
        # Render based on style
        if self.style == RetryButtonStyle.STANDARD:
            return self._render_standard_button(button_props, button_key)
        elif self.style == RetryButtonStyle.COMPACT:
            return self._render_compact_button(button_props, button_key)
        elif self.style == RetryButtonStyle.ICON_ONLY:
            return self._render_icon_button(button_props, button_key)
        elif self.style == RetryButtonStyle.TEXT_ONLY:
            return self._render_text_button(button_props, button_key)
        elif self.style == RetryButtonStyle.INLINE:
            return self._render_inline_button(button_props, button_key)
        
        return False
    
    def _initialize_state(self) -> None:
        """Initialize session state for this retry button"""
        state_key = f"retry_button_{self.operation_id}"
        
        if state_key not in st.session_state:
            st.session_state[state_key] = {
                "attempt_count": 0,
                "last_attempt": None,
                "is_loading": False,
                "last_success": None
            }
    
    def _get_current_state(self) -> RetryButtonState:
        """Determine the current state of the retry button"""
        state_key = f"retry_button_{self.operation_id}"
        button_state = st.session_state[state_key]
        
        # Check if loading
        if button_state["is_loading"]:
            return RetryButtonState.LOADING
        
        # Check if max retries exceeded
        if button_state["attempt_count"] >= self.max_retries:
            return RetryButtonState.DISABLED
        
        # Check cooldown
        if self.cooldown_seconds > 0 and button_state["last_attempt"]:
            time_since_last = datetime.now() - button_state["last_attempt"]
            if time_since_last.total_seconds() < self.cooldown_seconds:
                return RetryButtonState.COOLDOWN
        
        return RetryButtonState.ENABLED
    
    def _get_button_properties(
        self,
        current_state: RetryButtonState,
        label: Optional[str],
        help_text: Optional[str],
        disabled: bool
    ) -> Dict[str, Any]:
        """Generate button properties based on current state"""
        state_key = f"retry_button_{self.operation_id}"
        button_state = st.session_state[state_key]
        
        # Base properties
        props = {
            "disabled": disabled,
            "help": help_text or "Retry the failed operation"
        }
        
        # State-specific properties
        if current_state == RetryButtonState.ENABLED:
            props["label"] = label or "🔄 Retry"
            props["disabled"] = disabled
            
        elif current_state == RetryButtonState.LOADING:
            props["label"] = "⏳ Retrying..."
            props["disabled"] = True
            props["help"] = "Retry in progress..."
            
        elif current_state == RetryButtonState.DISABLED:
            props["label"] = f"❌ Max retries ({self.max_retries})"
            props["disabled"] = True
            props["help"] = f"Maximum retry attempts ({self.max_retries}) exceeded"
            
        elif current_state == RetryButtonState.COOLDOWN:
            remaining = self._get_cooldown_remaining()
            props["label"] = f"⏱️ Wait {remaining}s"
            props["disabled"] = True
            props["help"] = f"Please wait {remaining} seconds before retrying"
        
        # Add attempt count if enabled
        if self.show_attempt_count and button_state["attempt_count"] > 0:
            attempt_text = f" ({button_state['attempt_count']}/{self.max_retries})"
            if "Max retries" not in props["label"]:
                props["label"] += attempt_text
        
        return props
    
    def _get_cooldown_remaining(self) -> int:
        """Get remaining cooldown time in seconds"""
        state_key = f"retry_button_{self.operation_id}"
        button_state = st.session_state[state_key]
        
        if not button_state["last_attempt"]:
            return 0
        
        elapsed = (datetime.now() - button_state["last_attempt"]).total_seconds()
        remaining = max(0, self.cooldown_seconds - elapsed)
        return int(remaining)
    
    def _render_standard_button(self, props: Dict[str, Any], key: str) -> bool:
        """Render standard retry button"""
        if st.button(
            props["label"],
            help=props["help"],
            disabled=props["disabled"],
            key=key
        ):
            return self._handle_retry_click()
        return False
    
    def _render_compact_button(self, props: Dict[str, Any], key: str) -> bool:
        """Render compact retry button"""
        # Use smaller button with minimal styling
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button(
                props["label"],
                help=props["help"],
                disabled=props["disabled"],
                key=key
            ):
                return self._handle_retry_click()
        return False
    
    def _render_icon_button(self, props: Dict[str, Any], key: str) -> bool:
        """Render icon-only retry button"""
        # Extract just the icon from the label
        icon = props["label"].split()[0] if props["label"] else "🔄"
        
        if st.button(
            icon,
            help=props["help"],
            disabled=props["disabled"],
            key=key
        ):
            return self._handle_retry_click()
        return False
    
    def _render_text_button(self, props: Dict[str, Any], key: str) -> bool:
        """Render text-only retry button"""
        # Remove icons from label
        text_only = props["label"]
        for icon in ["🔄", "⏳", "❌", "⏱️"]:
            text_only = text_only.replace(icon, "").strip()
        
        if st.button(
            text_only,
            help=props["help"],
            disabled=props["disabled"],
            key=key
        ):
            return self._handle_retry_click()
        return False
    
    def _render_inline_button(self, props: Dict[str, Any], key: str) -> bool:
        """Render inline retry button"""
        # Use markdown with button-like styling
        if not props["disabled"]:
            if st.button(
                props["label"],
                help=props["help"],
                key=key
            ):
                return self._handle_retry_click()
        else:
            # Show disabled state as text
            st.markdown(f"*{props['label']}*")
            
        return False
    
    def _handle_retry_click(self) -> bool:
        """Handle retry button click"""
        state_key = f"retry_button_{self.operation_id}"
        button_state = st.session_state[state_key]
        
        try:
            # Set loading state
            button_state["is_loading"] = True
            button_state["last_attempt"] = datetime.now()
            button_state["attempt_count"] += 1
            
            # Force a rerun to show loading state
            st.rerun()
            
            # Call the retry callback
            success = self.retry_callback()
            
            # Update state based on result
            if success:
                button_state["last_success"] = datetime.now()
                button_state["attempt_count"] = 0  # Reset on success
                self.logger.info(f"Retry successful for operation: {self.operation_id}")
            else:
                self.logger.warning(f"Retry failed for operation: {self.operation_id}")
            
            button_state["is_loading"] = False
            return success
            
        except Exception as e:
            # Handle callback errors
            button_state["is_loading"] = False
            self.logger.error(f"Retry callback error for {self.operation_id}: {e}")
            return False
    
    def reset(self) -> None:
        """Reset the retry button state"""
        state_key = f"retry_button_{self.operation_id}"
        
        if state_key in st.session_state:
            st.session_state[state_key] = {
                "attempt_count": 0,
                "last_attempt": None,
                "is_loading": False,
                "last_success": None
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retry statistics for this button"""
        state_key = f"retry_button_{self.operation_id}"
        button_state = st.session_state.get(state_key, {})
        
        return {
            "operation_id": self.operation_id,
            "attempt_count": button_state.get("attempt_count", 0),
            "max_retries": self.max_retries,
            "last_attempt": button_state.get("last_attempt"),
            "last_success": button_state.get("last_success"),
            "is_loading": button_state.get("is_loading", False),
            "current_state": self._get_current_state().value
        }


# Convenience functions for common retry scenarios

def render_operation_retry_button(
    operation_name: str,
    retry_callback: Callable[[], bool],
    max_retries: int = 3,
    style: RetryButtonStyle = RetryButtonStyle.STANDARD,
    key: Optional[str] = None
) -> bool:
    """
    Render a retry button for a generic operation
    
    Args:
        operation_name: Name of the operation (used for ID and labeling)
        retry_callback: Function to call on retry
        max_retries: Maximum retry attempts
        style: Button style
        key: Streamlit widget key
        
    Returns:
        True if retry succeeded
    """
    retry_button = RetryButton(
        operation_id=f"operation_{operation_name}",
        retry_callback=retry_callback,
        max_retries=max_retries,
        style=style
    )
    
    return retry_button.render(
        label=f"🔄 Retry {operation_name.title()}",
        help_text=f"Retry the {operation_name} operation",
        key=key
    )


def render_api_retry_button(
    api_name: str,
    retry_callback: Callable[[], bool],
    cooldown_seconds: int = 2,
    key: Optional[str] = None
) -> bool:
    """
    Render a retry button for API calls with cooldown
    
    Args:
        api_name: Name of the API
        retry_callback: Function to call on retry
        cooldown_seconds: Cooldown between attempts
        key: Streamlit widget key
        
    Returns:
        True if retry succeeded
    """
    retry_button = RetryButton(
        operation_id=f"api_{api_name}",
        retry_callback=retry_callback,
        max_retries=5,
        cooldown_seconds=cooldown_seconds,
        style=RetryButtonStyle.STANDARD
    )
    
    return retry_button.render(
        label=f"🔄 Retry {api_name} API",
        help_text=f"Retry the {api_name} API call",
        key=key
    )


def render_message_retry_button(
    message_id: str,
    retry_callback: Callable[[], bool],
    key: Optional[str] = None
) -> bool:
    """
    Render a retry button for failed message operations
    
    Args:
        message_id: ID of the message
        retry_callback: Function to call on retry
        key: Streamlit widget key
        
    Returns:
        True if retry succeeded
    """
    retry_button = RetryButton(
        operation_id=f"message_{message_id}",
        retry_callback=retry_callback,
        max_retries=3,
        style=RetryButtonStyle.COMPACT,
        show_attempt_count=False
    )
    
    return retry_button.render(
        label="🔄 Retry",
        help_text="Retry sending this message",
        key=key
    )


def render_inline_retry_link(
    operation_id: str,
    retry_callback: Callable[[], bool],
    key: Optional[str] = None
) -> bool:
    """
    Render an inline retry link for subtle retry functionality
    
    Args:
        operation_id: ID of the operation
        retry_callback: Function to call on retry
        key: Streamlit widget key
        
    Returns:
        True if retry succeeded
    """
    retry_button = RetryButton(
        operation_id=operation_id,
        retry_callback=retry_callback,
        max_retries=2,
        style=RetryButtonStyle.INLINE,
        show_attempt_count=False
    )
    
    return retry_button.render(
        label="retry",
        help_text="Click to retry",
        key=key
    )


def get_all_retry_statistics() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all retry buttons in the session"""
    stats = {}
    
    for key, value in st.session_state.items():
        if key.startswith("retry_button_") and isinstance(value, dict):
            operation_id = key.replace("retry_button_", "")
            stats[operation_id] = {
                "operation_id": operation_id,
                "attempt_count": value.get("attempt_count", 0),
                "last_attempt": value.get("last_attempt"),
                "last_success": value.get("last_success"),
                "is_loading": value.get("is_loading", False)
            }
    
    return stats


def reset_all_retry_buttons() -> int:
    """Reset all retry buttons in the session"""
    reset_count = 0
    
    for key in list(st.session_state.keys()):
        if key.startswith("retry_button_"):
            st.session_state[key] = {
                "attempt_count": 0,
                "last_attempt": None,
                "is_loading": False,
                "last_success": None
            }
            reset_count += 1
    
    return reset_count