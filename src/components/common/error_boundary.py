"""
Error Boundary Components

This module provides error boundary functionality for Streamlit components,
catching and handling errors to prevent full application crashes while
providing recovery options to users.
"""

import streamlit as st
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager
from functools import wraps
import traceback

from src.utils.error_handler import (
    get_error_handler, ErrorCategory, ErrorSeverity, RecoveryAction
)
from src.utils.state_recovery import get_recovery_manager, create_error_snapshot


class ErrorBoundary:
    """Error boundary for Streamlit components"""
    
    def __init__(
        self,
        component_name: str,
        error_category: ErrorCategory = ErrorCategory.UI,
        fallback_content: Optional[str] = None,
        show_details: bool = False,
        auto_retry: bool = True,
        auto_snapshot: bool = True
    ):
        self.component_name = component_name
        self.error_category = error_category
        self.fallback_content = fallback_content
        self.show_details = show_details
        self.auto_retry = auto_retry
        self.auto_snapshot = auto_snapshot
        self.logger = logging.getLogger(__name__)
        
        # Initialize error tracking
        if f"error_boundary_{component_name}" not in st.session_state:
            st.session_state[f"error_boundary_{component_name}"] = {
                "error_count": 0,
                "last_error": None,
                "recovery_attempts": 0,
                "is_disabled": False
            }
    
    @contextmanager
    def catch_errors(self):
        """Context manager for catching component errors"""
        state_key = f"error_boundary_{self.component_name}"
        error_state = st.session_state[state_key]
        
        try:
            # Check if component is disabled due to repeated errors
            if error_state["is_disabled"]:
                self._render_disabled_component()
                return
            
            yield
            
            # Reset error count on successful execution
            if error_state["error_count"] > 0:
                error_state["error_count"] = 0
                error_state["recovery_attempts"] = 0
                self.logger.info(f"Component '{self.component_name}' recovered successfully")
            
        except Exception as e:
            self._handle_component_error(e, error_state)
    
    def _handle_component_error(self, error: Exception, error_state: Dict[str, Any]) -> None:
        """Handle an error within the component boundary"""
        try:
            # Create error snapshot if enabled
            snapshot_id = None
            if self.auto_snapshot:
                snapshot_id = create_error_snapshot(f"boundary_{self.component_name}")
            
            # Increment error count
            error_state["error_count"] += 1
            error_state["last_error"] = str(error)
            
            # Determine error severity based on frequency
            if error_state["error_count"] >= 5:
                severity = ErrorSeverity.HIGH
                error_state["is_disabled"] = True
            elif error_state["error_count"] >= 3:
                severity = ErrorSeverity.MEDIUM
            else:
                severity = ErrorSeverity.LOW
            
            # Handle error through central system
            error_handler = get_error_handler()
            error_info = error_handler.handle_error(
                error=error,
                category=self.error_category,
                severity=severity,
                user_message=f"Error in {self.component_name} component",
                context={
                    "component": self.component_name,
                    "error_count": error_state["error_count"],
                    "snapshot_id": snapshot_id
                }
            )
            
            # Render error UI
            self._render_error_ui(error_info, error_state)
            
            # Log the error
            self.logger.error(
                f"Error boundary caught error in '{self.component_name}': {error}",
                exc_info=True
            )
            
        except Exception as nested_error:
            # Fallback error handling if error boundary itself fails
            self.logger.critical(
                f"Error boundary failed while handling error: {nested_error}",
                exc_info=True
            )
            st.error("Critical error occurred. Please refresh the page.")
    
    def _render_error_ui(self, error_info, error_state: Dict[str, Any]) -> None:
        """Render error UI within the boundary"""
        
        # Create error container
        with st.container():
            # Error message
            if error_state["error_count"] >= 5:
                st.error(
                    f"🚫 **{self.component_name} Disabled**\n\n"
                    "This component has been disabled due to repeated errors. "
                    "Please refresh the page or contact support."
                )
            else:
                st.error(
                    f"❌ **Error in {self.component_name}**\n\n"
                    f"{error_info.user_message}"
                )
            
            # Recovery actions
            self._render_recovery_actions(error_info, error_state)
            
            # Error details (if enabled)
            if self.show_details or st.session_state.get("debug_mode", False):
                with st.expander("🔍 Error Details", expanded=False):
                    st.code(error_info.technical_details, language="python")
                    
                    if error_info.context:
                        st.json(error_info.context)
            
            # Fallback content
            if self.fallback_content and error_state["error_count"] < 3:
                st.info(f"ℹ️ **Fallback Mode**: {self.fallback_content}")
    
    def _render_recovery_actions(self, error_info, error_state: Dict[str, Any]) -> None:
        """Render recovery action buttons"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(
                "🔄 Retry",
                key=f"retry_{self.component_name}_{error_state['error_count']}",
                disabled=error_state["is_disabled"],
                help="Try to run the component again"
            ):
                self._handle_retry(error_state)
        
        with col2:
            if st.button(
                "🔧 Recover",
                key=f"recover_{self.component_name}_{error_state['error_count']}",
                help="Attempt automatic recovery"
            ):
                self._handle_recovery(error_info, error_state)
        
        with col3:
            if st.button(
                "🗑️ Reset",
                key=f"reset_{self.component_name}_{error_state['error_count']}",
                help="Reset component state"
            ):
                self._handle_reset(error_state)
        
        with col4:
            if st.button(
                "ℹ️ Details",
                key=f"details_{self.component_name}_{error_state['error_count']}",
                help="Show error details"
            ):
                st.session_state["debug_mode"] = not st.session_state.get("debug_mode", False)
                st.rerun()
    
    def _handle_retry(self, error_state: Dict[str, Any]) -> None:
        """Handle retry action"""
        error_state["recovery_attempts"] += 1
        error_state["error_count"] = max(0, error_state["error_count"] - 1)
        
        self.logger.info(f"Retrying component '{self.component_name}'")
        st.rerun()
    
    def _handle_recovery(self, error_info, error_state: Dict[str, Any]) -> None:
        """Handle automatic recovery"""
        try:
            error_handler = get_error_handler()
            recovery_manager = get_recovery_manager()
            
            # Try state recovery first
            if recovery_manager.recover_critical_state():
                st.success("🎉 State recovered successfully!")
                error_state["error_count"] = 0
                error_state["recovery_attempts"] = 0
                error_state["is_disabled"] = False
                st.rerun()
                return
            
            # Try other recovery actions
            for action in error_info.recovery_actions:
                if error_handler.execute_recovery_action(error_info, action):
                    st.success(f"🎉 Recovery action '{action.value}' succeeded!")
                    error_state["error_count"] = max(0, error_state["error_count"] - 2)
                    error_state["recovery_attempts"] += 1
                    st.rerun()
                    return
            
            st.warning("⚠️ Automatic recovery failed. Try manual actions.")
            
        except Exception as e:
            self.logger.error(f"Recovery failed for '{self.component_name}': {e}")
            st.error(f"❌ Recovery failed: {str(e)}")
    
    def _handle_reset(self, error_state: Dict[str, Any]) -> None:
        """Handle component reset"""
        # Reset error state
        error_state["error_count"] = 0
        error_state["last_error"] = None
        error_state["recovery_attempts"] = 0
        error_state["is_disabled"] = False
        
        # Clear component-specific session state
        keys_to_clear = [
            key for key in st.session_state.keys()
            if self.component_name.lower() in key.lower()
        ]
        
        for key in keys_to_clear:
            if key != f"error_boundary_{self.component_name}":
                del st.session_state[key]
        
        self.logger.info(f"Reset component '{self.component_name}'")
        st.success(f"🔄 {self.component_name} component reset")
        st.rerun()
    
    def _render_disabled_component(self) -> None:
        """Render UI for disabled component"""
        st.error(
            f"🚫 **{self.component_name} Disabled**\n\n"
            "This component has been disabled due to repeated errors."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(
                "🔄 Force Enable",
                key=f"force_enable_{self.component_name}",
                help="Force re-enable this component"
            ):
                state_key = f"error_boundary_{self.component_name}"
                st.session_state[state_key]["is_disabled"] = False
                st.session_state[state_key]["error_count"] = 0
                st.rerun()
        
        with col2:
            if st.button(
                "📞 Report Issue",
                key=f"report_{self.component_name}",
                help="Report this issue"
            ):
                self._show_issue_report()
    
    def _show_issue_report(self) -> None:
        """Show issue reporting interface"""
        st.info(
            f"📞 **Report Issue**\n\n"
            f"Component: {self.component_name}\n"
            f"Error: {st.session_state[f'error_boundary_{self.component_name}']['last_error']}\n\n"
            "Please provide this information to your system administrator."
        )


def error_boundary(
    component_name: str,
    error_category: ErrorCategory = ErrorCategory.UI,
    fallback_content: Optional[str] = None,
    show_details: bool = False,
    auto_retry: bool = True,
    auto_snapshot: bool = True
):
    """
    Decorator for wrapping functions with error boundary
    
    Args:
        component_name: Name of the component for error tracking
        error_category: Category of errors to expect
        fallback_content: Content to show in fallback mode
        show_details: Whether to show error details by default
        auto_retry: Whether to show retry option
        auto_snapshot: Whether to create snapshots on errors
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            boundary = ErrorBoundary(
                component_name=component_name,
                error_category=error_category,
                fallback_content=fallback_content,
                show_details=show_details,
                auto_retry=auto_retry,
                auto_snapshot=auto_snapshot
            )
            
            with boundary.catch_errors():
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class SafeContainer:
    """Safe container that wraps content in error boundaries"""
    
    def __init__(self, container_name: str):
        self.container_name = container_name
        self.boundary = ErrorBoundary(
            component_name=container_name,
            error_category=ErrorCategory.UI,
            fallback_content="Container temporarily unavailable",
            show_details=False,
            auto_retry=True,
            auto_snapshot=True
        )
    
    @contextmanager
    def render(self):
        """Context manager for safe rendering"""
        with self.boundary.catch_errors():
            with st.container():
                yield


def safe_component_render(
    render_func: Callable,
    component_name: str,
    fallback_func: Optional[Callable] = None,
    **kwargs
) -> Any:
    """
    Safely render a component with error boundary
    
    Args:
        render_func: Function to render the component
        component_name: Name for error tracking
        fallback_func: Optional fallback rendering function
        **kwargs: Additional arguments for ErrorBoundary
        
    Returns:
        Result of render_func or None if error occurred
    """
    boundary = ErrorBoundary(
        component_name=component_name,
        **kwargs
    )
    
    try:
        with boundary.catch_errors():
            return render_func()
    except Exception:
        # Error was handled by boundary, try fallback if available
        if fallback_func:
            try:
                return fallback_func()
            except Exception as e:
                logging.getLogger(__name__).error(f"Fallback function also failed: {e}")
        return None


def get_error_boundary_status(component_name: str) -> Dict[str, Any]:
    """Get error boundary status for a component"""
    state_key = f"error_boundary_{component_name}"
    
    if state_key not in st.session_state:
        return {
            "exists": False,
            "error_count": 0,
            "is_disabled": False
        }
    
    error_state = st.session_state[state_key]
    
    return {
        "exists": True,
        "error_count": error_state.get("error_count", 0),
        "last_error": error_state.get("last_error"),
        "recovery_attempts": error_state.get("recovery_attempts", 0),
        "is_disabled": error_state.get("is_disabled", False)
    }


def reset_all_error_boundaries() -> int:
    """Reset all error boundaries in the application"""
    reset_count = 0
    
    for key in list(st.session_state.keys()):
        if key.startswith("error_boundary_"):
            st.session_state[key] = {
                "error_count": 0,
                "last_error": None,
                "recovery_attempts": 0,
                "is_disabled": False
            }
            reset_count += 1
    
    return reset_count