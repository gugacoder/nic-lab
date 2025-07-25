"""
Error Display Components

This module provides various error display components for the NIC Chat system,
offering user-friendly error messages, contextual information, and recovery options.
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum

from src.utils.error_handler import ErrorInfo, ErrorSeverity, ErrorCategory, RecoveryAction


class ErrorDisplayStyle(Enum):
    """Error display styles"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    INLINE = "inline"
    TOAST = "toast"


class ErrorDisplay:
    """Main error display component"""
    
    @staticmethod
    def render_error(
        error_info: ErrorInfo,
        style: ErrorDisplayStyle = ErrorDisplayStyle.STANDARD,
        show_recovery_actions: bool = True,
        show_timestamp: bool = True,
        show_error_id: bool = False,
        custom_css: Optional[str] = None
    ) -> None:
        """
        Render an error with specified style and options
        
        Args:
            error_info: Error information to display
            style: Display style to use
            show_recovery_actions: Whether to show recovery action buttons
            show_timestamp: Whether to show error timestamp
            show_error_id: Whether to show error ID
            custom_css: Custom CSS for styling
        """
        if custom_css:
            st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
        
        if style == ErrorDisplayStyle.MINIMAL:
            ErrorDisplay._render_minimal_error(error_info)
        elif style == ErrorDisplayStyle.STANDARD:
            ErrorDisplay._render_standard_error(
                error_info, show_recovery_actions, show_timestamp, show_error_id
            )
        elif style == ErrorDisplayStyle.DETAILED:
            ErrorDisplay._render_detailed_error(
                error_info, show_recovery_actions, show_timestamp, show_error_id
            )
        elif style == ErrorDisplayStyle.INLINE:
            ErrorDisplay._render_inline_error(error_info, show_recovery_actions)
        elif style == ErrorDisplayStyle.TOAST:
            ErrorDisplay._render_toast_error(error_info)
    
    @staticmethod
    def _render_minimal_error(error_info: ErrorInfo) -> None:
        """Render minimal error display"""
        severity_icons = {
            ErrorSeverity.LOW: "ℹ️",
            ErrorSeverity.MEDIUM: "⚠️", 
            ErrorSeverity.HIGH: "❌",
            ErrorSeverity.CRITICAL: "🚨"
        }
        
        icon = severity_icons.get(error_info.severity, "❌")
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            st.error(f"{icon} {error_info.user_message}")
        elif error_info.severity == ErrorSeverity.HIGH:
            st.error(f"{icon} {error_info.user_message}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            st.warning(f"{icon} {error_info.user_message}")
        else:
            st.info(f"{icon} {error_info.user_message}")
    
    @staticmethod
    def _render_standard_error(
        error_info: ErrorInfo,
        show_recovery_actions: bool,
        show_timestamp: bool,
        show_error_id: bool
    ) -> None:
        """Render standard error display"""
        # Error container
        with st.container():
            # Main error message
            ErrorDisplay._render_error_header(error_info, show_timestamp, show_error_id)
            
            # Error message
            ErrorDisplay._render_error_message(error_info)
            
            # Recovery actions
            if show_recovery_actions and error_info.recovery_actions:
                ErrorDisplay._render_recovery_section(error_info)
            
            # Context information
            if error_info.context:
                ErrorDisplay._render_context_section(error_info.context)
    
    @staticmethod
    def _render_detailed_error(
        error_info: ErrorInfo,
        show_recovery_actions: bool,
        show_timestamp: bool,
        show_error_id: bool
    ) -> None:
        """Render detailed error display"""
        # Standard error display first
        ErrorDisplay._render_standard_error(
            error_info, show_recovery_actions, show_timestamp, show_error_id
        )
        
        # Additional detailed information
        with st.expander("🔍 Technical Details", expanded=False):
            # Error type and category
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Error Category", error_info.category.value.title())
                st.metric("Severity Level", error_info.severity.value.title())
            
            with col2:
                st.metric("Error Type", type(error_info.error).__name__)
                if hasattr(error_info.error, '__module__'):
                    st.metric("Module", error_info.error.__module__)
            
            # Technical details
            if error_info.technical_details:
                st.subheader("Stack Trace")
                st.code(error_info.technical_details, language="python")
            
            # Full context
            if error_info.context:
                st.subheader("Error Context")
                st.json(error_info.context)
    
    @staticmethod
    def _render_inline_error(error_info: ErrorInfo, show_recovery_actions: bool) -> None:
        """Render inline error display"""
        severity_colors = {
            ErrorSeverity.LOW: "#d1ecf1",
            ErrorSeverity.MEDIUM: "#fff3cd",
            ErrorSeverity.HIGH: "#f8d7da",
            ErrorSeverity.CRITICAL: "#721c24"
        }
        
        color = severity_colors.get(error_info.severity, "#f8d7da")
        
        st.markdown(
            f"""
            <div style="
                padding: 10px;
                border-left: 4px solid {color};
                background-color: {color}20;
                margin: 10px 0;
                border-radius: 4px;
            ">
                <strong>⚠️ {error_info.category.value.title()} Error</strong><br>
                {error_info.user_message}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if show_recovery_actions and error_info.recovery_actions:
            ErrorDisplay._render_inline_recovery_actions(error_info)
    
    @staticmethod
    def _render_toast_error(error_info: ErrorInfo) -> None:
        """Render toast-style error notification"""
        # Use Streamlit's built-in toast if available, otherwise use alert
        if hasattr(st, 'toast'):
            icon_map = {
                ErrorSeverity.LOW: "ℹ️",
                ErrorSeverity.MEDIUM: "⚠️",
                ErrorSeverity.HIGH: "❌",
                ErrorSeverity.CRITICAL: "🚨"
            }
            
            icon = icon_map.get(error_info.severity, "❌")
            st.toast(f"{icon} {error_info.user_message}")
        else:
            # Fallback to alert
            st.error(f"🔔 {error_info.user_message}")
    
    @staticmethod
    def _render_error_header(
        error_info: ErrorInfo,
        show_timestamp: bool,
        show_error_id: bool
    ) -> None:
        """Render error header with metadata"""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            severity_icons = {
                ErrorSeverity.LOW: "ℹ️",
                ErrorSeverity.MEDIUM: "⚠️",
                ErrorSeverity.HIGH: "❌",
                ErrorSeverity.CRITICAL: "🚨"
            }
            
            icon = severity_icons.get(error_info.severity, "❌")
            category = error_info.category.value.replace("_", " ").title()
            
            st.markdown(f"### {icon} {category} Error")
        
        with col2:
            if show_error_id:
                st.caption(f"ID: {error_info.error_id}")
        
        with col3:
            if show_timestamp:
                time_str = error_info.timestamp.strftime("%H:%M:%S")
                st.caption(f"Time: {time_str}")
    
    @staticmethod
    def _render_error_message(error_info: ErrorInfo) -> None:
        """Render the main error message"""
        if error_info.severity == ErrorSeverity.CRITICAL:
            st.error(error_info.user_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            st.error(error_info.user_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            st.warning(error_info.user_message)
        else:
            st.info(error_info.user_message)
    
    @staticmethod
    def _render_recovery_section(error_info: ErrorInfo) -> None:
        """Render recovery actions section"""
        st.markdown("**🔧 Recovery Options:**")
        
        # Group recovery actions into columns
        num_actions = len(error_info.recovery_actions)
        cols = st.columns(min(num_actions, 4))
        
        action_labels = {
            RecoveryAction.RETRY: "🔄 Retry",
            RecoveryAction.REFRESH: "🔄 Refresh",
            RecoveryAction.RESTART_SESSION: "🔄 Restart",
            RecoveryAction.CLEAR_DATA: "🗑️ Clear Data",
            RecoveryAction.CONTACT_SUPPORT: "📞 Support",
            RecoveryAction.IGNORE: "❌ Dismiss"
        }
        
        for i, action in enumerate(error_info.recovery_actions):
            with cols[i % len(cols)]:
                label = action_labels.get(action, action.value.title())
                
                if st.button(
                    label,
                    key=f"recovery_{error_info.error_id}_{action.value}",
                    help=f"Execute {action.value} recovery action"
                ):
                    ErrorDisplay._execute_recovery_action(error_info, action)
    
    @staticmethod
    def _render_inline_recovery_actions(error_info: ErrorInfo) -> None:
        """Render inline recovery actions"""
        cols = st.columns(len(error_info.recovery_actions))
        
        action_labels = {
            RecoveryAction.RETRY: "Retry",
            RecoveryAction.REFRESH: "Refresh", 
            RecoveryAction.IGNORE: "Dismiss"
        }
        
        for i, action in enumerate(error_info.recovery_actions[:3]):  # Limit inline actions
            with cols[i]:
                label = action_labels.get(action, action.value)
                
                if st.button(
                    label,
                    key=f"inline_recovery_{error_info.error_id}_{action.value}",
                    help=f"Execute {action.value}"
                ):
                    ErrorDisplay._execute_recovery_action(error_info, action)
    
    @staticmethod
    def _render_context_section(context: Dict[str, Any]) -> None:
        """Render error context information"""
        if not context:
            return
        
        with st.expander("ℹ️ Error Context", expanded=False):
            # Display important context items first
            important_keys = ["operation", "component", "user_action", "state"]
            
            for key in important_keys:
                if key in context:
                    st.text(f"{key.title()}: {context[key]}")
            
            # Display remaining context
            remaining_context = {
                k: v for k, v in context.items() 
                if k not in important_keys
            }
            
            if remaining_context:
                st.json(remaining_context)
    
    @staticmethod
    def _execute_recovery_action(error_info: ErrorInfo, action: RecoveryAction) -> None:
        """Execute a recovery action"""
        from src.utils.error_handler import get_error_handler
        
        error_handler = get_error_handler()
        
        with st.spinner(f"Executing {action.value}..."):
            success = error_handler.execute_recovery_action(error_info, action)
            
            if success:
                st.success(f"✅ {action.value.title()} completed successfully!")
                
                # Clear current error if action was successful
                error_handler.clear_current_error()
                
                # Small delay before rerun to show success message
                import time
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ {action.value.title()} failed. Please try another option.")


class ErrorSummaryWidget:
    """Widget for displaying error summaries and statistics"""
    
    @staticmethod
    def render_error_summary(
        error_history: List[ErrorInfo],
        max_display: int = 5,
        show_statistics: bool = True
    ) -> None:
        """Render error summary widget"""
        
        if not error_history:
            st.info("✅ No recent errors")
            return
        
        # Statistics
        if show_statistics:
            ErrorSummaryWidget._render_error_statistics(error_history)
        
        # Recent errors
        st.subheader("🕒 Recent Errors")
        
        recent_errors = error_history[-max_display:]
        
        for i, error_info in enumerate(reversed(recent_errors)):
            with st.expander(
                f"{error_info.category.value.title()}: {error_info.user_message[:50]}...",
                expanded=i == 0  # Expand most recent error
            ):
                ErrorDisplay.render_error(
                    error_info,
                    style=ErrorDisplayStyle.STANDARD,
                    show_recovery_actions=False
                )
    
    @staticmethod
    def _render_error_statistics(error_history: List[ErrorInfo]) -> None:
        """Render error statistics"""
        # Calculate statistics
        total_errors = len(error_history)
        
        category_counts = {}
        severity_counts = {}
        
        for error in error_history:
            category_counts[error.category] = category_counts.get(error.category, 0) + 1
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Errors", total_errors)
        
        with col2:
            critical_count = severity_counts.get(ErrorSeverity.CRITICAL, 0)
            st.metric("Critical", critical_count, delta=None if critical_count == 0 else "⚠️")
        
        with col3:
            most_common_category = max(category_counts, key=category_counts.get)
            st.metric("Most Common", most_common_category.value.title())
        
        with col4:
            recent_hour_count = len([
                e for e in error_history
                if (datetime.now() - e.timestamp).total_seconds() < 3600
            ])
            st.metric("Last Hour", recent_hour_count)


def render_error_from_session_state(
    error_key: str = "current_error",
    style: ErrorDisplayStyle = ErrorDisplayStyle.STANDARD,
    clear_after_display: bool = False
) -> bool:
    """
    Render error from session state
    
    Args:
        error_key: Session state key containing error info
        style: Display style to use
        clear_after_display: Whether to clear error after displaying
        
    Returns:
        True if error was displayed, False if no error found
    """
    from src.utils.error_handler import get_error_handler
    
    error_handler = get_error_handler()
    error_info = error_handler.get_current_error()
    
    if not error_info:
        return False
    
    ErrorDisplay.render_error(error_info, style=style)
    
    if clear_after_display:
        error_handler.clear_current_error()
    
    return True


def create_error_notification(
    message: str,
    error_type: str = "error",
    duration: int = 5,
    show_close_button: bool = True
) -> None:
    """
    Create a temporary error notification
    
    Args:
        message: Error message to display
        error_type: Type of notification (error, warning, info)
        duration: How long to show notification (seconds)
        show_close_button: Whether to show close button
    """
    # Create notification container
    placeholder = st.empty()
    
    # Render notification
    with placeholder.container():
        if error_type == "error":
            st.error(f"🔔 {message}")
        elif error_type == "warning":
            st.warning(f"⚠️ {message}")
        else:
            st.info(f"ℹ️ {message}")
        
        if show_close_button:
            if st.button("❌ Close", key=f"close_notification_{hash(message)}"):
                placeholder.empty()
                return
    
    # Auto-clear after duration
    if duration > 0:
        import time
        time.sleep(duration)
        placeholder.empty()