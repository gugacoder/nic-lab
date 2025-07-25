"""
Centralized Error Handling System

This module provides comprehensive error handling capabilities for the NIC Chat system,
including error classification, recovery strategies, logging, and user-friendly messaging.
"""

import logging
import traceback
from typing import Optional, Dict, Any, Callable, List, Tuple
from enum import Enum
from datetime import datetime, timedelta
import streamlit as st


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    AI_API = "ai_api"
    UI = "ui"
    DATA = "data"
    SESSION = "session"
    VALIDATION = "validation"
    SYSTEM = "system"


class RecoveryAction(Enum):
    """Available recovery actions"""
    RETRY = "retry"
    REFRESH = "refresh"
    RESTART_SESSION = "restart_session"
    CLEAR_DATA = "clear_data"
    CONTACT_SUPPORT = "contact_support"
    IGNORE = "ignore"


class ErrorInfo:
    """Error information container"""
    
    def __init__(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        user_message: str,
        recovery_actions: List[RecoveryAction],
        technical_details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.error = error
        self.category = category
        self.severity = severity
        self.user_message = user_message
        self.recovery_actions = recovery_actions
        self.technical_details = technical_details or str(error)
        self.context = context or {}
        self.timestamp = datetime.now()
        self.error_id = f"{category.value}_{int(self.timestamp.timestamp())}"


class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ErrorInfo] = []
        self.recovery_handlers: Dict[RecoveryAction, Callable] = {}
        self.retry_counts: Dict[str, int] = {}
        self.max_retries = 3
        self.retry_backoff = [1, 2, 5]  # seconds
    
    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_message: str = None,
        recovery_actions: List[RecoveryAction] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorInfo:
        """
        Handle an error with comprehensive processing
        
        Args:
            error: The caught exception
            category: Error category
            severity: Error severity level
            user_message: User-friendly message
            recovery_actions: Available recovery actions
            context: Additional context information
            
        Returns:
            ErrorInfo object containing processed error details
        """
        # Auto-generate user message if not provided
        if user_message is None:
            user_message = self._generate_user_message(error, category)
        
        # Auto-determine recovery actions if not provided
        if recovery_actions is None:
            recovery_actions = self._determine_recovery_actions(category, severity)
        
        # Create error info
        error_info = ErrorInfo(
            error=error,
            category=category,
            severity=severity,
            user_message=user_message,
            recovery_actions=recovery_actions,
            technical_details=traceback.format_exc(),
            context=context
        )
        
        # Log the error
        self._log_error(error_info)
        
        # Store in history
        self.error_history.append(error_info)
        
        # Update session state
        self._update_session_state(error_info)
        
        return error_info
    
    def _generate_user_message(self, error: Exception, category: ErrorCategory) -> str:
        """Generate user-friendly error message"""
        message_templates = {
            ErrorCategory.NETWORK: "Connection issue occurred. Please check your network and try again.",
            ErrorCategory.AI_API: "AI service is temporarily unavailable. Please try again in a moment.",
            ErrorCategory.UI: "Interface error occurred. Please refresh the page or try again.",
            ErrorCategory.DATA: "Data processing error. Please verify your input and try again.",
            ErrorCategory.SESSION: "Session error occurred. You may need to refresh the page.",
            ErrorCategory.VALIDATION: "Input validation failed. Please check your input and try again.",
            ErrorCategory.SYSTEM: "System error occurred. Please try again or contact support."
        }
        
        base_message = message_templates.get(category, "An unexpected error occurred.")
        
        # Add specific details for common errors
        error_type = type(error).__name__
        if "timeout" in str(error).lower():
            base_message = "Operation timed out. Please try again."
        elif "connection" in str(error).lower():
            base_message = "Connection failed. Please check your network and try again."
        elif "permission" in str(error).lower():
            base_message = "Permission denied. Please check your access rights."
        
        return base_message
    
    def _determine_recovery_actions(
        self, 
        category: ErrorCategory, 
        severity: ErrorSeverity
    ) -> List[RecoveryAction]:
        """Determine appropriate recovery actions"""
        recovery_map = {
            ErrorCategory.NETWORK: [RecoveryAction.RETRY, RecoveryAction.REFRESH],
            ErrorCategory.AI_API: [RecoveryAction.RETRY, RecoveryAction.REFRESH],
            ErrorCategory.UI: [RecoveryAction.REFRESH, RecoveryAction.RETRY],
            ErrorCategory.DATA: [RecoveryAction.RETRY, RecoveryAction.CLEAR_DATA],
            ErrorCategory.SESSION: [RecoveryAction.RESTART_SESSION, RecoveryAction.REFRESH],
            ErrorCategory.VALIDATION: [RecoveryAction.RETRY],
            ErrorCategory.SYSTEM: [RecoveryAction.REFRESH, RecoveryAction.CONTACT_SUPPORT]
        }
        
        actions = recovery_map.get(category, [RecoveryAction.RETRY])
        
        # Add contact support for critical errors
        if severity == ErrorSeverity.CRITICAL:
            actions.append(RecoveryAction.CONTACT_SUPPORT)
        
        return actions
    
    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error with appropriate level"""
        log_levels = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        
        level = log_levels.get(error_info.severity, logging.ERROR)
        
        self.logger.log(
            level,
            f"[{error_info.error_id}] {error_info.category.value.upper()}: {error_info.user_message}",
            extra={
                'error_id': error_info.error_id,
                'category': error_info.category.value,
                'severity': error_info.severity.value,
                'technical_details': error_info.technical_details,
                'context': error_info.context
            }
        )
    
    def _update_session_state(self, error_info: ErrorInfo) -> None:
        """Update Streamlit session state with error information"""
        if "error_handler" not in st.session_state:
            st.session_state["error_handler"] = {}
        
        st.session_state["error_handler"]["current_error"] = error_info
        st.session_state["error_handler"]["error_history"] = self.error_history[-10:]  # Keep last 10 errors
    
    def can_retry(self, operation_id: str) -> bool:
        """Check if operation can be retried"""
        retry_count = self.retry_counts.get(operation_id, 0)
        return retry_count < self.max_retries
    
    def get_retry_delay(self, operation_id: str) -> int:
        """Get retry delay in seconds"""
        retry_count = self.retry_counts.get(operation_id, 0)
        if retry_count < len(self.retry_backoff):
            return self.retry_backoff[retry_count]
        return self.retry_backoff[-1]
    
    def record_retry(self, operation_id: str) -> None:
        """Record a retry attempt"""
        self.retry_counts[operation_id] = self.retry_counts.get(operation_id, 0) + 1
    
    def reset_retry_count(self, operation_id: str) -> None:
        """Reset retry count for successful operation"""
        if operation_id in self.retry_counts:
            del self.retry_counts[operation_id]
    
    def register_recovery_handler(
        self, 
        action: RecoveryAction, 
        handler: Callable[[ErrorInfo], bool]
    ) -> None:
        """Register a recovery action handler"""
        self.recovery_handlers[action] = handler
    
    def execute_recovery_action(self, error_info: ErrorInfo, action: RecoveryAction) -> bool:
        """Execute a recovery action"""
        if action in self.recovery_handlers:
            try:
                return self.recovery_handlers[action](error_info)
            except Exception as e:
                self.logger.error(f"Recovery action {action.value} failed: {e}")
                return False
        
        # Default recovery actions
        return self._execute_default_recovery(error_info, action)
    
    def _execute_default_recovery(self, error_info: ErrorInfo, action: RecoveryAction) -> bool:
        """Execute default recovery actions"""
        try:
            if action == RecoveryAction.REFRESH:
                st.rerun()
                return True
            
            elif action == RecoveryAction.RESTART_SESSION:
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key not in ['_get_option_by_key', '_get_widget_states']:
                        del st.session_state[key]
                st.rerun()
                return True
            
            elif action == RecoveryAction.CLEAR_DATA:
                # Clear application data
                if "messages" in st.session_state:
                    st.session_state["messages"] = []
                st.rerun()
                return True
            
            elif action == RecoveryAction.IGNORE:
                # Simply clear the error from session state
                if "error_handler" in st.session_state:
                    st.session_state["error_handler"]["current_error"] = None
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Default recovery action {action.value} failed: {e}")
            return False
    
    def get_current_error(self) -> Optional[ErrorInfo]:
        """Get current error from session state"""
        error_handler_state = st.session_state.get("error_handler", {})
        return error_handler_state.get("current_error")
    
    def clear_current_error(self) -> None:
        """Clear current error from session state"""
        if "error_handler" in st.session_state:
            st.session_state["error_handler"]["current_error"] = None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {"total_errors": 0}
        
        recent_errors = [
            error for error in self.error_history 
            if error.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors_24h": len(recent_errors),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "most_common_category": max(category_counts, key=category_counts.get) if category_counts else None,
            "retry_counts": dict(self.retry_counts)
        }


# Global error handler instance
_global_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_streamlit_error(func: Callable) -> Callable:
    """Decorator for handling Streamlit component errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler = get_error_handler()
            error_info = error_handler.handle_error(
                error=e,
                category=ErrorCategory.UI,
                severity=ErrorSeverity.MEDIUM,
                context={
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }
            )
            
            # Re-raise for debugging in development
            if st.session_state.get("debug_mode", False):
                raise
            
            return None
    
    return wrapper


def handle_api_error(func: Callable) -> Callable:
    """Decorator for handling API call errors"""
    def wrapper(*args, **kwargs):
        operation_id = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
        error_handler = get_error_handler()
        
        try:
            result = func(*args, **kwargs)
            error_handler.reset_retry_count(operation_id)
            return result
            
        except Exception as e:
            if not error_handler.can_retry(operation_id):
                # Max retries exceeded
                error_handler.handle_error(
                    error=e,
                    category=ErrorCategory.AI_API,
                    severity=ErrorSeverity.HIGH,
                    user_message="Operation failed after multiple attempts. Please try again later.",
                    context={"operation_id": operation_id, "retry_count": error_handler.retry_counts.get(operation_id, 0)}
                )
                raise
            
            # Record retry and re-raise for retry logic
            error_handler.record_retry(operation_id)
            raise
    
    return wrapper