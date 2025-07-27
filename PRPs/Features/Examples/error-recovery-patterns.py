"""
Error Recovery Patterns Example

Comprehensive examples of error handling and recovery strategies
for LLM integration, covering all failure modes and user-friendly
fallback mechanisms.
"""

import asyncio
import streamlit as st
from typing import Optional, Dict, Any, List
from enum import Enum
import time
import random
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur in LLM integration"""
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "auth_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    STREAM_INTERRUPTED = "stream_interrupted"
    QUOTA_EXCEEDED = "quota_exceeded"
    SERVICE_UNAVAILABLE = "service_unavailable"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    error_type: ErrorType
    error_message: str
    timestamp: datetime
    retry_count: int = 0
    user_message: Optional[str] = None
    recovery_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.recovery_suggestions is None:
            self.recovery_suggestions = []


class ErrorRecoveryManager:
    """
    Comprehensive error recovery manager for LLM integration.
    
    Handles different error types with appropriate recovery strategies,
    user-friendly messaging, and graceful degradation.
    """
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.retry_limits = {
            ErrorType.NETWORK_ERROR: 3,
            ErrorType.API_ERROR: 2,
            ErrorType.TIMEOUT_ERROR: 2,
            ErrorType.STREAM_INTERRUPTED: 1,
            ErrorType.SERVICE_UNAVAILABLE: 0,  # No retries for service issues
            ErrorType.AUTHENTICATION_ERROR: 0,  # No retries for auth issues
            ErrorType.RATE_LIMIT_ERROR: 0,  # Handle with queuing instead
            ErrorType.QUOTA_EXCEEDED: 0,  # No retries for quota issues
        }
        self.fallback_responses = {
            ErrorType.NETWORK_ERROR: "I'm having trouble connecting to the AI service. Let me try again...",
            ErrorType.AUTHENTICATION_ERROR: "There's an authentication issue with the AI service. Please check your API configuration.",
            ErrorType.RATE_LIMIT_ERROR: "The AI service is currently busy. Your request has been queued.",
            ErrorType.API_ERROR: "The AI service encountered an error. Let me try a different approach...",
            ErrorType.TIMEOUT_ERROR: "The AI service is taking longer than expected. Let me try again with a shorter request...",
            ErrorType.STREAM_INTERRUPTED: "The response was interrupted. Here's what I had generated so far...",
            ErrorType.QUOTA_EXCEEDED: "You've reached your usage limit for the AI service. Please try again later or contact support.",
            ErrorType.SERVICE_UNAVAILABLE: "The AI service is temporarily unavailable. Please try again in a few minutes."
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """
        Main error handling entry point.
        
        Classifies error, determines recovery strategy, and returns
        appropriate error context for UI display.
        """
        error_type = self._classify_error(error)
        error_context = ErrorContext(
            error_type=error_type,
            error_message=str(error),
            timestamp=datetime.now(),
            user_message=context.get("user_message") if context else None
        )
        
        # Add recovery suggestions
        error_context.recovery_suggestions = self._get_recovery_suggestions(error_type)
        
        # Store in history
        self.error_history.append(error_context)
        
        # Log error
        logger.error(f"LLM Error ({error_type.value}): {error}", extra=context or {})
        
        return error_context
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error into appropriate error type"""
        error_str = str(error).lower()
        
        if "authentication" in error_str or "unauthorized" in error_str or "api key" in error_str:
            return ErrorType.AUTHENTICATION_ERROR
        elif "rate limit" in error_str or "too many requests" in error_str:
            return ErrorType.RATE_LIMIT_ERROR
        elif "quota" in error_str or "billing" in error_str:
            return ErrorType.QUOTA_EXCEEDED
        elif "timeout" in error_str or "timed out" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "connection" in error_str or "network" in error_str:
            return ErrorType.NETWORK_ERROR
        elif "service unavailable" in error_str or "503" in error_str:
            return ErrorType.SERVICE_UNAVAILABLE
        elif "stream" in error_str and "interrupt" in error_str:
            return ErrorType.STREAM_INTERRUPTED
        else:
            return ErrorType.API_ERROR
    
    def _get_recovery_suggestions(self, error_type: ErrorType) -> List[str]:
        """Get user-friendly recovery suggestions for error type"""
        suggestions = {
            ErrorType.NETWORK_ERROR: [
                "Check your internet connection",
                "Try again in a moment",
                "Verify the AI service is accessible"
            ],
            ErrorType.AUTHENTICATION_ERROR: [
                "Verify your API key is correct",
                "Check if your API key has expired",
                "Contact administrator for access"
            ],
            ErrorType.RATE_LIMIT_ERROR: [
                "Wait a moment before trying again",
                "Your request will be processed in queue",
                "Consider breaking large requests into smaller parts"
            ],
            ErrorType.API_ERROR: [
                "Try rephrasing your request",
                "Use a shorter message",
                "Try again in a moment"
            ],
            ErrorType.TIMEOUT_ERROR: [
                "Try a shorter request",
                "Check your connection speed",
                "The service may be under heavy load"
            ],
            ErrorType.STREAM_INTERRUPTED: [
                "The partial response has been saved",
                "You can continue from where it left off",
                "Try asking for the remainder"
            ],
            ErrorType.QUOTA_EXCEEDED: [
                "Check your usage limits",
                "Contact administrator to increase quota",
                "Try again after your quota resets"
            ],
            ErrorType.SERVICE_UNAVAILABLE: [
                "The service is temporarily down",
                "Check the service status page",
                "Try again in a few minutes"
            ]
        }
        return suggestions.get(error_type, ["Try again later"])
    
    def can_retry(self, error_type: ErrorType, current_retry_count: int = 0) -> bool:
        """Check if error type allows retries"""
        max_retries = self.retry_limits.get(error_type, 0)
        return current_retry_count < max_retries
    
    def get_fallback_response(self, error_type: ErrorType) -> str:
        """Get user-friendly fallback response for error"""
        return self.fallback_responses.get(error_type, "I encountered an unexpected error. Please try again.")
    
    async def retry_with_backoff(self, operation, error_type: ErrorType, max_retries: int = None):
        """Retry operation with exponential backoff"""
        if max_retries is None:
            max_retries = self.retry_limits.get(error_type, 0)
        
        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                # Exponential backoff
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Retry attempt {attempt + 1} after {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        raise Exception(f"Operation failed after {max_retries} retries")


class ErrorDisplayManager:
    """
    Manages user-friendly error display in Streamlit UI.
    
    Provides consistent error messaging, recovery actions,
    and progress feedback during error recovery.
    """
    
    def __init__(self, recovery_manager: ErrorRecoveryManager):
        self.recovery_manager = recovery_manager
    
    def display_error(self, error_context: ErrorContext, container_key: str = "error_display"):
        """Display comprehensive error information with recovery options"""
        
        # Error severity styling
        severity_config = self._get_severity_config(error_context.error_type)
        
        # Main error message
        st.error(f"🚨 **{severity_config['title']}**")
        
        # Error details in expandable section
        with st.expander("📋 Error Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Error Type", error_context.error_type.value.replace('_', ' ').title())
                st.metric("Timestamp", error_context.timestamp.strftime("%H:%M:%S"))
            
            with col2:
                st.metric("Retry Count", error_context.retry_count)
                if error_context.user_message:
                    st.text_area("User Message", error_context.user_message, height=60, disabled=True)
        
        # Recovery suggestions
        if error_context.recovery_suggestions:
            st.markdown("### 💡 **What you can do:**")
            for suggestion in error_context.recovery_suggestions:
                st.markdown(f"• {suggestion}")
        
        # Recovery actions
        self._render_recovery_actions(error_context, container_key)
    
    def _get_severity_config(self, error_type: ErrorType) -> Dict[str, str]:
        """Get severity configuration for error type"""
        configs = {
            ErrorType.NETWORK_ERROR: {"title": "Connection Issue", "color": "orange"},
            ErrorType.AUTHENTICATION_ERROR: {"title": "Authentication Failed", "color": "red"},
            ErrorType.RATE_LIMIT_ERROR: {"title": "Service Busy", "color": "yellow"},
            ErrorType.API_ERROR: {"title": "Service Error", "color": "orange"},
            ErrorType.TIMEOUT_ERROR: {"title": "Request Timeout", "color": "orange"},
            ErrorType.STREAM_INTERRUPTED: {"title": "Response Interrupted", "color": "yellow"},
            ErrorType.QUOTA_EXCEEDED: {"title": "Usage Limit Reached", "color": "red"},
            ErrorType.SERVICE_UNAVAILABLE: {"title": "Service Unavailable", "color": "red"}
        }
        return configs.get(error_type, {"title": "Unknown Error", "color": "red"})
    
    def _render_recovery_actions(self, error_context: ErrorContext, container_key: str):
        """Render recovery action buttons"""
        st.markdown("### 🔧 **Recovery Actions:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if self.recovery_manager.can_retry(error_context.error_type, error_context.retry_count):
                if st.button("🔄 Retry", key=f"{container_key}_retry"):
                    st.session_state[f"{container_key}_action"] = "retry"
                    st.rerun()
        
        with col2:
            if st.button("📞 Get Help", key=f"{container_key}_help"):
                self._show_help_dialog(error_context)
        
        with col3:
            if st.button("📊 View Status", key=f"{container_key}_status"):
                self._show_status_dialog()
        
        with col4:
            if st.button("🗑️ Clear Error", key=f"{container_key}_clear"):
                st.session_state[f"{container_key}_action"] = "clear"
                st.rerun()
    
    def _show_help_dialog(self, error_context: ErrorContext):
        """Show help dialog with detailed troubleshooting"""
        st.info("📞 **Need Help?**")
        
        troubleshooting_steps = {
            ErrorType.AUTHENTICATION_ERROR: [
                "1. Check that your API key is correctly set in the environment",
                "2. Verify the API key hasn't expired",
                "3. Ensure you have the correct permissions",
                "4. Contact your administrator if the issue persists"
            ],
            ErrorType.NETWORK_ERROR: [
                "1. Check your internet connection",
                "2. Verify firewall settings allow API access",
                "3. Try accessing the service from a different network",
                "4. Contact IT support if issues continue"
            ],
            ErrorType.RATE_LIMIT_ERROR: [
                "1. Wait 60 seconds before trying again",
                "2. Break large requests into smaller parts",
                "3. Consider upgrading your service plan",
                "4. Contact support for rate limit increases"
            ]
        }
        
        steps = troubleshooting_steps.get(error_context.error_type, [
            "1. Try again in a few minutes",
            "2. Check the service status page",
            "3. Contact technical support with error details"
        ])
        
        st.markdown("**Troubleshooting Steps:**")
        for step in steps:
            st.markdown(step)
    
    def _show_status_dialog(self):
        """Show service status information"""
        st.info("📊 **Service Status**")
        
        # Mock service status - in real implementation, this would check actual service health
        status_items = [
            {"service": "Groq API", "status": "✅ Operational", "response_time": "145ms"},
            {"service": "Authentication", "status": "✅ Operational", "response_time": "23ms"},
            {"service": "Rate Limiting", "status": "⚠️ Degraded", "response_time": "2.1s"},
            {"service": "Streaming", "status": "✅ Operational", "response_time": "89ms"}
        ]
        
        for item in status_items:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**{item['service']}**")
            with col2:
                st.write(item['status'])
            with col3:
                st.write(item['response_time'])
    
    def display_recovery_progress(self, operation_name: str, progress: float):
        """Display progress during error recovery operations"""
        st.markdown(f"### 🔄 **{operation_name}**")
        
        progress_bar = st.progress(progress)
        status_text = st.empty()
        
        if progress < 0.3:
            status_text.text("Initializing recovery...")
        elif progress < 0.6:
            status_text.text("Attempting connection...")
        elif progress < 0.9:
            status_text.text("Verifying service...")
        else:
            status_text.text("Recovery complete!")
        
        return progress_bar, status_text


class FallbackResponseGenerator:
    """
    Generates helpful fallback responses when LLM is unavailable.
    
    Provides contextually appropriate responses that maintain
    user engagement while the service recovers.
    """
    
    def __init__(self):
        self.cached_responses = {}
        self.context_patterns = [
            {"keywords": ["hello", "hi", "greeting"], "response": "Hello! I'm currently experiencing some technical difficulties, but I'm here to help as soon as the service is restored."},
            {"keywords": ["help", "support", "assistance"], "response": "I'd be happy to help! The AI service is temporarily unavailable, but you can try again in a moment or contact support for immediate assistance."},
            {"keywords": ["status", "working", "broken"], "response": "I'm currently experiencing connectivity issues with the AI service. The technical team has been notified and is working on a resolution."},
            {"keywords": ["document", "generate", "create"], "response": "Document generation is temporarily unavailable due to AI service issues. Your request has been saved and will be processed once the service is restored."}
        ]
    
    def generate_fallback(self, user_message: str, error_type: ErrorType) -> str:
        """Generate contextually appropriate fallback response"""
        
        # Check for cached response
        cache_key = f"{error_type.value}_{hash(user_message.lower())}"
        if cache_key in self.cached_responses:
            return self.cached_responses[cache_key]
        
        # Generate contextual response
        response = self._generate_contextual_response(user_message, error_type)
        
        # Cache for future use
        self.cached_responses[cache_key] = response
        
        return response
    
    def _generate_contextual_response(self, user_message: str, error_type: ErrorType) -> str:
        """Generate response based on message context and error type"""
        user_message_lower = user_message.lower()
        
        # Find matching context pattern
        for pattern in self.context_patterns:
            if any(keyword in user_message_lower for keyword in pattern["keywords"]):
                base_response = pattern["response"]
                break
        else:
            base_response = "I'm sorry, but I'm currently unable to process your request due to technical difficulties."
        
        # Add error-specific guidance
        error_guidance = {
            ErrorType.RATE_LIMIT_ERROR: " Please wait a moment and try again.",
            ErrorType.SERVICE_UNAVAILABLE: " The service should be back online shortly.",
            ErrorType.QUOTA_EXCEEDED: " You may have reached your usage limit for today.",
            ErrorType.AUTHENTICATION_ERROR: " There may be a configuration issue that needs attention."
        }
        
        guidance = error_guidance.get(error_type, " Please try again in a few minutes.")
        
        return f"{base_response}{guidance}"


# Demo application
def main():
    """Demo application showing error recovery patterns"""
    st.set_page_config(
        page_title="Error Recovery Patterns Demo",
        page_icon="🚨",
        layout="wide"
    )
    
    st.title("🚨 Error Recovery Patterns Demo")
    st.markdown("Comprehensive demonstration of LLM integration error handling and recovery")
    
    # Initialize managers
    if "recovery_manager" not in st.session_state:
        st.session_state.recovery_manager = ErrorRecoveryManager()
    
    if "display_manager" not in st.session_state:
        st.session_state.display_manager = ErrorDisplayManager(st.session_state.recovery_manager)
    
    if "fallback_generator" not in st.session_state:
        st.session_state.fallback_generator = FallbackResponseGenerator()
    
    # Demo tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Error Simulation", "🔄 Recovery Demo", "💬 Fallback Responses", "📊 Error Analytics"])
    
    with tab1:
        demo_error_simulation()
    
    with tab2:
        demo_recovery_process()
    
    with tab3:
        demo_fallback_responses()
    
    with tab4:
        demo_error_analytics()


def demo_error_simulation():
    """Demo error simulation and handling"""
    st.subheader("🚨 Error Simulation")
    
    # Error type selection
    error_type = st.selectbox(
        "Select Error Type to Simulate:",
        options=list(ErrorType),
        format_func=lambda x: x.value.replace('_', ' ').title()
    )
    
    user_message = st.text_input(
        "User Message (for context):",
        value="Can you help me generate a document?"
    )
    
    if st.button("🎭 Simulate Error"):
        # Create mock error
        mock_error = Exception(f"Simulated {error_type.value} error")
        
        # Handle error
        error_context = st.session_state.recovery_manager.handle_error(
            mock_error, 
            {"user_message": user_message}
        )
        
        # Display error
        st.session_state.display_manager.display_error(error_context, "simulation")
        
        # Show fallback response
        st.markdown("### 🤖 **Fallback Response:**")
        fallback = st.session_state.fallback_generator.generate_fallback(user_message, error_type)
        st.info(fallback)


def demo_recovery_process():
    """Demo recovery process with progress indication"""
    st.subheader("🔄 Recovery Process Demo")
    
    if st.button("🔧 Start Recovery Process"):
        # Simulate recovery process
        progress_container = st.container()
        
        with progress_container:
            progress_bar, status_text = st.session_state.display_manager.display_recovery_progress(
                "Service Recovery", 0.0
            )
            
            # Simulate recovery steps
            for i in range(101):
                progress = i / 100.0
                progress_bar.progress(progress)
                
                if i == 30:
                    status_text.text("Checking service connectivity...")
                elif i == 60:
                    status_text.text("Restoring API connection...")
                elif i == 90:
                    status_text.text("Verifying functionality...")
                elif i == 100:
                    status_text.text("✅ Recovery complete!")
                    st.success("Service has been successfully restored!")
                
                time.sleep(0.02)  # Simulate work


def demo_fallback_responses():
    """Demo fallback response generation"""
    st.subheader("💬 Fallback Response Generator")
    
    test_messages = [
        "Hello, how are you?",
        "Can you help me with something?",
        "Is the service working?",
        "I need to generate a document",
        "What's your status?",
        "Why is this not working?"
    ]
    
    selected_message = st.selectbox("Select Test Message:", test_messages)
    custom_message = st.text_input("Or enter custom message:")
    
    message_to_test = custom_message if custom_message else selected_message
    
    error_for_fallback = st.selectbox(
        "Error Type for Fallback:",
        options=list(ErrorType),
        format_func=lambda x: x.value.replace('_', ' ').title()
    )
    
    if st.button("Generate Fallback Response"):
        fallback = st.session_state.fallback_generator.generate_fallback(
            message_to_test, error_for_fallback
        )
        
        st.markdown("### 🤖 **Generated Fallback:**")
        st.success(fallback)


def demo_error_analytics():
    """Demo error analytics and reporting"""
    st.subheader("📊 Error Analytics")
    
    # Generate mock error history
    if not st.session_state.recovery_manager.error_history:
        st.info("No errors recorded yet. Use the Error Simulation tab to generate some test data.")
    else:
        errors = st.session_state.recovery_manager.error_history
        
        # Error type distribution
        error_counts = {}
        for error in errors:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Error Type Distribution:**")
            for error_type, count in error_counts.items():
                st.metric(error_type.replace('_', ' ').title(), count)
        
        with col2:
            st.markdown("**Recent Errors:**")
            for error in errors[-5:]:  # Show last 5 errors
                st.text(f"{error.timestamp.strftime('%H:%M:%S')} - {error.error_type.value}")
        
        # Error timeline
        st.markdown("**Error Timeline:**")
        error_timeline = []
        for error in errors:
            error_timeline.append({
                "Time": error.timestamp.strftime('%H:%M:%S'),
                "Type": error.error_type.value,
                "Message": error.error_message[:50] + "..." if len(error.error_message) > 50 else error.error_message
            })
        
        if error_timeline:
            st.dataframe(error_timeline, use_container_width=True)


if __name__ == "__main__":
    main()