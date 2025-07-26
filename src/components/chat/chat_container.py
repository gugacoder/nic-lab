"""
Chat Container Component

This is the main chat container that orchestrates all chat components including
message display, input handling, and user interactions. It provides the complete
chat interface experience with comprehensive error recovery capabilities.
"""

import streamlit as st
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import asyncio
import logging

from src.utils.session import ChatStateManager, UIStateManager
from src.components.chat.message import MessageData, create_message_data
from src.components.chat.message_list import MessageListComponent, MessageListActions
from src.components.chat.chat_input import ChatInputComponent, QuickActions, InputHistory
from src.components.common.loading import LoadingIndicators, ProgressTracker
from src.components.common.error_boundary import ErrorBoundary, error_boundary, SafeContainer
from src.components.common.error_display import ErrorDisplay, ErrorDisplayStyle
from src.utils.error_handler import get_error_handler, ErrorCategory, ErrorSeverity
from src.utils.state_recovery import get_recovery_manager, create_error_snapshot


class ChatContainer:
    """Main chat container component that orchestrates the chat interface"""
    
    @staticmethod
    def render_chat_interface(
        ai_handler: Optional[Callable[[str], str]] = None,
        enable_streaming: bool = True,
        show_quick_actions: bool = True,
        show_conversation_stats: bool = True,
        custom_css: Optional[str] = None
    ) -> None:
        """
        Render the complete chat interface with comprehensive error recovery
        
        Args:
            ai_handler: Function to handle AI responses
            enable_streaming: Whether to enable response streaming
            show_quick_actions: Whether to show quick action buttons
            show_conversation_stats: Whether to show conversation statistics
            custom_css: Custom CSS for styling
        """
        # Create main error boundary for the entire chat interface
        chat_boundary = ErrorBoundary(
            component_name="chat_interface",
            error_category=ErrorCategory.UI,
            fallback_content="Chat interface temporarily unavailable. Please refresh to try again.",
            show_details=st.session_state.get("debug_mode", False),
            auto_snapshot=True
        )
        
        with chat_boundary.catch_errors():
            # Apply custom CSS if provided
            if custom_css:
                st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
            
            # Initialize chat state if needed
            ChatContainer._initialize_chat_state()
            
            # Display any current errors with enhanced error display
            ChatContainer._render_enhanced_error_display()
            
            # Render main chat layout within safe containers
            ChatContainer._render_chat_layout_safe(
                ai_handler=ai_handler,
                enable_streaming=enable_streaming,
                show_quick_actions=show_quick_actions,
                show_conversation_stats=show_conversation_stats
            )
            
            # Handle background processing with error recovery
            ChatContainer._handle_background_processing_safe(ai_handler, enable_streaming)
    
    @staticmethod
    def _initialize_chat_state() -> None:
        """Initialize chat state and session variables"""
        # Ensure session state is properly initialized
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        
        if "is_processing" not in st.session_state:
            st.session_state["is_processing"] = False
        
        if "chat_error" not in st.session_state:
            st.session_state["chat_error"] = None
        
        # Initialize UI preferences
        if "chat_settings" not in st.session_state:
            st.session_state["chat_settings"] = {
                "show_timestamps": True,
                "enable_markdown": True,
                "auto_scroll": True,
                "max_messages": UIStateManager.get_preference("max_messages", 50)
            }
    
    @staticmethod
    def _render_enhanced_error_display() -> None:
        """Render enhanced error display with comprehensive recovery options"""
        try:
            # Check for errors from multiple sources
            chat_error = ChatStateManager.get_error()
            error_handler = get_error_handler()
            current_error = error_handler.get_current_error()
            
            # Display chat-specific errors
            if chat_error:
                ChatContainer._render_chat_error(chat_error)
            
            # Display system errors from error handler
            if current_error:
                ErrorDisplay.render_error(
                    current_error,
                    style=ErrorDisplayStyle.STANDARD,
                    show_recovery_actions=True,
                    show_timestamp=True
                )
            
        except Exception as e:
            # Fallback error display
            st.error(f"❌ Error displaying errors: {str(e)}")
            if st.button("🔄 Reset Error System"):
                ChatContainer._reset_error_system()
    
    @staticmethod
    def _render_chat_error(error_message: str) -> None:
        """Render chat-specific error with enhanced recovery options"""
        with st.container():
            st.error(f"❌ **Chat Error**: {error_message}")
            
            # Enhanced recovery options
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 Retry", help="Retry the last action", key="chat_retry"):
                    ChatContainer._retry_last_action()
            
            with col2:
                if st.button("🔧 Recover", help="Attempt automatic recovery", key="chat_recover"):
                    ChatContainer._attempt_chat_recovery()
            
            with col3:
                if st.button("🗑️ Reset", help="Reset chat state", key="chat_reset"):
                    ChatContainer._reset_chat_state()
            
            with col4:
                if st.button("❌ Clear", help="Clear error message", key="chat_clear"):
                    ChatStateManager.clear_error()
                    st.rerun()
    
    @staticmethod
    def _render_chat_layout_safe(
        ai_handler: Optional[Callable[[str], str]],
        enable_streaming: bool,
        show_quick_actions: bool,
        show_conversation_stats: bool
    ) -> None:
        """Render the main chat layout with all components using error boundaries"""
        
        # Chat header with error boundary
        header_container = SafeContainer("chat_header")
        with header_container.render():
            ChatContainer._render_chat_header()
        
        # Main chat area with error boundaries
        chat_col1, chat_col2 = st.columns([4, 1])
        
        with chat_col1:
            # Message history with error boundary
            messages_container = SafeContainer("message_history")
            with messages_container.render():
                messages = ChatContainer._get_display_messages()
                
                MessageListComponent.render_message_list(
                    messages=messages,
                    show_search=True,
                    show_filters=True,
                    show_timestamps=st.session_state["chat_settings"]["show_timestamps"],
                    enable_pagination=len(messages) > 50
                )
            
            # Message list actions with error boundary
            actions_container = SafeContainer("message_actions")
            with actions_container.render():
                MessageListActions.render_list_actions()
            
            # Quick actions with error boundary
            if show_quick_actions and not ChatStateManager.is_processing():
                st.markdown("---")
                quick_actions_container = SafeContainer("quick_actions")
                with quick_actions_container.render():
                    QuickActions.render_quick_actions()
            
            # Input history with error boundary
            history_container = SafeContainer("input_history")
            with history_container.render():
                InputHistory.render_history_selector()
            
            # Chat input with error boundary
            st.markdown("---")
            input_container = SafeContainer("chat_input")
            with input_container.render():
                ChatContainer._render_input_section_safe(ai_handler)
        
        with chat_col2:
            # Chat sidebar with error boundary
            sidebar_container = SafeContainer("chat_sidebar")
            with sidebar_container.render():
                ChatContainer._render_chat_sidebar(show_conversation_stats)
    
    @staticmethod
    def _render_chat_header() -> None:
        """Render chat header with title and status"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("### 💬 Chat Interface")
        
        with col2:
            # Connection status
            is_connected = not bool(ChatStateManager.get_error())
            LoadingIndicators.render_connection_status(is_connected, "NIC Chat")
        
        with col3:
            # Processing status
            if ChatStateManager.is_processing():
                st.markdown("🔄 **Processing...**")
            else:
                message_count = len(ChatStateManager.get_messages())
                st.markdown(f"💬 **{message_count} messages**")
    
    @staticmethod
    def _render_input_section_safe(ai_handler: Optional[Callable[[str], str]]) -> None:
        """Render the input section with chat input component using error boundaries"""
        try:
            # Handle suggested input from various sources
            from src.components.chat.chat_input import handle_suggested_input
            suggested = handle_suggested_input()
            
            if suggested:
                # Auto-submit suggested input with error handling
                ChatContainer._handle_message_submission_safe(suggested, ai_handler)
                return
            
            # Render chat input with error boundary
            submitted_message = ChatInputComponent.render_chat_input(
                placeholder="Type your message here...",
                max_chars=4000,
                disabled=ChatStateManager.is_processing(),
                on_submit=lambda msg: ChatContainer._handle_message_submission_safe(msg, ai_handler),
                show_char_count=True,
                show_suggestions=True
            )
            
            # Handle submission with error recovery
            if submitted_message:
                ChatContainer._handle_message_submission_safe(submitted_message, ai_handler)
                
        except Exception as e:
            error_handler = get_error_handler()
            error_handler.handle_error(
                error=e,
                category=ErrorCategory.UI,
                severity=ErrorSeverity.MEDIUM,
                user_message="Chat input encountered an error. Please try refreshing the page.",
                context={"component": "chat_input"}
            )
    
    @staticmethod
    def _render_chat_sidebar(show_conversation_stats: bool) -> None:
        """Render chat sidebar with settings and information"""
        
        # Chat settings
        with st.expander("⚙️ Chat Settings", expanded=False):
            # Timestamps toggle
            show_timestamps = st.checkbox(
                "Show timestamps",
                value=st.session_state["chat_settings"]["show_timestamps"],
                key="timestamps_toggle"
            )
            
            if show_timestamps != st.session_state["chat_settings"]["show_timestamps"]:
                st.session_state["chat_settings"]["show_timestamps"] = show_timestamps
                st.rerun()
            
            # Auto-scroll toggle
            auto_scroll = st.checkbox(
                "Auto-scroll to new messages",
                value=st.session_state["chat_settings"]["auto_scroll"],
                key="auto_scroll_toggle"
            )
            
            if auto_scroll != st.session_state["chat_settings"]["auto_scroll"]:
                st.session_state["chat_settings"]["auto_scroll"] = auto_scroll
                UIStateManager.set_preference("auto_scroll", auto_scroll)
            
            # Max messages setting
            max_messages = st.slider(
                "Max messages to display",
                min_value=10,
                max_value=200,
                value=st.session_state["chat_settings"]["max_messages"],
                step=10,
                key="max_messages_slider"
            )
            
            if max_messages != st.session_state["chat_settings"]["max_messages"]:
                st.session_state["chat_settings"]["max_messages"] = max_messages
                UIStateManager.set_preference("max_messages", max_messages)
        
        # Conversation stats
        if show_conversation_stats:
            messages = ChatStateManager.get_messages()
            ChatContainer._render_quick_stats(messages)
        
        # Debug information (if debug mode enabled)
        if st.session_state.get("debug_mode", False):
            ChatContainer._render_debug_info()
    
    @staticmethod
    def _render_quick_stats(messages: list) -> None:
        """Render quick conversation statistics"""
        with st.expander("📊 Quick Stats", expanded=False):
            if messages:
                user_count = len([m for m in messages if m.role == "user"])
                assistant_count = len([m for m in messages if m.role == "assistant"])
                
                st.metric("Your messages", user_count)
                st.metric("AI responses", assistant_count)
                
                # Average response time (if available)
                response_times = []
                for i, msg in enumerate(messages):
                    if (msg.role == "assistant" and 
                        msg.metadata and 
                        "processing_time" in msg.metadata):
                        response_times.append(msg.metadata["processing_time"])
                
                if response_times:
                    avg_time = sum(response_times) / len(response_times)
                    st.metric("Avg response time", f"{avg_time:.1f}s")
            else:
                st.info("No conversation yet")
    
    @staticmethod
    def _render_debug_info() -> None:
        """Render debug information"""
        with st.expander("🐛 Debug Info", expanded=False):
            debug_data = {
                "Session ID": st.session_state.get("session_id", "Unknown"),
                "Messages Count": len(st.session_state.get("messages", [])),
                "Is Processing": st.session_state.get("is_processing", False),
                "Current Error": st.session_state.get("chat_error"),
                "Chat Settings": st.session_state.get("chat_settings", {})
            }
            
            st.json(debug_data)
    
    @staticmethod
    def _get_display_messages() -> list:
        """Get messages formatted for display"""
        raw_messages = ChatStateManager.get_messages()
        max_messages = st.session_state["chat_settings"]["max_messages"]
        
        # Convert to MessageData objects if needed
        display_messages = []
        
        for msg in raw_messages[-max_messages:]:  # Show only recent messages
            if isinstance(msg, MessageData):
                display_messages.append(msg)
            else:
                # Convert from session state format to MessageData
                display_messages.append(create_message_data(
                    role=getattr(msg, 'role', 'unknown'),
                    content=getattr(msg, 'content', str(msg)),
                    message_id=getattr(msg, 'id', None),
                    metadata=getattr(msg, 'metadata', None)
                ))
        
        return display_messages
    
    @staticmethod
    def _handle_message_submission_safe(message: str, ai_handler: Optional[Callable[[str], str]]) -> None:
        """Handle message submission with comprehensive error recovery"""
        # Create snapshot before processing
        snapshot_id = create_error_snapshot("message_submission")
        
        try:
            # Validate message
            if not message or not message.strip():
                ChatStateManager.set_error("Please enter a message before submitting.")
                return
            
            # Add user message
            user_message = create_message_data("user", message.strip())
            ChatStateManager.add_message("user", message.strip())
            
            # Add to input history
            InputHistory.add_to_history(message.strip())
            
            # Set processing state
            ChatStateManager.set_processing(True)
            ChatStateManager.clear_error()
            
            logging.getLogger(__name__).info(f"Message submitted successfully: {message[:50]}...")
            
            # Rerun to show user message and processing state
            st.rerun()
            
        except Exception as e:
            # Handle error with comprehensive recovery
            error_handler = get_error_handler()
            error_info = error_handler.handle_error(
                error=e,
                category=ErrorCategory.UI,
                severity=ErrorSeverity.MEDIUM,
                user_message=f"Failed to submit message: {str(e)}",
                context={
                    "operation": "message_submission",
                    "message_length": len(message) if message else 0,
                    "snapshot_id": snapshot_id
                }
            )
            
            # Set chat-specific error for UI display
            ChatStateManager.set_error(f"Message submission failed: {str(e)}")
            ChatStateManager.set_processing(False)
            
            logging.getLogger(__name__).error(f"Message submission failed: {e}", exc_info=True)
            st.rerun()
    
    @staticmethod
    def _handle_background_processing_safe(
        ai_handler: Optional[Callable[[str], str]], 
        enable_streaming: bool
    ) -> None:
        """Handle AI response processing with comprehensive error recovery"""
        
        if not ChatStateManager.is_processing():
            return
        
        messages = ChatStateManager.get_messages()
        if not messages:
            ChatStateManager.set_processing(False)
            return
        
        # Get the last user message
        last_message = messages[-1]
        
        if last_message.role != "user":
            ChatStateManager.set_processing(False)
            return
        
        # Create snapshot before AI processing
        snapshot_id = create_error_snapshot("ai_processing")
        
        # Process AI response with error recovery
        try:
            start_time = datetime.now()
            
            if ai_handler:
                # Use custom AI handler with error handling
                response = ChatContainer._call_ai_handler_safe(ai_handler, last_message.content)
            else:
                # Use default placeholder response
                response = ChatContainer._generate_placeholder_response(last_message.content)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Add AI response with metadata
            ChatStateManager.add_message("assistant", response)
            ChatStateManager.set_processing(False)
            
            logging.getLogger(__name__).info(f"AI processing completed in {processing_time:.2f}s")
            
            # Rerun to show response
            st.rerun()
            
        except Exception as e:
            # Handle AI processing error with recovery options
            error_handler = get_error_handler()
            error_info = error_handler.handle_error(
                error=e,
                category=ErrorCategory.AI_API,
                severity=ErrorSeverity.HIGH,
                user_message=f"AI processing failed: {str(e)}. You can retry the request or try a different message.",
                context={
                    "operation": "ai_processing",
                    "user_message": last_message.content[:100],
                    "snapshot_id": snapshot_id,
                    "enable_streaming": enable_streaming
                }
            )
            
            # Set chat error for UI display
            ChatStateManager.set_error(f"AI processing failed: {str(e)}")
            ChatStateManager.set_processing(False)
            
            logging.getLogger(__name__).error(f"AI processing failed: {e}", exc_info=True)
            st.rerun()
    
    @staticmethod
    def _generate_placeholder_response(user_message: str) -> str:
        """Generate AI response using real LLM integration"""
        from src.integrations.llm_chat_bridge import handle_ai_response_sync
        return handle_ai_response_sync(user_message)
    
    @staticmethod
    def _call_ai_handler_safe(ai_handler: Callable[[str], str], message: str) -> str:
        """Safely call AI handler with error recovery"""
        operation_id = f"ai_call_{hash(message)}"
        error_handler = get_error_handler()
        
        # Check if we can retry
        if not error_handler.can_retry(operation_id):
            raise Exception("Maximum retry attempts exceeded for this AI request")
        
        try:
            response = ai_handler(message)
            error_handler.reset_retry_count(operation_id)
            return response
        except Exception as e:
            error_handler.record_retry(operation_id)
            raise e
    
    @staticmethod
    def _attempt_chat_recovery() -> None:
        """Attempt automatic chat recovery"""
        try:
            recovery_manager = get_recovery_manager()
            
            with st.spinner("Attempting recovery..."):
                # Try to recover critical chat state
                if recovery_manager.recover_critical_state():
                    st.success("🎉 Chat state recovered successfully!")
                    ChatStateManager.clear_error()
                    st.rerun()
                else:
                    st.warning("⚠️ Automatic recovery failed. Try manual reset.")
        except Exception as e:
            st.error(f"❌ Recovery failed: {str(e)}")
            logging.getLogger(__name__).error(f"Chat recovery failed: {e}")
    
    @staticmethod
    def _reset_chat_state() -> None:
        """Reset chat state to clean state"""
        try:
            with st.spinner("Resetting chat state..."):
                # Clear messages and errors
                ChatStateManager.clear_messages()
                ChatStateManager.clear_error()
                ChatStateManager.set_processing(False)
                
                # Clear input history
                from src.components.chat.chat_input import clear_input_state
                clear_input_state()
                
                # Reset error boundaries
                from src.components.common.error_boundary import reset_all_error_boundaries
                reset_count = reset_all_error_boundaries()
                
                st.success(f"🔄 Chat state reset successfully! ({reset_count} error boundaries cleared)")
                logging.getLogger(__name__).info("Chat state reset by user")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Reset failed: {str(e)}")
            logging.getLogger(__name__).error(f"Chat state reset failed: {e}")
    
    @staticmethod
    def _reset_error_system() -> None:
        """Reset the entire error handling system"""
        try:
            with st.spinner("Resetting error system..."):
                # Clear all error-related session state
                error_keys = [key for key in st.session_state.keys() if "error" in key.lower()]
                for key in error_keys:
                    del st.session_state[key]
                
                # Reset error handler
                error_handler = get_error_handler()
                error_handler.error_history.clear()
                error_handler.retry_counts.clear()
                
                # Reset recovery manager
                recovery_manager = get_recovery_manager()
                recovery_manager.cleanup_snapshots(max_age_hours=0)
                
                st.success("🔧 Error system reset successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error system reset failed: {str(e)}")
            logging.getLogger(__name__).error(f"Error system reset failed: {e}")
    
    @staticmethod
    def _retry_last_action() -> None:
        """Retry the last failed action with enhanced logic"""
        try:
            messages = ChatStateManager.get_messages()
            
            if messages and messages[-1].role == "user":
                # Retry processing the last user message
                ChatStateManager.clear_error()
                ChatStateManager.set_processing(True)
                
                # Clear current error from error handler
                error_handler = get_error_handler()
                error_handler.clear_current_error()
                
                logging.getLogger(__name__).info("Retrying last user message")
                st.rerun()
            else:
                ChatStateManager.set_error("No user message to retry")
        except Exception as e:
            ChatStateManager.set_error(f"Retry failed: {str(e)}")
            logging.getLogger(__name__).error(f"Retry action failed: {e}")


class ChatContainerConfig:
    """Configuration class for chat container customization"""
    
    def __init__(self):
        self.enable_streaming = True
        self.show_quick_actions = True
        self.show_conversation_stats = True
        self.custom_css = None
        self.ai_handler = None
        self.max_messages = 50
        self.enable_debug = False
    
    @classmethod
    def create_default(cls) -> 'ChatContainerConfig':
        """Create default configuration"""
        return cls()
    
    @classmethod
    def create_minimal(cls) -> 'ChatContainerConfig':
        """Create minimal configuration"""
        config = cls()
        config.show_quick_actions = False
        config.show_conversation_stats = False
        config.max_messages = 20
        return config
    
    @classmethod
    def create_debug(cls) -> 'ChatContainerConfig':
        """Create debug configuration"""
        config = cls()
        config.enable_debug = True
        return config


# Utility functions for chat container
def render_chat_with_config(config: ChatContainerConfig) -> None:
    """Render chat interface with provided configuration"""
    # Set session state based on config
    if config.enable_debug:
        st.session_state["debug_mode"] = True
    
    if config.max_messages:
        st.session_state["chat_settings"]["max_messages"] = config.max_messages
    
    # Render chat interface
    ChatContainer.render_chat_interface(
        ai_handler=config.ai_handler,
        enable_streaming=config.enable_streaming,
        show_quick_actions=config.show_quick_actions,
        show_conversation_stats=config.show_conversation_stats,
        custom_css=config.custom_css
    )


def clear_chat_state() -> None:
    """Clear all chat-related state"""
    ChatStateManager.clear_messages()
    ChatStateManager.clear_error()
    ChatStateManager.set_processing(False)
    
    # Clear input-related state
    from src.components.chat.chat_input import clear_input_state
    clear_input_state()


def get_chat_export_data() -> Dict[str, Any]:
    """Get chat data for export"""
    messages = ChatStateManager.get_messages()
    
    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in messages
        ],
        "exported_at": datetime.now().isoformat(),
        "total_messages": len(messages)
    }