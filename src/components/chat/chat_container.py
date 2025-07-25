"""
Chat Container Component

This is the main chat container that orchestrates all chat components including
message display, input handling, and user interactions. It provides the complete
chat interface experience.
"""

import streamlit as st
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import asyncio

from src.utils.session import ChatStateManager, UIStateManager
from src.components.chat.message import MessageData, create_message_data
from src.components.chat.message_list import MessageListComponent, MessageListActions
from src.components.chat.chat_input import ChatInputComponent, QuickActions, InputHistory
from src.components.common.loading import LoadingIndicators, ProgressTracker


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
        Render the complete chat interface
        
        Args:
            ai_handler: Function to handle AI responses
            enable_streaming: Whether to enable response streaming
            show_quick_actions: Whether to show quick action buttons
            show_conversation_stats: Whether to show conversation statistics
            custom_css: Custom CSS for styling
        """
        # Apply custom CSS if provided
        if custom_css:
            st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
        
        # Initialize chat state if needed
        ChatContainer._initialize_chat_state()
        
        # Handle any errors
        ChatContainer._render_error_display()
        
        # Render main chat layout
        ChatContainer._render_chat_layout(
            ai_handler=ai_handler,
            enable_streaming=enable_streaming,
            show_quick_actions=show_quick_actions,
            show_conversation_stats=show_conversation_stats
        )
        
        # Handle background processing
        ChatContainer._handle_background_processing(ai_handler, enable_streaming)
    
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
    def _render_error_display() -> None:
        """Render any chat errors"""
        error = ChatStateManager.get_error()
        
        if error:
            st.error(f"❌ {error}")
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if st.button("🔄 Retry", help="Retry the last action"):
                    ChatContainer._retry_last_action()
            
            with col2:
                if st.button("❌ Clear Error", help="Clear the error message"):
                    ChatStateManager.clear_error()
                    st.rerun()
    
    @staticmethod
    def _render_chat_layout(
        ai_handler: Optional[Callable[[str], str]],
        enable_streaming: bool,
        show_quick_actions: bool,
        show_conversation_stats: bool
    ) -> None:
        """Render the main chat layout with all components"""
        
        # Chat header with actions
        ChatContainer._render_chat_header()
        
        # Main chat area
        chat_col1, chat_col2 = st.columns([4, 1])
        
        with chat_col1:
            # Message history
            messages = ChatContainer._get_display_messages()
            
            MessageListComponent.render_message_list(
                messages=messages,
                show_search=True,
                show_filters=True,
                show_timestamps=st.session_state["chat_settings"]["show_timestamps"],
                enable_pagination=len(messages) > 50
            )
            
            # Message list actions
            MessageListActions.render_list_actions()
            
            # Quick actions
            if show_quick_actions and not ChatStateManager.is_processing():
                st.markdown("---")
                QuickActions.render_quick_actions()
            
            # Input history
            InputHistory.render_history_selector()
            
            # Chat input
            st.markdown("---")
            ChatContainer._render_input_section(ai_handler)
        
        with chat_col2:
            # Chat sidebar with settings and stats
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
    def _render_input_section(ai_handler: Optional[Callable[[str], str]]) -> None:
        """Render the input section with chat input component"""
        
        # Handle suggested input from various sources
        from src.components.chat.chat_input import handle_suggested_input
        suggested = handle_suggested_input()
        
        if suggested:
            # Auto-submit suggested input
            ChatContainer._handle_message_submission(suggested, ai_handler)
            return
        
        # Render chat input
        submitted_message = ChatInputComponent.render_chat_input(
            placeholder="Type your message here...",
            max_chars=4000,
            disabled=ChatStateManager.is_processing(),
            on_submit=lambda msg: ChatContainer._handle_message_submission(msg, ai_handler),
            show_char_count=True,
            show_suggestions=True
        )
        
        # Handle submission
        if submitted_message:
            ChatContainer._handle_message_submission(submitted_message, ai_handler)
    
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
    def _handle_message_submission(message: str, ai_handler: Optional[Callable[[str], str]]) -> None:
        """Handle message submission with AI processing"""
        try:
            # Add user message
            user_message = create_message_data("user", message)
            ChatStateManager.add_message("user", message)
            
            # Add to input history
            InputHistory.add_to_history(message)
            
            # Set processing state
            ChatStateManager.set_processing(True)
            ChatStateManager.clear_error()
            
            # Rerun to show user message and processing state
            st.rerun()
            
        except Exception as e:
            ChatStateManager.set_error(f"Failed to submit message: {str(e)}")
            ChatStateManager.set_processing(False)
            st.rerun()
    
    @staticmethod
    def _handle_background_processing(
        ai_handler: Optional[Callable[[str], str]], 
        enable_streaming: bool
    ) -> None:
        """Handle AI response processing in background"""
        
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
        
        # Process AI response
        try:
            if ai_handler:
                # Use custom AI handler
                response = ai_handler(last_message.content)
            else:
                # Use default placeholder response
                response = ChatContainer._generate_placeholder_response(last_message.content)
            
            # Add AI response
            ChatStateManager.add_message("assistant", response)
            ChatStateManager.set_processing(False)
            
            # Rerun to show response
            st.rerun()
            
        except Exception as e:
            ChatStateManager.set_error(f"AI processing failed: {str(e)}")
            ChatStateManager.set_processing(False)
            st.rerun()
    
    @staticmethod
    def _generate_placeholder_response(user_message: str) -> str:
        """Generate a placeholder response for testing"""
        import time
        time.sleep(1)  # Simulate processing time
        
        responses = [
            f"Thank you for your message: '{user_message}'. This is a placeholder response from the NIC Chat system.",
            f"I understand you're asking about: '{user_message}'. AI integration will be implemented in future tasks.",
            f"Your query about '{user_message}' has been received. This is a demonstration response.",
            f"Regarding '{user_message}': This is a test response from the chat interface components."
        ]
        
        import random
        return random.choice(responses)
    
    @staticmethod
    def _retry_last_action() -> None:
        """Retry the last failed action"""
        messages = ChatStateManager.get_messages()
        
        if messages and messages[-1].role == "user":
            # Retry processing the last user message
            ChatStateManager.clear_error()
            ChatStateManager.set_processing(True)
            st.rerun()
        else:
            ChatStateManager.set_error("No action to retry")


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