"""
Chat Message Component

This component handles the display of individual chat messages with role distinction,
markdown formatting, and message actions like copy, retry, and delete.
"""

import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from src.utils.session import ChatStateManager

# Optional dependency for clipboard functionality
try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False


@dataclass
class MessageData:
    """Data structure for a chat message"""
    id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class MessageComponent:
    """Component for rendering individual chat messages"""
    
    @staticmethod
    def render_message(message: MessageData, show_actions: bool = True) -> None:
        """
        Render a single message with appropriate styling and actions
        
        Args:
            message: The message data to render
            show_actions: Whether to show message actions (copy, retry, delete)
        """
        # Determine message styling based on role
        role_config = MessageComponent._get_role_config(message.role)
        
        # Create message container with appropriate styling
        with st.container():
            # Message header with role indicator and timestamp
            MessageComponent._render_message_header(message, role_config)
            
            # Message content with markdown support
            MessageComponent._render_message_content(message, role_config)
            
            # Message actions (copy, retry, delete)
            if show_actions:
                MessageComponent._render_message_actions(message)
    
    @staticmethod
    def _get_role_config(role: str) -> Dict[str, Any]:
        """Get configuration for message role styling"""
        configs = {
            "user": {
                "icon": "👤",
                "name": "You",
                "bg_color": "#f0f8ff",
                "border_color": "#4a90e2",
                "text_color": "#2c3e50"
            },
            "assistant": {
                "icon": "🤖",
                "name": "NIC Assistant",
                "bg_color": "#f8f9fa",
                "border_color": "#6c757d",
                "text_color": "#495057"
            },
            "system": {
                "icon": "⚙️",
                "name": "System",
                "bg_color": "#fff3cd",
                "border_color": "#ffc107",
                "text_color": "#856404"
            }
        }
        return configs.get(role, configs["assistant"])
    
    @staticmethod
    def _render_message_header(message: MessageData, role_config: Dict[str, Any]) -> None:
        """Render message header with role indicator and timestamp"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(
                f"**{role_config['icon']} {role_config['name']}**",
                help=f"Message from {role_config['name']}"
            )
        
        with col2:
            timestamp_str = message.timestamp.strftime("%H:%M:%S")
            st.caption(f"🕐 {timestamp_str}")
    
    @staticmethod
    def _render_message_content(message: MessageData, role_config: Dict[str, Any]) -> None:
        """Render message content with markdown support and proper styling"""
        # Apply custom styling for the message content
        message_style = f"""
        <div style="
            background-color: {role_config['bg_color']};
            border-left: 4px solid {role_config['border_color']};
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
            color: {role_config['text_color']};
        ">
        """
        
        # Render the styled container
        st.markdown(message_style, unsafe_allow_html=True)
        
        # Render content with markdown support
        try:
            st.markdown(message.content)
        except Exception as e:
            # Fallback to plain text if markdown fails
            st.text(message.content)
            if st.session_state.get("debug_mode", False):
                st.caption(f"⚠️ Markdown rendering failed: {str(e)}")
        
        # Close the styled container
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show metadata if available and in debug mode
        if message.metadata and st.session_state.get("debug_mode", False):
            with st.expander("🔍 Message Metadata", expanded=False):
                st.json(message.metadata)
    
    @staticmethod
    def _render_message_actions(message: MessageData) -> None:
        """Render message actions (copy, retry, delete)"""
        col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
        
        with col1:
            if st.button("📋", key=f"copy_{message.id}", help="Copy message"):
                MessageComponent._copy_message(message)
        
        with col2:
            if message.role == "assistant" and st.button("🔄", key=f"retry_{message.id}", help="Retry message"):
                MessageComponent._retry_message(message)
        
        with col3:
            if st.button("🗑️", key=f"delete_{message.id}", help="Delete message"):
                MessageComponent._delete_message(message)
    
    @staticmethod
    def _copy_message(message: MessageData) -> None:
        """Copy message content to clipboard"""
        if HAS_PYPERCLIP:
            try:
                # Try to copy to clipboard using pyperclip
                pyperclip.copy(message.content)
                st.success("📋 Message copied to clipboard!")
            except Exception as e:
                # Fallback: show content in a text area for manual copying
                st.warning(f"📋 Clipboard copy failed: {str(e)}")
                MessageComponent._show_copy_fallback(message)
        else:
            # No pyperclip available, show fallback
            MessageComponent._show_copy_fallback(message)
    
    @staticmethod
    def _show_copy_fallback(message: MessageData) -> None:
        """Show text area for manual copying when clipboard is not available"""
        st.info("📋 Select and copy the text below:")
        st.text_area(
            "Copy this text:",
            value=message.content,
            height=100,
            key=f"copy_fallback_{message.id}"
        )
    
    @staticmethod
    def _retry_message(message: MessageData) -> None:
        """Retry generating the assistant message"""
        try:
            # Find the user message that prompted this response
            messages = ChatStateManager.get_messages()
            message_index = next(
                (i for i, msg in enumerate(messages) if msg.id == message.id), 
                None
            )
            
            if message_index is not None and message_index > 0:
                user_message = messages[message_index - 1]
                if user_message.role == "user":
                    # Remove the assistant message and regenerate
                    ChatStateManager.remove_message(message.id)
                    
                    # Set processing state and rerun to trigger regeneration
                    ChatStateManager.set_processing(True)
                    st.rerun()
                else:
                    st.error("❌ Cannot find the user message to retry")
            else:
                st.error("❌ Cannot retry this message")
                
        except Exception as e:
            st.error(f"❌ Retry failed: {str(e)}")
    
    @staticmethod
    def _delete_message(message: MessageData) -> None:
        """Delete a message from the conversation"""
        try:
            # Confirm deletion
            if st.session_state.get(f"confirm_delete_{message.id}", False):
                ChatStateManager.remove_message(message.id)
                st.success("🗑️ Message deleted")
                st.rerun()
            else:
                st.session_state[f"confirm_delete_{message.id}"] = True
                st.warning("⚠️ Click delete again to confirm")
                
        except Exception as e:
            st.error(f"❌ Delete failed: {str(e)}")


class StreamingMessageComponent:
    """Component for rendering messages that are being streamed in real-time"""
    
    @staticmethod
    def render_streaming_message(role: str, content_placeholder: st.empty, 
                                current_content: str = "") -> None:
        """
        Render a message that is being streamed
        
        Args:
            role: The message role (user/assistant/system)
            content_placeholder: Streamlit placeholder for updating content
            current_content: The current content being streamed
        """
        role_config = MessageComponent._get_role_config(role)
        
        # Create the streaming message container
        with content_placeholder.container():
            # Header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{role_config['icon']} {role_config['name']}**")
            with col2:
                st.caption("🔄 Typing...")
            
            # Streaming content with typing indicator
            message_style = f"""
            <div style="
                background-color: {role_config['bg_color']};
                border-left: 4px solid {role_config['border_color']};
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 0 8px 8px 0;
                color: {role_config['text_color']};
            ">
            """
            
            st.markdown(message_style, unsafe_allow_html=True)
            
            # Current content with cursor
            display_content = current_content + "▋"  # Add typing cursor
            st.markdown(display_content)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    @staticmethod
    def finalize_streaming_message(content_placeholder: st.empty, 
                                  final_message: MessageData) -> None:
        """
        Finalize a streaming message by replacing it with the complete message
        
        Args:
            content_placeholder: The placeholder that was used for streaming
            final_message: The complete message data
        """
        with content_placeholder.container():
            MessageComponent.render_message(final_message, show_actions=True)


# Enhanced streaming integration
def create_enhanced_streaming_message(
    message_id: str,
    role: str = "assistant",
    **kwargs
):
    """
    Create enhanced streaming message using new streaming infrastructure
    
    This function provides backward compatibility while enabling enhanced features
    """
    try:
        from src.components.chat.streaming_message import create_streaming_message
        return create_streaming_message(message_id, role, **kwargs)
    except ImportError:
        # Fallback to basic streaming if enhanced version not available
        return None


def use_enhanced_streaming() -> bool:
    """Check if enhanced streaming is available"""
    try:
        from src.components.chat.streaming_message import EnhancedStreamingMessage
        return True
    except ImportError:
        return False


class StreamingBridge:
    """Bridge between old and new streaming systems"""
    
    def __init__(self, message_id: str, role: str = "assistant"):
        self.message_id = message_id
        self.role = role
        self.enhanced_stream = None
        self.placeholder = None
        
        # Try to use enhanced streaming
        if use_enhanced_streaming():
            self.enhanced_stream = create_enhanced_streaming_message(
                message_id=message_id,
                role=role,
                show_progress=True,
                show_metrics=False
            )
    
    def initialize(self) -> st.empty:
        """Initialize streaming display and return placeholder"""
        if self.enhanced_stream:
            # Enhanced streaming handles its own initialization
            return None
        else:
            # Fallback to basic streaming
            self.placeholder = st.empty()
            return self.placeholder
    
    def add_content(self, content: str) -> None:
        """Add content to the streaming display"""
        if self.enhanced_stream:
            # Use enhanced streaming
            from src.ai.streaming import StreamChunk
            chunk = StreamChunk(content=content)
            self.enhanced_stream.add_chunk(chunk)
        elif self.placeholder:
            # Fallback to basic display
            current_content = getattr(self, '_accumulated_content', '') + content
            self._accumulated_content = current_content
            StreamingMessageComponent.render_streaming_message(
                self.role, 
                self.placeholder, 
                current_content
            )
    
    def finalize(self, final_content: str = None) -> MessageData:
        """Finalize the streaming message"""
        if self.enhanced_stream:
            # Enhanced streaming finalization
            return self.enhanced_stream.finalize()
        elif self.placeholder:
            # Basic streaming finalization
            final_content = final_content or getattr(self, '_accumulated_content', '')
            final_message = create_message_data(
                role=self.role,
                content=final_content,
                message_id=self.message_id
            )
            StreamingMessageComponent.finalize_streaming_message(
                self.placeholder, 
                final_message
            )
            return final_message
        
        return create_message_data(
            role=self.role,
            content=final_content or "",
            message_id=self.message_id
        )
    
    def interrupt(self) -> None:
        """Interrupt the streaming"""
        if self.enhanced_stream:
            self.enhanced_stream.interrupt()


# Utility functions for message handling
def create_message_data(role: str, content: str, message_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> MessageData:
    """
    Create a MessageData object with proper ID generation
    
    Args:
        role: Message role (user/assistant/system)
        content: Message content
        message_id: Optional custom message ID
        metadata: Optional metadata dictionary
    
    Returns:
        MessageData object
    """
    import uuid
    
    if message_id is None:
        message_id = str(uuid.uuid4())
    
    return MessageData(
        id=message_id,
        role=role,
        content=content,
        timestamp=datetime.now(),
        metadata=metadata or {}
    )


def render_message_separator() -> None:
    """Render a subtle separator between messages"""
    st.markdown(
        "<div style='height: 1px; background: linear-gradient(90deg, transparent, #e0e0e0, transparent); margin: 0.5rem 0;'></div>",
        unsafe_allow_html=True
    )