"""
Enhanced Streaming Message Component

Provides sophisticated real-time streaming display for AI responses with smooth
token buffering, progress indicators, and graceful interruption handling.
"""

import asyncio
import time
import streamlit as st
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from src.ai.streaming import StreamChunk, StreamProcessor, StreamState
from src.components.chat.message import MessageData, MessageComponent, create_message_data


@dataclass
class StreamingState:
    """Track streaming message state"""
    is_active: bool = False
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    token_count: int = 0
    char_count: int = 0
    accumulated_content: str = ""
    buffer: deque = field(default_factory=lambda: deque(maxlen=50))
    interrupted: bool = False
    error: Optional[str] = None


class EnhancedStreamingMessage:
    """Enhanced streaming message display with sophisticated buffering and UI updates"""
    
    def __init__(
        self,
        message_id: str,
        role: str = "assistant",
        buffer_size: int = 10,
        update_frequency: float = 0.05,  # 50ms updates
        show_progress: bool = True,
        show_metrics: bool = False
    ):
        self.message_id = message_id
        self.role = role
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.show_progress = show_progress
        self.show_metrics = show_metrics
        
        # State management
        self.state = StreamingState()
        self.placeholder: Optional[st.empty] = None
        self.progress_placeholder: Optional[st.empty] = None
        self.metrics_placeholder: Optional[st.empty] = None
        
        # UI configuration
        self.role_config = MessageComponent._get_role_config(role)
        
        # Session state integration
        if f"streaming_{message_id}" not in st.session_state:
            st.session_state[f"streaming_{message_id}"] = self.state
    
    def initialize(self) -> None:
        """Initialize the streaming display containers"""
        if self.placeholder is None:
            # Create main content placeholder
            self.placeholder = st.empty()
            
            # Create progress indicator if enabled
            if self.show_progress:
                self.progress_placeholder = st.empty()
            
            # Create metrics display if enabled
            if self.show_metrics:
                self.metrics_placeholder = st.empty()
            
            self.state.is_active = True
            self.state.start_time = time.time()
    
    def add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to the streaming display"""
        if not self.state.is_active or self.state.interrupted:
            return
        
        current_time = time.time()
        
        # Update state
        if chunk.content:
            self.state.accumulated_content += chunk.content
            self.state.char_count += len(chunk.content)
            self.state.token_count += chunk.token_count if hasattr(chunk, 'token_count') else 1
            self.state.buffer.append((chunk.content, current_time))
        
        self.state.last_update = current_time
        
        # Check if we should update display
        time_since_update = current_time - getattr(self, '_last_display_update', 0)
        if (
            time_since_update >= self.update_frequency or
            chunk.is_final or
            len(self.state.buffer) >= self.buffer_size
        ):
            self._update_display()
            self._last_display_update = current_time
        
        # Handle completion
        if chunk.is_final:
            self.finalize(chunk.finish_reason, chunk.usage)
    
    def _update_display(self) -> None:
        """Update the streaming display with current content"""
        if not self.placeholder or not self.state.is_active:
            return
        
        try:
            with self.placeholder.container():
                self._render_streaming_header()
                self._render_streaming_content()
            
            # Update progress indicator
            if self.show_progress and self.progress_placeholder:
                self._render_progress_indicator()
            
            # Update metrics if enabled
            if self.show_metrics and self.metrics_placeholder:
                self._render_metrics()
                
        except Exception as e:
            self.state.error = str(e)
            self._render_error()
    
    def _render_streaming_header(self) -> None:
        """Render the message header with streaming indicators"""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**{self.role_config['icon']} {self.role_config['name']}**")
        
        with col2:
            # Animated typing indicator
            elapsed = time.time() - self.state.start_time
            dots = "." * (int(elapsed * 2) % 4)
            st.caption(f"🔄 Typing{dots}")
        
        with col3:
            # Interrupt button
            if st.button("⏹️", key=f"interrupt_{self.message_id}", help="Stop generation"):
                self.interrupt()
    
    def _render_streaming_content(self) -> None:
        """Render the streaming content with smooth updates"""
        # Apply message styling
        message_style = f"""
        <div style="
            background-color: {self.role_config['bg_color']};
            border-left: 4px solid {self.role_config['border_color']};
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
            color: {self.role_config['text_color']};
            min-height: 2rem;
            position: relative;
        ">
        """
        
        st.markdown(message_style, unsafe_allow_html=True)
        
        # Render accumulated content with cursor
        display_content = self.state.accumulated_content
        
        # Add typing cursor with animation
        cursor_char = self._get_animated_cursor()
        display_content += cursor_char
        
        # Render with markdown support
        try:
            st.markdown(display_content)
        except Exception:
            # Fallback to text if markdown fails
            st.text(display_content)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _get_animated_cursor(self) -> str:
        """Get animated typing cursor"""
        cursor_chars = ["▋", "▊", "▉", "█", "▉", "▊"]
        elapsed = time.time() - self.state.start_time
        cursor_index = int(elapsed * 4) % len(cursor_chars)
        return cursor_chars[cursor_index]
    
    def _render_progress_indicator(self) -> None:
        """Render progress indicator with statistics"""
        if not self.progress_placeholder:
            return
        
        with self.progress_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                elapsed = time.time() - self.state.start_time
                st.caption(f"⏱️ {elapsed:.1f}s")
            
            with col2:
                st.caption(f"📝 {self.state.char_count} chars")
            
            with col3:
                if elapsed > 0:
                    cps = self.state.char_count / elapsed
                    st.caption(f"⚡ {cps:.0f} c/s")
                else:
                    st.caption("⚡ 0 c/s")
            
            with col4:
                st.caption(f"🔢 {self.state.token_count} tokens")
    
    def _render_metrics(self) -> None:
        """Render detailed streaming metrics"""
        if not self.metrics_placeholder:
            return
        
        elapsed = time.time() - self.state.start_time
        
        with self.metrics_placeholder.expander("📊 Streaming Metrics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Duration", f"{elapsed:.2f}s")
                st.metric("Characters", self.state.char_count)
                st.metric("Tokens", self.state.token_count)
            
            with col2:
                cps = self.state.char_count / elapsed if elapsed > 0 else 0
                tps = self.state.token_count / elapsed if elapsed > 0 else 0
                st.metric("Chars/sec", f"{cps:.1f}")
                st.metric("Tokens/sec", f"{tps:.1f}")
                st.metric("Buffer Size", len(self.state.buffer))
    
    def _render_error(self) -> None:
        """Render error state"""
        if self.placeholder and self.state.error:
            with self.placeholder.container():
                st.error(f"⚠️ Streaming error: {self.state.error}")
                st.button("🔄 Retry", key=f"retry_{self.message_id}")
    
    def interrupt(self) -> None:
        """Interrupt the streaming process"""
        self.state.interrupted = True
        self.state.is_active = False
        
        # Show interruption message
        if self.placeholder:
            with self.placeholder.container():
                st.warning("⏹️ Generation interrupted by user")
                
                # Show partial content if any
                if self.state.accumulated_content:
                    with st.expander("📄 Partial Response", expanded=True):
                        st.markdown(self.state.accumulated_content)
    
    def finalize(self, finish_reason: Optional[str] = None, usage: Optional[Dict[str, int]] = None) -> MessageData:
        """Finalize the streaming message and convert to static message"""
        self.state.is_active = False
        
        # Create final message data
        final_message = create_message_data(
            role=self.role,
            content=self.state.accumulated_content,
            message_id=self.message_id,
            metadata={
                "streaming": True,
                "finish_reason": finish_reason,
                "usage": usage or {},
                "stream_duration": time.time() - self.state.start_time,
                "char_count": self.state.char_count,
                "token_count": self.state.token_count,
                "interrupted": self.state.interrupted
            }
        )
        
        # Replace streaming display with final message
        if self.placeholder:
            with self.placeholder.container():
                MessageComponent.render_message(final_message, show_actions=True)
        
        # Clear progress and metrics displays
        if self.progress_placeholder:
            self.progress_placeholder.empty()
        if self.metrics_placeholder:
            self.metrics_placeholder.empty()
        
        # Clean up session state
        session_key = f"streaming_{self.message_id}"
        if session_key in st.session_state:
            del st.session_state[session_key]
        
        return final_message
    
    def get_current_content(self) -> str:
        """Get current accumulated content"""
        return self.state.accumulated_content
    
    def is_active(self) -> bool:
        """Check if streaming is currently active"""
        return self.state.is_active and not self.state.interrupted


class StreamingMessageManager:
    """Manage multiple concurrent streaming messages"""
    
    def __init__(self):
        self.active_streams: Dict[str, EnhancedStreamingMessage] = {}
        self._cleanup_threshold = 100  # Maximum concurrent streams
    
    def create_stream(
        self,
        message_id: str,
        role: str = "assistant",
        **kwargs
    ) -> EnhancedStreamingMessage:
        """Create a new streaming message"""
        if message_id in self.active_streams:
            # Clean up existing stream
            self.cleanup_stream(message_id)
        
        stream = EnhancedStreamingMessage(message_id, role, **kwargs)
        stream.initialize()
        self.active_streams[message_id] = stream
        
        # Cleanup old streams if needed
        self._cleanup_old_streams()
        
        return stream
    
    def get_stream(self, message_id: str) -> Optional[EnhancedStreamingMessage]:
        """Get streaming message by ID"""
        return self.active_streams.get(message_id)
    
    def cleanup_stream(self, message_id: str) -> bool:
        """Cleanup completed stream"""
        if message_id in self.active_streams:
            stream = self.active_streams[message_id]
            if stream.placeholder:
                stream.placeholder.empty()
            if stream.progress_placeholder:
                stream.progress_placeholder.empty()
            if stream.metrics_placeholder:
                stream.metrics_placeholder.empty()
            
            del self.active_streams[message_id]
            return True
        return False
    
    def interrupt_all(self) -> None:
        """Interrupt all active streams"""
        for stream in self.active_streams.values():
            if stream.is_active():
                stream.interrupt()
    
    def _cleanup_old_streams(self) -> None:
        """Cleanup old inactive streams"""
        if len(self.active_streams) <= self._cleanup_threshold:
            return
        
        # Find inactive streams
        inactive_streams = [
            message_id for message_id, stream in self.active_streams.items()
            if not stream.is_active()
        ]
        
        # Remove oldest inactive streams
        for message_id in inactive_streams[:10]:  # Remove up to 10 at a time
            self.cleanup_stream(message_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        active_count = sum(1 for stream in self.active_streams.values() if stream.is_active())
        total_count = len(self.active_streams)
        
        return {
            "active_streams": active_count,
            "total_streams": total_count,
            "memory_usage": total_count * 1024  # Rough estimate
        }


# Global streaming manager instance
_streaming_manager: Optional[StreamingMessageManager] = None

def get_streaming_manager() -> StreamingMessageManager:
    """Get global streaming message manager"""
    global _streaming_manager
    if _streaming_manager is None:
        _streaming_manager = StreamingMessageManager()
    return _streaming_manager


# Convenience functions
def create_streaming_message(
    message_id: str,
    role: str = "assistant",
    **kwargs
) -> EnhancedStreamingMessage:
    """Create and initialize a streaming message"""
    manager = get_streaming_manager()
    return manager.create_stream(message_id, role, **kwargs)


def finalize_streaming_message(message_id: str) -> Optional[MessageData]:
    """Finalize a streaming message"""
    manager = get_streaming_manager()
    stream = manager.get_stream(message_id)
    if stream:
        return stream.finalize()
    return None


def interrupt_streaming_message(message_id: str) -> bool:
    """Interrupt a specific streaming message"""
    manager = get_streaming_manager()
    stream = manager.get_stream(message_id)
    if stream and stream.is_active():
        stream.interrupt()
        return True
    return False


def cleanup_streaming_messages() -> None:
    """Cleanup all streaming message resources"""
    manager = get_streaming_manager()
    manager.interrupt_all()
    for message_id in list(manager.active_streams.keys()):
        manager.cleanup_stream(message_id)