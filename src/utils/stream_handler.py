"""
Stream Handler Bridge

Bridges the advanced streaming infrastructure with Streamlit UI components,
providing seamless integration between Groq client streaming and message display.
"""

import asyncio
import uuid
import logging
from typing import AsyncGenerator, Optional, Dict, Any, Callable, List
from dataclasses import dataclass
import streamlit as st

from src.ai.streaming import StreamChunk, StreamProcessor, StreamState, get_stream_manager
from src.ai.groq_client import GroqClient, get_client
from src.components.chat.streaming_message import (
    EnhancedStreamingMessage,
    get_streaming_manager,
    create_streaming_message
)
from src.components.chat.message import MessageData, create_message_data
from src.utils.session import ChatStateManager

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for streaming operations"""
    buffer_size: int = 10
    update_frequency: float = 0.05  # 50ms
    show_progress: bool = True
    show_metrics: bool = False
    enable_interruption: bool = True
    auto_scroll: bool = True
    cache_response: bool = False


class StreamHandler:
    """
    High-level stream handler that orchestrates streaming between
    Groq API, streaming infrastructure, and UI components
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self._stream_counter = 0
    
    async def stream_completion(
        self,
        prompt: str,
        message_id: Optional[str] = None,
        role: str = "assistant",
        model: Optional[str] = None,
        **groq_kwargs
    ) -> MessageData:
        """
        Stream a completion with full UI integration
        
        Args:
            prompt: Input prompt for the AI
            message_id: Optional message ID (auto-generated if not provided)
            role: Message role (default: assistant)
            model: Model to use (optional)
            **groq_kwargs: Additional arguments for Groq API
        
        Returns:
            MessageData: Final completed message
        """
        # Generate message ID if not provided
        if message_id is None:
            self._stream_counter += 1
            message_id = f"stream_msg_{self._stream_counter}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting stream completion for message {message_id}")
        
        try:
            # Create streaming UI component
            streaming_msg = create_streaming_message(
                message_id=message_id,
                role=role,
                buffer_size=self.config.buffer_size,
                update_frequency=self.config.update_frequency,
                show_progress=self.config.show_progress,
                show_metrics=self.config.show_metrics
            )
            
            # Get Groq client
            groq_client = await get_client()
            
            # Configure streaming parameters
            stream_params = {
                "stream": True,
                "cache": self.config.cache_response,
                "model": model,
                **groq_kwargs
            }
            
            # Store active stream info
            self.active_streams[message_id] = {
                "streaming_msg": streaming_msg,
                "start_time": asyncio.get_event_loop().time(),
                "prompt": prompt,
                "config": self.config
            }
            
            # Start streaming from Groq
            groq_stream = await groq_client.complete(prompt, **stream_params)
            
            # Process stream with advanced handler
            final_message = await self._process_stream(
                message_id=message_id,
                groq_stream=groq_stream,
                streaming_msg=streaming_msg
            )
            
            logger.info(f"Completed stream for message {message_id}")
            return final_message
            
        except Exception as e:
            logger.error(f"Stream completion failed for {message_id}: {e}")
            
            # Handle error in UI
            if message_id in self.active_streams:
                streaming_msg = self.active_streams[message_id]["streaming_msg"]
                streaming_msg.state.error = str(e)
                streaming_msg._render_error()
            
            # Create error message
            error_message = create_message_data(
                role="system",
                content=f"⚠️ Stream error: {str(e)}",
                message_id=message_id,
                metadata={"error": True, "error_message": str(e)}
            )
            
            return error_message
        
        finally:
            # Cleanup
            if message_id in self.active_streams:
                del self.active_streams[message_id]
    
    async def _process_stream(
        self,
        message_id: str,
        groq_stream: AsyncGenerator,
        streaming_msg: EnhancedStreamingMessage
    ) -> MessageData:
        """Process the streaming response with UI updates"""
        
        # Callbacks for stream processor
        def chunk_callback(chunk: StreamChunk):
            """Handle each streaming chunk"""
            if not streaming_msg.state.interrupted:
                streaming_msg.add_chunk(chunk)
        
        def error_callback(error: Exception):
            """Handle streaming errors"""
            logger.error(f"Stream error for {message_id}: {error}")
            streaming_msg.state.error = str(error)
        
        def completion_callback(metrics):
            """Handle stream completion"""
            logger.info(f"Stream {message_id} completed with metrics: {metrics.to_dict()}")
        
        # Create stream processor
        stream_manager = await get_stream_manager()
        processor_id = await stream_manager.create_stream(
            stream_id=f"processor_{message_id}",
            chunk_callback=chunk_callback,
            error_callback=error_callback,
            completion_callback=completion_callback,
            buffer_size=self.config.buffer_size,
            enable_buffering=True
        )
        
        try:
            processor = await stream_manager.get_stream(processor_id)
            if not processor:
                raise RuntimeError("Failed to create stream processor")
            
            # Convert Groq stream to StreamChunk format
            async def chunk_adapter():
                async for groq_chunk in groq_stream:
                    # Convert Groq chunk to StreamChunk
                    stream_chunk = StreamChunk(
                        content=groq_chunk.content,
                        finish_reason=groq_chunk.finish_reason,
                        usage=groq_chunk.usage,
                        metadata={"groq_chunk": True}
                    )
                    yield stream_chunk
            
            # Process the stream
            final_chunk = None
            async for processed_chunk in processor.process_stream(chunk_adapter()):
                if processed_chunk.is_final:
                    final_chunk = processed_chunk
                
                # Check for user interruption
                if streaming_msg.state.interrupted:
                    processor.cancel()
                    break
            
            # Finalize the streaming message
            final_message = streaming_msg.finalize(
                finish_reason=final_chunk.finish_reason if final_chunk else "interrupted",
                usage=final_chunk.usage if final_chunk else None
            )
            
            return final_message
            
        except Exception as e:
            logger.error(f"Stream processing failed for {message_id}: {e}")
            # Create error message
            return create_message_data(
                role="system",
                content=f"⚠️ Processing error: {str(e)}",
                message_id=message_id,
                metadata={"error": True, "processing_error": str(e)}
            )
        
        finally:
            # Cleanup stream processor
            await stream_manager.cleanup_stream(processor_id)
    
    def interrupt_stream(self, message_id: str) -> bool:
        """Interrupt an active stream"""
        if message_id in self.active_streams:
            streaming_msg = self.active_streams[message_id]["streaming_msg"]
            streaming_msg.interrupt()
            return True
        return False
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream message IDs"""
        return list(self.active_streams.keys())
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get statistics about active streams"""
        active_count = len(self.active_streams)
        
        # Get stats from streaming manager
        streaming_manager = get_streaming_manager()
        ui_stats = streaming_manager.get_stats()
        
        return {
            "active_streams": active_count,
            "ui_streams": ui_stats,
            "stream_ids": list(self.active_streams.keys())
        }


class ChatStreamHandler:
    """Specialized stream handler for chat interface integration"""
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.base_handler = StreamHandler(config)
        self.chat_state = ChatStateManager()
    
    async def send_streaming_message(
        self,
        user_message: str,
        context: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> MessageData:
        """
        Send a user message and get streaming AI response
        
        Args:
            user_message: The user's message
            context: Optional conversation context
            **kwargs: Additional parameters for streaming
        
        Returns:
            MessageData: The AI's response message
        """
        try:
            # Add user message to chat state
            user_msg_data = create_message_data(
                role="user",
                content=user_message
            )
            self.chat_state.add_message(user_msg_data)
            
            # Prepare prompt with context
            if context:
                # Build context-aware prompt
                context_str = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in context[-10:]  # Last 10 messages
                ])
                prompt = f"Context:\n{context_str}\n\nUser: {user_message}\nAssistant:"
            else:
                prompt = user_message
            
            # Stream the response
            ai_response = await self.base_handler.stream_completion(
                prompt=prompt,
                role="assistant",
                **kwargs
            )
            
            # Add AI response to chat state
            if not ai_response.metadata.get("error", False):
                self.chat_state.add_message(ai_response)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Chat streaming failed: {e}")
            
            # Create error message
            error_message = create_message_data(
                role="system",
                content=f"⚠️ Chat error: {str(e)}",
                metadata={"error": True, "chat_error": str(e)}
            )
            
            return error_message
    
    def clear_chat(self) -> None:
        """Clear chat history and active streams"""
        # Interrupt all active streams
        for stream_id in self.base_handler.get_active_streams():
            self.base_handler.interrupt_stream(stream_id)
        
        # Clear chat state
        self.chat_state.clear_messages()
    
    def get_chat_history(self) -> List[MessageData]:
        """Get current chat history"""
        return self.chat_state.get_messages()


# Global handlers
_stream_handler: Optional[StreamHandler] = None
_chat_stream_handler: Optional[ChatStreamHandler] = None


def get_stream_handler(config: Optional[StreamConfig] = None) -> StreamHandler:
    """Get global stream handler instance"""
    global _stream_handler
    if _stream_handler is None:
        _stream_handler = StreamHandler(config)
    return _stream_handler


def get_chat_stream_handler(config: Optional[StreamConfig] = None) -> ChatStreamHandler:
    """Get global chat stream handler instance"""
    global _chat_stream_handler
    if _chat_stream_handler is None:
        _chat_stream_handler = ChatStreamHandler(config)
    return _chat_stream_handler


# Convenience functions for Streamlit integration
def stream_ai_response(
    prompt: str,
    show_progress: bool = True,
    show_metrics: bool = False,
    **kwargs
) -> MessageData:
    """
    Streamlit-friendly function to stream AI response
    
    Usage in Streamlit:
        response = stream_ai_response("Hello, how are you?")
    """
    config = StreamConfig(
        show_progress=show_progress,
        show_metrics=show_metrics
    )
    
    handler = get_stream_handler(config)
    
    # Use asyncio to run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(
            handler.stream_completion(prompt, **kwargs)
        )
    finally:
        loop.close()


def stream_chat_response(
    user_message: str,
    **kwargs
) -> MessageData:
    """
    Streamlit-friendly function for chat streaming
    
    Usage in Streamlit:
        response = stream_chat_response("What's the weather like?")
    """
    handler = get_chat_stream_handler()
    
    # Use asyncio to run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(
            handler.send_streaming_message(user_message, **kwargs)
        )
    finally:
        loop.close()


# Streamlit components integration
def render_streaming_chat():
    """Render a complete streaming chat interface"""
    st.header("🤖 NIC Assistant")
    
    # Initialize chat handler
    handler = get_chat_stream_handler(
        config=StreamConfig(
            show_progress=st.sidebar.checkbox("Show Progress", True),
            show_metrics=st.sidebar.checkbox("Show Metrics", False),
            enable_interruption=st.sidebar.checkbox("Enable Interruption", True)
        )
    )
    
    # Display chat history
    chat_history = handler.get_chat_history()
    for message in chat_history:
        with st.container():
            if message.role == "user":
                st.markdown(f"**👤 You:** {message.content}")
            elif message.role == "assistant":
                st.markdown(f"**🤖 Assistant:** {message.content}")
            elif message.role == "system":
                st.warning(message.content)
    
    # Chat input
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Show user message immediately
        st.markdown(f"**👤 You:** {user_input}")
        
        # Stream AI response
        with st.spinner("Generating response..."):
            response = stream_chat_response(user_input)
        
        # Refresh to show the response
        st.rerun()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🔧 Controls")
        
        if st.button("🗑️ Clear Chat"):
            handler.clear_chat()
            st.rerun()
        
        # Stream statistics
        stats = handler.base_handler.get_stream_stats()
        st.subheader("📊 Statistics")
        st.json(stats)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(f"Streaming response to: {prompt}")
        
        response = stream_ai_response(prompt, show_progress=True, show_metrics=True)
        print(f"\nFinal response: {response.content}")
        print(f"Metadata: {response.metadata}")
    else:
        print("Usage: python -m src.utils.stream_handler 'Your prompt here'")