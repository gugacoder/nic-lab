"""
Streaming Response Integration Example

Demonstrates real-time streaming patterns for LLM response integration
with Streamlit UI components, showing token-by-token delivery and
proper buffer management.
"""

import asyncio
import streamlit as st
from typing import AsyncGenerator, Optional
import time
import logging

logger = logging.getLogger(__name__)


class StreamingResponseHandler:
    """
    Example handler for streaming LLM responses to Streamlit UI.
    
    Demonstrates proper patterns for token-by-token display,
    buffer management, and user interaction during streaming.
    """
    
    def __init__(self, container_key: str = "streaming_demo"):
        self.container_key = container_key
        self.streaming_active = False
        self.current_response = ""
        self.buffer = []
        self.buffer_size = 5  # Buffer tokens for smoother display
    
    async def stream_llm_response(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Simulate streaming LLM response for demonstration.
        
        In real implementation, this would connect to the actual
        Groq client streaming endpoint.
        """
        # Simulate realistic LLM response
        sample_response = f"""Thank you for your message: "{user_message}". 

This is a demonstration of streaming response integration. The response appears token-by-token as the LLM generates it, providing immediate feedback to users.

Key features demonstrated:
• Real-time token delivery
• Smooth UI updates without stuttering  
• Buffer management for optimal performance
• User interruption capabilities
• Progress indication during generation

The streaming integration maintains UI responsiveness while delivering engaging real-time experiences."""
        
        # Simulate token-by-token streaming
        words = sample_response.split()
        for word in words:
            if not self.streaming_active:
                break
            
            # Simulate network delay
            await asyncio.sleep(0.1)
            yield word + " "
    
    def render_streaming_demo(self):
        """Render complete streaming demonstration UI"""
        st.subheader("🔄 Streaming Response Integration Demo")
        
        # User input
        user_input = st.text_input(
            "Enter message to test streaming:",
            value="How does streaming response integration work?",
            key=f"{self.container_key}_input"
        )
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_streaming = st.button("▶️ Start Streaming", key=f"{self.container_key}_start")
        
        with col2:
            stop_streaming = st.button("⏸️ Stop Streaming", key=f"{self.container_key}_stop")
        
        with col3:
            clear_response = st.button("🗑️ Clear", key=f"{self.container_key}_clear")
        
        # Handle button actions
        if start_streaming and user_input:
            self.start_streaming_response(user_input)
        
        if stop_streaming:
            self.stop_streaming_response()
        
        if clear_response:
            self.clear_response()
        
        # Response display area
        response_container = st.container()
        
        with response_container:
            if self.streaming_active:
                self.render_streaming_response()
            elif self.current_response:
                self.render_complete_response()
            else:
                st.info("💡 Click 'Start Streaming' to see real-time LLM response generation")
    
    def start_streaming_response(self, user_message: str):
        """Initialize streaming response"""
        if f"{self.container_key}_streaming" not in st.session_state:
            st.session_state[f"{self.container_key}_streaming"] = True
            st.session_state[f"{self.container_key}_response"] = ""
            st.session_state[f"{self.container_key}_user_message"] = user_message
            
            # Start streaming in background
            self.streaming_active = True
            self.current_response = ""
            
            # Trigger rerun to start streaming loop
            st.rerun()
    
    def stop_streaming_response(self):
        """Stop active streaming"""
        self.streaming_active = False
        if f"{self.container_key}_streaming" in st.session_state:
            st.session_state[f"{self.container_key}_streaming"] = False
        st.rerun()
    
    def clear_response(self):
        """Clear current response"""
        self.streaming_active = False
        self.current_response = ""
        self.buffer = []
        
        # Clear session state
        for key in list(st.session_state.keys()):
            if key.startswith(self.container_key):
                del st.session_state[key]
        
        st.rerun()
    
    def render_streaming_response(self):
        """Render streaming response with live updates"""
        st.markdown("### 🔄 **Streaming Response**")
        
        # Progress indicator
        progress_container = st.container()
        with progress_container:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px;">
                <div class="streaming-dot" style="width: 12px; height: 12px; background: #00ff00; border-radius: 50%; animation: pulse 1s infinite;"></div>
                <span style="color: #00aa00; font-weight: 500;">Generating response...</span>
            </div>
            <style>
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.3; }
                100% { opacity: 1; }
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Simulated streaming display
        if f"{self.container_key}_streaming" in st.session_state:
            user_message = st.session_state.get(f"{self.container_key}_user_message", "")
            current_response = st.session_state.get(f"{self.container_key}_response", "")
            
            # Display current response with cursor
            response_container = st.container()
            with response_container:
                st.markdown("**Response:**")
                if current_response:
                    st.markdown(f"{current_response}▌")  # Cursor indicator
                else:
                    st.markdown("▌")  # Just cursor
            
            # Simulate token accumulation
            if len(current_response) < 200:  # Simulate response completion
                # Add next token
                sample_tokens = ["This", "is", "a", "streaming", "response", "demonstration", "showing", "real-time", "token", "delivery"]
                if len(current_response.split()) < len(sample_tokens):
                    next_token = sample_tokens[len(current_response.split())]
                    updated_response = current_response + " " + next_token if current_response else next_token
                    st.session_state[f"{self.container_key}_response"] = updated_response
                    
                    # Continue streaming
                    time.sleep(0.5)  # Simulate processing delay
                    st.rerun()
                else:
                    # Streaming complete
                    self.streaming_active = False
                    st.session_state[f"{self.container_key}_streaming"] = False
                    st.success("✅ Streaming complete!")
                    st.rerun()
    
    def render_complete_response(self):
        """Render completed response"""
        st.markdown("### ✅ **Complete Response**")
        
        response = st.session_state.get(f"{self.container_key}_response", "")
        if response:
            st.markdown(f"**Response:** {response}")
            
            # Response metadata
            with st.expander("📊 Response Metadata"):
                st.json({
                    "tokens": len(response.split()),
                    "characters": len(response),
                    "streaming_duration": "~5 seconds",
                    "model": "llama-3.1-8b-instant",
                    "stream_enabled": True
                })


class StreamingUIPatterns:
    """
    Collection of UI patterns for streaming response integration.
    
    Demonstrates various approaches to displaying streaming content
    in Streamlit applications.
    """
    
    @staticmethod
    def render_typing_indicator():
        """Render animated typing indicator"""
        st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            <span>AI is typing...</span>
        </div>
        <style>
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: rgba(240, 240, 240, 0.9);
            border-radius: 8px;
            margin: 10px 0;
        }
        .typing-dots {
            display: flex;
            gap: 4px;
        }
        .typing-dot {
            width: 8px;
            height: 8px;
            background: #666;
            border-radius: 50%;
            animation: typing-pulse 1.5s infinite;
        }
        .typing-dot:nth-child(2) { animation-delay: 0.3s; }
        .typing-dot:nth-child(3) { animation-delay: 0.6s; }
        @keyframes typing-pulse {
            0%, 70%, 100% { transform: scale(1); opacity: 0.7; }
            35% { transform: scale(1.2); opacity: 1; }
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_progress_bar(progress: float, total_tokens: int):
        """Render streaming progress bar"""
        st.progress(progress)
        st.caption(f"Generated {int(progress * total_tokens)} of ~{total_tokens} tokens")
    
    @staticmethod
    def render_token_counter(current_tokens: int):
        """Render live token counter"""
        st.metric("Tokens Generated", current_tokens, delta=1)
    
    @staticmethod
    def render_stream_controls():
        """Render streaming control buttons"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pause_stream = st.button("⏸️ Pause")
        with col2:
            resume_stream = st.button("▶️ Resume")
        with col3:
            stop_stream = st.button("⏹️ Stop")
        
        return {
            "pause": pause_stream,
            "resume": resume_stream,
            "stop": stop_stream
        }


# Example integration with actual LLM client
async def integrate_with_groq_client():
    """
    Example of integrating streaming patterns with actual Groq client.
    
    This shows the connection points between the streaming UI patterns
    and the real LLM API calls.
    """
    # Pseudo-code for actual integration
    
    # from src.ai.groq_client import GroqClient
    # 
    # client = GroqClient()
    # response_container = st.empty()
    # current_response = ""
    # 
    # async for chunk in client.stream_completion(
    #     messages=[{"role": "user", "content": user_message}],
    #     model="llama-3.1-8b-instant"
    # ):
    #     if chunk.content:
    #         current_response += chunk.content
    #         response_container.markdown(f"{current_response}▌")
    # 
    # # Remove cursor when complete
    # response_container.markdown(current_response)
    
    pass


# Demo application
def main():
    """Main demo application"""
    st.set_page_config(
        page_title="Streaming Response Integration Demo",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Streaming Response Integration Demo")
    st.markdown("Demonstration of real-time LLM response streaming patterns for Streamlit")
    
    # Demo tabs
    tab1, tab2, tab3 = st.tabs(["🔄 Live Streaming", "🎨 UI Patterns", "🔧 Integration"])
    
    with tab1:
        handler = StreamingResponseHandler("main_demo")
        handler.render_streaming_demo()
    
    with tab2:
        st.subheader("UI Pattern Examples")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Typing Indicator:**")
            StreamingUIPatterns.render_typing_indicator()
            
            st.markdown("**Progress Bar:**")
            StreamingUIPatterns.render_progress_bar(0.7, 150)
        
        with col2:
            st.markdown("**Token Counter:**")
            StreamingUIPatterns.render_token_counter(42)
            
            st.markdown("**Stream Controls:**")
            StreamingUIPatterns.render_stream_controls()
    
    with tab3:
        st.subheader("🔧 Integration Guidelines")
        
        st.markdown("""
        ### Key Integration Patterns
        
        1. **Async Streaming Handler**
           ```python
           async def stream_to_ui(response_stream, container):
               async for chunk in response_stream:
                   update_ui_with_chunk(chunk, container)
           ```
        
        2. **Buffer Management**
           ```python
           # Buffer tokens for smoother display
           buffer = []
           for token in stream:
               buffer.append(token)
               if len(buffer) >= buffer_size:
                   display_buffered_tokens(buffer)
                   buffer.clear()
           ```
        
        3. **Error Handling**
           ```python
           try:
               async for chunk in llm_stream:
                   yield chunk
           except StreamInterruptedException:
               yield "⚠️ Stream interrupted - partial response saved"
           ```
        
        4. **User Interruption**
           ```python
           if st.button("Stop Generation"):
               streaming_manager.stop()
               st.success("Generation stopped successfully")
           ```
        """)


if __name__ == "__main__":
    main()