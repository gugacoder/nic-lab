"""
Loading Indicators Component

This component provides various loading indicators and progress displays
for AI processing, message streaming, and other asynchronous operations.
"""

import streamlit as st
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta


class LoadingIndicators:
    """Collection of loading indicators for different use cases"""
    
    @staticmethod
    def render_typing_indicator(message: str = "AI is thinking...") -> None:
        """
        Render a typing indicator for chat messages
        
        Args:
            message: The message to display while typing
        """
        # Create animated typing dots
        dots_placeholder = st.empty()
        
        # Use session state to track animation frame
        if "typing_frame" not in st.session_state:
            st.session_state.typing_frame = 0
        
        # Animated dots pattern
        dots_patterns = ["   ", ".  ", ".. ", "..."]
        current_dots = dots_patterns[st.session_state.typing_frame % len(dots_patterns)]
        
        # Render the typing indicator
        typing_html = f"""
        <div style="
            display: flex;
            align-items: center;
            padding: 0.5rem 1rem;
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid #6c757d;
        ">
            <div style="
                width: 8px;
                height: 8px;
                background-color: #6c757d;
                border-radius: 50%;
                margin-right: 4px;
                animation: pulse 1.5s ease-in-out infinite;
            "></div>
            <div style="
                width: 8px;
                height: 8px;
                background-color: #6c757d;
                border-radius: 50%;
                margin-right: 4px;
                animation: pulse 1.5s ease-in-out infinite 0.3s;
            "></div>
            <div style="
                width: 8px;
                height: 8px;
                background-color: #6c757d;
                border-radius: 50%;
                margin-right: 12px;
                animation: pulse 1.5s ease-in-out infinite 0.6s;
            "></div>
            <span style="color: #6c757d; font-style: italic;">{message}</span>
        </div>
        
        <style>
        @keyframes pulse {{
            0%, 70%, 100% {{
                transform: scale(1);
                opacity: 0.7;
            }}
            35% {{
                transform: scale(1.2);
                opacity: 1;
            }}
        }}
        </style>
        """
        
        dots_placeholder.markdown(typing_html, unsafe_allow_html=True)
        
        # Update animation frame
        st.session_state.typing_frame += 1
    
    @staticmethod
    def render_processing_spinner(message: str = "Processing...") -> None:
        """
        Render a processing spinner for longer operations
        
        Args:
            message: The message to display while processing
        """
        with st.spinner(message):
            st.empty()
    
    @staticmethod
    def render_progress_bar(progress: float, message: str = "Loading...") -> None:
        """
        Render a progress bar for operations with known progress
        
        Args:
            progress: Progress value between 0.0 and 1.0
            message: The message to display
        """
        st.text(message)
        progress_bar = st.progress(progress)
        
        # Show percentage
        percentage = int(progress * 100)
        st.caption(f"{percentage}% complete")
    
    @staticmethod
    def render_streaming_indicator() -> None:
        """Render an indicator for streaming responses"""
        streaming_html = """
        <div style="
            display: flex;
            align-items: center;
            padding: 0.25rem 0.5rem;
            background-color: #e3f2fd;
            border-radius: 4px;
            margin: 0.25rem 0;
            border-left: 3px solid #2196f3;
        ">
            <div style="
                width: 6px;
                height: 6px;
                background-color: #2196f3;
                border-radius: 50%;
                margin-right: 8px;
                animation: blink 1s ease-in-out infinite;
            "></div>
            <span style="color: #1976d2; font-size: 0.875rem;">Streaming response...</span>
        </div>
        
        <style>
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
        </style>
        """
        
        st.markdown(streaming_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_connection_status(is_connected: bool, service_name: str = "Service") -> None:
        """
        Render connection status indicator
        
        Args:
            is_connected: Whether the service is connected
            service_name: Name of the service being checked
        """
        if is_connected:
            status_html = f"""
            <div style="
                display: inline-flex;
                align-items: center;
                padding: 0.25rem 0.5rem;
                background-color: #d4edda;
                color: #155724;
                border-radius: 4px;
                border: 1px solid #c3e6cb;
                font-size: 0.875rem;
            ">
                <span style="color: #28a745; margin-right: 4px;">●</span>
                {service_name} Connected
            </div>
            """
        else:
            status_html = f"""
            <div style="
                display: inline-flex;
                align-items: center;
                padding: 0.25rem 0.5rem;
                background-color: #f8d7da;
                color: #721c24;
                border-radius: 4px;
                border: 1px solid #f5c6cb;
                font-size: 0.875rem;
            ">
                <span style="color: #dc3545; margin-right: 4px;">●</span>
                {service_name} Disconnected
            </div>
            """
        
        st.markdown(status_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_loading_skeleton(num_lines: int = 3) -> None:
        """
        Render a loading skeleton for content that's being loaded
        
        Args:
            num_lines: Number of skeleton lines to show
        """
        skeleton_html = """
        <div style="margin: 1rem 0;">
        """
        
        for i in range(num_lines):
            width = "100%" if i < num_lines - 1 else "60%"
            skeleton_html += f"""
            <div style="
                height: 1rem;
                background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                background-size: 200% 100%;
                animation: loading 1.5s infinite;
                border-radius: 4px;
                margin-bottom: 0.5rem;
                width: {width};
            "></div>
            """
        
        skeleton_html += """
        </div>
        
        <style>
        @keyframes loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        </style>
        """
        
        st.markdown(skeleton_html, unsafe_allow_html=True)


class ProgressTracker:
    """Helper class for tracking and displaying operation progress"""
    
    def __init__(self, total_steps: int, operation_name: str = "Operation"):
        """
        Initialize progress tracker
        
        Args:
            total_steps: Total number of steps in the operation
            operation_name: Name of the operation being tracked
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.step_messages: List[str] = []
        self.start_time = datetime.now()
        
        # Initialize UI elements
        self.progress_container = st.empty()
        self.status_container = st.empty()
    
    def update_progress(self, step_message: str = "") -> None:
        """
        Update progress to next step
        
        Args:
            step_message: Message describing the current step
        """
        self.current_step += 1
        if step_message:
            self.step_messages.append(step_message)
        
        self._render_progress()
    
    def set_step(self, step_number: int, step_message: str = "") -> None:
        """
        Set progress to specific step
        
        Args:
            step_number: The step number to set (0-based)
            step_message: Message describing the step
        """
        self.current_step = step_number
        if step_message:
            # Ensure step_messages list is long enough
            while len(self.step_messages) <= step_number:
                self.step_messages.append("")
            self.step_messages[step_number] = step_message
        
        self._render_progress()
    
    def complete(self, success_message: str = "Operation completed successfully!") -> None:
        """
        Mark operation as complete
        
        Args:
            success_message: Message to display on completion
        """
        self.current_step = self.total_steps
        
        with self.progress_container.container():
            st.success(f"✅ {success_message}")
            
            elapsed_time = datetime.now() - self.start_time
            st.caption(f"Completed in {elapsed_time.total_seconds():.1f} seconds")
        
        self.status_container.empty()
    
    def error(self, error_message: str) -> None:
        """
        Mark operation as failed
        
        Args:
            error_message: Error message to display
        """
        with self.progress_container.container():
            st.error(f"❌ {error_message}")
            
            elapsed_time = datetime.now() - self.start_time
            st.caption(f"Failed after {elapsed_time.total_seconds():.1f} seconds")
        
        self.status_container.empty()
    
    def _render_progress(self) -> None:
        """Render the current progress state"""
        progress_value = min(self.current_step / self.total_steps, 1.0)
        
        with self.progress_container.container():
            st.text(f"{self.operation_name}")
            st.progress(progress_value)
            
            percentage = int(progress_value * 100)
            step_info = f"Step {self.current_step}/{self.total_steps} ({percentage}%)"
            
            if self.current_step > 0 and self.current_step <= len(self.step_messages):
                current_message = self.step_messages[self.current_step - 1]
                if current_message:
                    step_info += f" - {current_message}"
            
            st.caption(step_info)


# Utility functions for common loading scenarios
def with_loading(func, loading_message: str = "Loading..."):
    """
    Decorator-like function to wrap operations with loading indicator
    
    Args:
        func: Function to execute
        loading_message: Message to show while loading
    
    Returns:
        Function result
    """
    with st.spinner(loading_message):
        return func()


def show_processing_time(start_time: datetime, operation_name: str = "Operation") -> None:
    """
    Show how long an operation took
    
    Args:
        start_time: When the operation started
        operation_name: Name of the operation
    """
    elapsed = datetime.now() - start_time
    st.caption(f"⏱️ {operation_name} completed in {elapsed.total_seconds():.2f} seconds")