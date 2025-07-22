"""
Streamlit Chat Example
Demonstrates basic chat interface implementation
"""

import streamlit as st
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="NIC Chat Example",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = datetime.now().isoformat()

if "processing" not in st.session_state:
    st.session_state.processing = False

# Header
st.title("💬 NIC Chat Interface Example")
st.caption("Demonstrating Streamlit chat components")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.subheader("Session Info")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    st.text(f"Messages: {len(st.session_state.messages)}")

# Main chat container
chat_container = st.container()

# Display messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"_{message['timestamp']}_")

# Input container
input_container = st.container()

# User input
with input_container:
    if prompt := st.chat_input("Type your message here...", disabled=st.session_state.processing):
        # Add user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        }
        st.session_state.messages.append(user_message)
        
        # Display user message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
                st.caption(f"_{timestamp}_")
        
        # Set processing state
        st.session_state.processing = True
        st.rerun()

# Simulate AI response
if st.session_state.processing:
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Simulate streaming response
            response = "Thank you for your message! This is a simulated response demonstrating the chat interface. "
            response += "In the actual implementation, this would connect to the Groq API for intelligent responses."
            
            displayed_text = ""
            for char in response:
                displayed_text += char
                message_placeholder.markdown(displayed_text + "▌")
                time.sleep(0.01)  # Simulate streaming delay
            
            message_placeholder.markdown(displayed_text)
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.caption(f"_{timestamp}_")
    
    # Save assistant message
    assistant_message = {
        "role": "assistant",
        "content": response,
        "timestamp": timestamp
    }
    st.session_state.messages.append(assistant_message)
    st.session_state.processing = False
    st.rerun()

# Footer
st.divider()
st.caption("This is an example implementation for the NIC Chat system")