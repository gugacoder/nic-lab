"""
Common UI Components

This module provides reusable UI components that are shared across the application.
These components follow Streamlit best practices and maintain consistent styling.
"""

import streamlit as st
from typing import Optional, Dict, Any, List
from datetime import datetime


def render_loading_spinner(message: str = "Loading...") -> None:
    """Render a loading spinner with message"""
    with st.spinner(message):
        st.empty()


def render_error_message(error: str, show_details: bool = False) -> None:
    """Render an error message with optional details"""
    st.error(f"❌ {error}")
    
    if show_details and st.button("Show Details"):
        st.code(error)


def render_success_message(message: str) -> None:
    """Render a success message"""
    st.success(f"✅ {message}")


def render_info_message(message: str) -> None:
    """Render an info message"""
    st.info(f"ℹ️ {message}")


def render_warning_message(message: str) -> None:
    """Render a warning message"""
    st.warning(f"⚠️ {message}")


def render_status_indicator(status: str, label: str = "") -> None:
    """Render a status indicator"""
    status_icons = {
        "success": "🟢",
        "error": "🔴", 
        "warning": "🟡",
        "info": "🔵",
        "processing": "🟠"
    }
    
    icon = status_icons.get(status.lower(), "⚪")
    text = f"{icon} {label}" if label else icon
    st.markdown(text)


def render_timestamp(timestamp: datetime, format: str = "%Y-%m-%d %H:%M:%S") -> None:
    """Render a formatted timestamp"""
    formatted_time = timestamp.strftime(format)
    st.caption(f"🕐 {formatted_time}")


def render_divider_with_text(text: str) -> None:
    """Render a divider with centered text"""
    st.markdown(f"<div style='text-align: center; margin: 1rem 0;'><small>{text}</small></div>", unsafe_allow_html=True)
    st.divider()


def render_metric_card(title: str, value: str, delta: Optional[str] = None) -> None:
    """Render a metric card"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.metric(label=title, value=value, delta=delta)


def render_key_value_pairs(data: Dict[str, Any], title: Optional[str] = None) -> None:
    """Render key-value pairs in a formatted way"""
    if title:
        st.subheader(title)
    
    for key, value in data.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.text(f"{key}:")
        with col2:
            st.text(str(value))


# Placeholder for future components that will be implemented in other tasks
class ChatComponents:
    """Placeholder for chat-specific components"""
    pass


class DocumentComponents:
    """Placeholder for document-specific components"""
    pass


class FormComponents:
    """Placeholder for form-specific components"""
    pass