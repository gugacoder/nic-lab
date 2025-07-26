"""
NIC Chat - Main Application

This is the main entry point for the NIC Chat Streamlit application.
It provides the foundational structure for the AI-powered corporate chat system.
"""

import streamlit as st
import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime
import atexit

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our modules
from config.settings import get_settings
from utils.session import (
    init_session_state, 
    SessionManager, 
    ChatStateManager, 
    UIStateManager,
    get_session_debug_info
)
from health_endpoint import start_health_server, stop_health_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load application settings
settings = get_settings()

# Start health check server
try:
    health_server = start_health_server()
    # Register cleanup function
    atexit.register(stop_health_server)
    logger.info("Health check server started successfully")
except Exception as e:
    logger.warning(f"Could not start health check server: {e}")


def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=settings.streamlit.page_title,
        page_icon=settings.streamlit.page_icon,
        layout=settings.streamlit.layout,
        initial_sidebar_state=settings.streamlit.initial_sidebar_state,
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': f"# {settings.app_name} v{settings.version}\n"
                    f"AI-powered corporate chat system\n\n"
                    f"Environment: {settings.environment}"
        }
    )


def render_header():
    """Render application header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title(f"💬 {settings.app_name}")
        st.caption("AI-Powered Corporate Knowledge Assistant")
        
        # Show environment indicator in debug mode
        if settings.is_debug():
            st.info(f"🔧 Running in {settings.environment} mode")


def render_sidebar():
    """Render application sidebar"""
    with st.sidebar:
        st.header("🔧 System")
        
        # Session information
        with st.expander("Session Info", expanded=False):
            session_info = SessionManager.get_session_info()
            st.text(f"Session: {session_info['session_id'][:8]}...")
            st.text(f"Messages: {session_info['message_count']}")
            st.text(f"Connected: {'✅' if session_info['gitlab_connected'] else '❌'}")
            
            if st.button("Clear Session", help="Clear all session data"):
                SessionManager.clear_session()
                st.rerun()
        
        # Navigation
        st.header("📱 Navigation")
        current_page = UIStateManager.get_page()
        
        # Page selection (for future expansion)
        pages = {
            "chat": "💬 Chat",
            "documents": "📄 Documents", 
            "settings": "⚙️ Settings"
        }
        
        selected_page = st.selectbox(
            "Page",
            options=list(pages.keys()),
            index=list(pages.keys()).index(current_page),
            format_func=lambda x: pages[x]
        )
        
        if selected_page != current_page:
            UIStateManager.set_page(selected_page)
            st.rerun()
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
            ChatStateManager.clear_messages()
            ChatStateManager.clear_error()
            st.rerun()
        
        # Configuration status
        st.header("🔧 Configuration")
        missing_config = settings.validate_required_settings()
        
        if missing_config:
            st.error(f"Missing: {', '.join(missing_config)}")
            st.info("Add missing environment variables to proceed")
        else:
            st.success("✅ Configuration complete")
        
        # Debug information (only in debug mode)
        if settings.is_debug():
            with st.expander("🐛 Debug Info", expanded=False):
                debug_info = get_session_debug_info()
                st.json(debug_info)


def render_chat_page():
    """Render the chat interface page using the new chat components"""
    # Import the chat container component
    from components.chat.chat_container import ChatContainer
    
    # Load custom CSS
    try:
        with open("src/styles/chat.css", "r") as f:
            css_content = f.read()
    except FileNotFoundError:
        css_content = None
    
    # Render the complete chat interface using our new components
    ChatContainer.render_chat_interface(
        ai_handler=_handle_ai_response,
        enable_streaming=True,
        show_quick_actions=True,
        show_conversation_stats=True,
        custom_css=css_content
    )


def _handle_ai_response(user_message: str) -> str:
    """Handle AI response generation using real LLM integration"""
    from src.integrations.llm_chat_bridge import handle_ai_response_sync
    return handle_ai_response_sync(user_message)


def render_documents_page():
    """Render the documents page (placeholder)"""
    st.header("📄 Document Management")
    st.info("🚧 Document generation features will be implemented in future tasks.")
    
    # Placeholder UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Documents")
        st.empty()
    
    with col2:
        st.subheader("Quick Actions")
        st.button("📝 Generate Document", disabled=True)
        st.button("📤 Export to GitLab", disabled=True)


def render_settings_page():
    """Render the settings page"""
    st.header("⚙️ Settings")
    
    # Application settings
    st.subheader("Application")
    
    # Theme preferences (using session state)
    theme = UIStateManager.get_preference("theme", "auto")
    new_theme = st.selectbox(
        "Theme",
        options=["auto", "light", "dark"],
        index=["auto", "light", "dark"].index(theme)
    )
    if new_theme != theme:
        UIStateManager.set_preference("theme", new_theme)
    
    # Chat settings
    st.subheader("Chat")
    
    max_messages = UIStateManager.get_preference("max_messages", 50)
    new_max_messages = st.slider(
        "Maximum messages to display",
        min_value=10,
        max_value=200,
        value=max_messages,
        step=10
    )
    if new_max_messages != max_messages:
        UIStateManager.set_preference("max_messages", new_max_messages)
    
    # Auto-scroll setting
    auto_scroll = UIStateManager.get_preference("auto_scroll", True)
    new_auto_scroll = st.checkbox("Auto-scroll to new messages", value=auto_scroll)
    if new_auto_scroll != auto_scroll:
        UIStateManager.set_preference("auto_scroll", new_auto_scroll)
    
    # Configuration information
    st.subheader("Configuration")
    
    with st.expander("Current Configuration", expanded=False):
        config_info = {
            "App Name": settings.app_name,
            "Version": settings.version,
            "Environment": settings.environment,
            "Debug Mode": settings.is_debug(),
            "GitLab URL": settings.gitlab.url or "Not configured",
            "Groq Model": settings.groq.model
        }
        
        for key, value in config_info.items():
            st.text(f"{key}: {value}")


def render_main_content():
    """Render main content based on current page"""
    page = UIStateManager.get_page()
    
    try:
        if page == "chat":
            render_chat_page()
        elif page == "documents":
            render_documents_page()
        elif page == "settings":
            render_settings_page()
        else:
            st.error(f"Unknown page: {page}")
            UIStateManager.set_page("chat")
            st.rerun()
    
    except Exception as e:
        st.error(f"Error rendering page: {str(e)}")
        logger.error(f"Page rendering error: {traceback.format_exc()}")
        
        if settings.is_debug():
            st.code(traceback.format_exc())


def render_footer():
    """Render application footer"""
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.caption(
            f"🤖 {settings.app_name} v{settings.version} | "
            f"Session: {SessionManager.get_session_info()['session_id'][:8]} | "
            f"Environment: {settings.environment}"
        )


def main():
    """Main application entry point"""
    try:
        # Configure page
        configure_page()
        
        # Initialize session state
        init_session_state()
        
        # Check for session expiry
        if SessionManager.is_session_expired():
            st.warning("⚠️ Your session has expired. Please refresh the page.")
            if st.button("Refresh Session"):
                SessionManager.clear_session()
                st.rerun()
            return
        
        # Render application layout
        render_header()
        
        # Main layout
        col1, col2 = st.columns([1, 4])
        
        with col1:
            render_sidebar()
        
        with col2:
            render_main_content()
        
        render_footer()
    
    except Exception as e:
        # Global error handler
        st.error("🚨 Application Error")
        st.error(f"An unexpected error occurred: {str(e)}")
        
        logger.error(f"Application error: {traceback.format_exc()}")
        
        if settings.is_debug():
            st.code(traceback.format_exc())
        
        if st.button("🔄 Restart Application"):
            SessionManager.clear_session()
            st.rerun()


if __name__ == "__main__":
    main()