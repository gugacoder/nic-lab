"""
Chat Input Component

This component handles user input for the chat interface, including
multi-line text support, submit handling, keyboard shortcuts, and input validation.
"""

import streamlit as st
from typing import Optional, Callable, Dict, Any
import re
from datetime import datetime

from utils.session import ChatStateManager
from components.common.loading import LoadingIndicators


class ChatInputComponent:
    """Component for handling chat input with enhanced features"""
    
    @staticmethod
    def render_chat_input(
        placeholder: str = "Type your message here...",
        max_chars: int = 4000,
        disabled: bool = False,
        on_submit: Optional[Callable[[str], None]] = None,
        show_char_count: bool = True,
        show_suggestions: bool = True
    ) -> Optional[str]:
        """
        Render the chat input component with enhanced features
        
        Args:
            placeholder: Placeholder text for the input
            max_chars: Maximum number of characters allowed
            disabled: Whether the input should be disabled
            on_submit: Callback function when message is submitted
            show_char_count: Whether to show character count
            show_suggestions: Whether to show input suggestions
            
        Returns:
            The submitted message content, or None if no submission
        """
        # Check if currently processing
        is_processing = ChatStateManager.is_processing()
        input_disabled = disabled or is_processing
        
        # Show processing indicator if needed
        if is_processing:
            LoadingIndicators.render_typing_indicator("AI is processing your message...")
        
        # Create input container
        input_container = st.container()
        
        with input_container:
            # Show suggestions if enabled and not processing
            if show_suggestions and not is_processing:
                ChatInputComponent._render_input_suggestions()
            
            # Main input area
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Use chat_input for better UX
                user_input = st.chat_input(
                    placeholder=placeholder,
                    disabled=input_disabled,
                    key="main_chat_input"
                )
                
                # Alternative: Use text_area for multi-line support
                if st.session_state.get("use_multiline_input", False):
                    user_input = st.text_area(
                        "Message",
                        placeholder=placeholder,
                        max_chars=max_chars,
                        disabled=input_disabled,
                        key="multiline_input",
                        label_visibility="collapsed"
                    )
            
            with col2:
                # Input options and controls
                ChatInputComponent._render_input_controls(max_chars, show_char_count)
            
            # Handle input submission
            if user_input and not input_disabled:
                # Validate input
                validation_result = ChatInputComponent._validate_input(user_input, max_chars)
                
                if validation_result["valid"]:
                    # Process the input
                    processed_input = ChatInputComponent._process_input(user_input)
                    
                    # Call the submit callback or default handler
                    if on_submit:
                        on_submit(processed_input)
                    else:
                        ChatInputComponent._default_submit_handler(processed_input)
                    
                    return processed_input
                else:
                    # Show validation error
                    st.error(f"❌ {validation_result['error']}")
        
        return None
    
    @staticmethod
    def _render_input_suggestions() -> None:
        """Render input suggestions based on context"""
        suggestions = ChatInputComponent._get_contextual_suggestions()
        
        if suggestions:
            st.markdown("**💡 Try asking:**")
            
            # Create suggestion buttons
            cols = st.columns(min(len(suggestions), 3))
            
            for i, suggestion in enumerate(suggestions[:3]):
                with cols[i % 3]:
                    if st.button(
                        suggestion["text"],
                        key=f"suggestion_{i}",
                        help=suggestion.get("description", ""),
                        use_container_width=True
                    ):
                        # Set the suggestion as input
                        st.session_state["suggested_input"] = suggestion["text"]
                        st.rerun()
    
    @staticmethod
    def _render_input_controls(max_chars: int, show_char_count: bool) -> None:
        """Render input controls and options"""
        # Toggle for multiline input
        multiline = st.checkbox(
            "📝",
            value=st.session_state.get("use_multiline_input", False),
            help="Use multi-line input",
            key="multiline_toggle"
        )
        
        if multiline != st.session_state.get("use_multiline_input", False):
            st.session_state["use_multiline_input"] = multiline
            st.rerun()
        
        # Character count if enabled
        if show_char_count:
            current_input = st.session_state.get("multiline_input", "")
            if current_input:
                char_count = len(current_input)
                color = "red" if char_count > max_chars else "gray"
                st.markdown(
                    f"<small style='color: {color};'>{char_count}/{max_chars}</small>",
                    unsafe_allow_html=True
                )
        
        # Input format options
        with st.expander("⚙️ Options", expanded=False):
            # Input format
            input_format = st.selectbox(
                "Input format",
                options=["plain", "markdown"],
                index=0,
                key="input_format",
                help="Choose how to interpret your input"
            )
            
            # Auto-submit option
            auto_submit = st.checkbox(
                "Auto-submit on Enter",
                value=st.session_state.get("auto_submit", True),
                key="auto_submit_toggle",
                help="Automatically submit when pressing Enter"
            )
    
    @staticmethod
    def _get_contextual_suggestions() -> list:
        """Get contextual suggestions based on conversation history"""
        messages = ChatStateManager.get_messages()
        
        # Default suggestions for new conversations
        if not messages:
            return [
                {
                    "text": "What can you help me with?",
                    "description": "Learn about NIC Chat capabilities"
                },
                {
                    "text": "Search the knowledge base for...",
                    "description": "Search corporate documents"
                },
                {
                    "text": "Generate a document about...",
                    "description": "Create a new document"
                }
            ]
        
        # Contextual suggestions based on recent messages
        last_message = messages[-1] if messages else None
        
        if last_message and last_message.role == "assistant":
            return [
                {
                    "text": "Can you explain this further?",
                    "description": "Get more details about the response"
                },
                {
                    "text": "Show me related information",
                    "description": "Find related content"
                },
                {
                    "text": "Generate a document from this",
                    "description": "Create document from conversation"
                }
            ]
        
        return []
    
    @staticmethod
    def _validate_input(input_text: str, max_chars: int) -> Dict[str, Any]:
        """
        Validate user input
        
        Args:
            input_text: The input text to validate
            max_chars: Maximum allowed characters
            
        Returns:
            Dictionary with validation result
        """
        # Check if input is empty or only whitespace
        if not input_text or not input_text.strip():
            return {
                "valid": False,
                "error": "Please enter a message"
            }
        
        # Check character limit
        if len(input_text) > max_chars:
            return {
                "valid": False,
                "error": f"Message too long ({len(input_text)}/{max_chars} characters)"
            }
        
        # Check for potentially malicious content
        suspicious_patterns = [
            r'<script.*?>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                return {
                    "valid": False,
                    "error": "Input contains potentially unsafe content"
                }
        
        return {"valid": True}
    
    @staticmethod
    def _process_input(input_text: str) -> str:
        """
        Process and clean user input
        
        Args:
            input_text: Raw input text
            
        Returns:
            Processed input text
        """
        # Trim whitespace
        processed = input_text.strip()
        
        # Handle input format
        input_format = st.session_state.get("input_format", "plain")
        
        if input_format == "markdown":
            # Validate markdown syntax
            processed = ChatInputComponent._validate_markdown(processed)
        
        # Log input (in debug mode)
        if st.session_state.get("debug_mode", False):
            st.sidebar.text(f"Input processed: {len(processed)} chars")
        
        return processed
    
    @staticmethod
    def _validate_markdown(markdown_text: str) -> str:
        """
        Validate and clean markdown input
        
        Args:
            markdown_text: Input text with markdown
            
        Returns:
            Cleaned markdown text
        """
        # For now, just return as-is
        # In the future, could add markdown validation/sanitization
        return markdown_text
    
    @staticmethod
    def _default_submit_handler(message: str) -> None:
        """
        Default handler for message submission
        
        Args:
            message: The message to handle
        """
        # Add user message to chat
        ChatStateManager.add_message("user", message)
        
        # Set processing state
        ChatStateManager.set_processing(True)
        
        # Clear any previous errors
        ChatStateManager.clear_error()
        
        # Trigger rerun to update UI
        st.rerun()


class QuickActions:
    """Component for rendering quick action buttons"""
    
    @staticmethod
    def render_quick_actions() -> None:
        """Render quick action buttons above the input"""
        st.markdown("**⚡ Quick Actions:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 Search", help="Search the knowledge base"):
                st.session_state["suggested_input"] = "Search for: "
                st.rerun()
        
        with col2:
            if st.button("📝 Generate", help="Generate a document"):
                st.session_state["suggested_input"] = "Generate a document about: "
                st.rerun()
        
        with col3:
            if st.button("❓ Explain", help="Ask for explanation"):
                st.session_state["suggested_input"] = "Please explain: "
                st.rerun()
        
        with col4:
            if st.button("📊 Summarize", help="Request a summary"):
                st.session_state["suggested_input"] = "Summarize: "
                st.rerun()


class InputHistory:
    """Component for managing input history"""
    
    @staticmethod
    def add_to_history(input_text: str) -> None:
        """Add input to history"""
        if "input_history" not in st.session_state:
            st.session_state["input_history"] = []
        
        # Avoid duplicates and limit history size
        if input_text not in st.session_state["input_history"]:
            st.session_state["input_history"].insert(0, input_text)
            # Keep only last 20 items
            st.session_state["input_history"] = st.session_state["input_history"][:20]
    
    @staticmethod
    def render_history_selector() -> None:
        """Render input history selector"""
        if "input_history" not in st.session_state or not st.session_state["input_history"]:
            return
        
        with st.expander("📜 Recent Messages", expanded=False):
            for i, historic_input in enumerate(st.session_state["input_history"][:5]):
                # Truncate long messages for display
                display_text = historic_input[:50] + "..." if len(historic_input) > 50 else historic_input
                
                if st.button(
                    display_text,
                    key=f"history_{i}",
                    help=f"Reuse: {historic_input}",
                    use_container_width=True
                ):
                    st.session_state["suggested_input"] = historic_input
                    st.rerun()


# Utility functions
def handle_suggested_input() -> Optional[str]:
    """Handle suggested input from various sources"""
    suggested = st.session_state.get("suggested_input")
    
    if suggested:
        # Clear the suggestion
        del st.session_state["suggested_input"]
        return suggested
    
    return None


def clear_input_state() -> None:
    """Clear input-related session state"""
    keys_to_clear = [
        "main_chat_input",
        "multiline_input", 
        "suggested_input",
        "use_multiline_input"
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]