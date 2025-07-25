"""
Message List Component

This component handles the display of conversation history with smooth scrolling,
message filtering, search functionality, and performance optimization for long conversations.
"""

import streamlit as st
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
import math

from src.utils.session import ChatStateManager, UIStateManager
from src.components.chat.message import MessageComponent, MessageData, render_message_separator
from src.components.common.loading import LoadingIndicators


class MessageListComponent:
    """Component for rendering conversation history with advanced features"""
    
    @staticmethod
    def render_message_list(
        messages: List[MessageData],
        max_height: str = "60vh",
        show_search: bool = True,
        show_filters: bool = True,
        show_timestamps: bool = True,
        enable_pagination: bool = True,
        messages_per_page: int = 50
    ) -> None:
        """
        Render the complete message list with all features
        
        Args:
            messages: List of messages to display
            max_height: Maximum height for the scrollable area
            show_search: Whether to show search functionality
            show_filters: Whether to show message filters
            show_timestamps: Whether to show message timestamps
            enable_pagination: Whether to enable pagination for long conversations
            messages_per_page: Number of messages per page when pagination is enabled
        """
        if not messages:
            MessageListComponent._render_empty_state()
            return
        
        # Show list controls if enabled
        if show_search or show_filters:
            filtered_messages = MessageListComponent._render_list_controls(
                messages, show_search, show_filters
            )
        else:
            filtered_messages = messages
        
        # Handle pagination if enabled
        if enable_pagination and len(filtered_messages) > messages_per_page:
            paginated_messages = MessageListComponent._handle_pagination(
                filtered_messages, messages_per_page
            )
        else:
            paginated_messages = filtered_messages
        
        # Render the scrollable message container
        MessageListComponent._render_scrollable_container(
            paginated_messages, max_height, show_timestamps
        )
        
        # Show conversation stats
        MessageListComponent._render_conversation_stats(messages, filtered_messages)
    
    @staticmethod
    def _render_empty_state() -> None:
        """Render empty state when no messages exist"""
        empty_container = st.container()
        
        with empty_container:
            st.markdown(
                """
                <div style="
                    text-align: center;
                    padding: 3rem 1rem;
                    color: #6c757d;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    border: 2px dashed #dee2e6;
                ">
                    <h3>👋 Welcome to NIC Chat!</h3>
                    <p>Start a conversation by typing a message below.</p>
                    <p><small>Your messages will appear here as you chat with the AI assistant.</small></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Show some conversation starters
            st.markdown("**💡 Try asking:**")
            
            starter_cols = st.columns(3)
            starters = [
                "What can you help me with?",
                "Search for recent documents",
                "Explain how this system works"
            ]
            
            for i, starter in enumerate(starters):
                with starter_cols[i]:
                    if st.button(starter, key=f"starter_{i}", use_container_width=True):
                        ChatStateManager.add_message("user", starter)
                        ChatStateManager.set_processing(True)
                        st.rerun()
    
    @staticmethod
    def _render_list_controls(
        messages: List[MessageData], 
        show_search: bool, 
        show_filters: bool
    ) -> List[MessageData]:
        """
        Render search and filter controls
        
        Args:
            messages: Original list of messages
            show_search: Whether to show search
            show_filters: Whether to show filters
            
        Returns:
            Filtered list of messages
        """
        controls_container = st.container()
        
        with controls_container:
            col1, col2 = st.columns([2, 1])
            
            filtered_messages = messages
            
            # Search functionality
            if show_search:
                with col1:
                    search_query = st.text_input(
                        "🔍 Search messages",
                        placeholder="Search conversation...",
                        key="message_search",
                        label_visibility="collapsed"
                    )
                    
                    if search_query:
                        filtered_messages = MessageListComponent._search_messages(
                            filtered_messages, search_query
                        )
            
            # Filter controls
            if show_filters:
                with col2:
                    with st.expander("🎛️ Filters", expanded=False):
                        # Role filter
                        selected_roles = st.multiselect(
                            "Show messages from:",
                            options=["user", "assistant", "system"],
                            default=["user", "assistant", "system"],
                            key="role_filter"
                        )
                        
                        # Time filter
                        time_filter = st.selectbox(
                            "Time range:",
                            options=["all", "last_hour", "today", "this_week"],
                            index=0,
                            key="time_filter"
                        )
                        
                        filtered_messages = MessageListComponent._apply_filters(
                            filtered_messages, selected_roles, time_filter
                        )
            
            # Show filter results
            if len(filtered_messages) != len(messages):
                st.info(f"📊 Showing {len(filtered_messages)} of {len(messages)} messages")
        
        return filtered_messages
    
    @staticmethod
    def _search_messages(messages: List[MessageData], query: str) -> List[MessageData]:
        """
        Search messages by content
        
        Args:
            messages: Messages to search
            query: Search query
            
        Returns:
            Filtered messages matching the query
        """
        if not query.strip():
            return messages
        
        query_lower = query.lower()
        
        return [
            msg for msg in messages
            if query_lower in msg.content.lower()
        ]
    
    @staticmethod
    def _apply_filters(
        messages: List[MessageData], 
        selected_roles: List[str], 
        time_filter: str
    ) -> List[MessageData]:
        """
        Apply role and time filters to messages
        
        Args:
            messages: Messages to filter
            selected_roles: Roles to include
            time_filter: Time range filter
            
        Returns:
            Filtered messages
        """
        filtered = messages
        
        # Apply role filter
        if selected_roles:
            filtered = [msg for msg in filtered if msg.role in selected_roles]
        
        # Apply time filter
        if time_filter != "all":
            now = datetime.now()
            
            if time_filter == "last_hour":
                cutoff = now - timedelta(hours=1)
            elif time_filter == "today":
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif time_filter == "this_week":
                cutoff = now - timedelta(days=7)
            else:
                cutoff = datetime.min
            
            filtered = [msg for msg in filtered if msg.timestamp >= cutoff]
        
        return filtered
    
    @staticmethod
    def _handle_pagination(
        messages: List[MessageData], 
        messages_per_page: int
    ) -> List[MessageData]:
        """
        Handle pagination for long conversations
        
        Args:
            messages: All messages
            messages_per_page: Messages to show per page
            
        Returns:
            Messages for current page
        """
        total_pages = math.ceil(len(messages) / messages_per_page)
        
        if total_pages <= 1:
            return messages
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            current_page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.get("current_message_page", total_pages),  # Default to last page
                key="message_page_input"
            )
            
            st.caption(f"Page {current_page} of {total_pages} ({len(messages)} messages)")
        
        # Calculate message range for current page
        start_idx = (current_page - 1) * messages_per_page
        end_idx = start_idx + messages_per_page
        
        return messages[start_idx:end_idx]
    
    @staticmethod
    def _render_scrollable_container(
        messages: List[MessageData], 
        max_height: str,
        show_timestamps: bool
    ) -> None:
        """
        Render the scrollable container with messages
        
        Args:
            messages: Messages to render
            max_height: Maximum height for scrolling
            show_timestamps: Whether to show timestamps
        """
        # Create scrollable container with custom CSS
        scroll_container_css = f"""
        <style>
        .message-container {{
            max-height: {max_height};
            overflow-y: auto;
            padding: 1rem;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            background-color: #ffffff;
        }}
        
        .message-container::-webkit-scrollbar {{
            width: 8px;
        }}
        
        .message-container::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 4px;
        }}
        
        .message-container::-webkit-scrollbar-thumb {{
            background: #c1c1c1;
            border-radius: 4px;
        }}
        
        .message-container::-webkit-scrollbar-thumb:hover {{
            background: #a8a8a8;
        }}
        </style>
        """
        
        st.markdown(scroll_container_css, unsafe_allow_html=True)
        
        # Container for messages
        messages_container = st.container()
        
        with messages_container:
            # Auto-scroll setting
            auto_scroll = UIStateManager.get_preference("auto_scroll", True)
            
            # Group messages by date if showing timestamps
            if show_timestamps:
                MessageListComponent._render_messages_with_date_groups(messages)
            else:
                MessageListComponent._render_messages_simple(messages)
            
            # Auto-scroll indicator
            if auto_scroll and messages:
                MessageListComponent._render_scroll_to_bottom()
    
    @staticmethod
    def _render_messages_with_date_groups(messages: List[MessageData]) -> None:
        """Render messages grouped by date"""
        if not messages:
            return
        
        current_date = None
        
        for i, message in enumerate(messages):
            message_date = message.timestamp.date()
            
            # Show date separator if date changed
            if current_date != message_date:
                MessageListComponent._render_date_separator(message_date)
                current_date = message_date
            
            # Render message
            MessageComponent.render_message(message)
            
            # Add separator between messages (except last)
            if i < len(messages) - 1:
                render_message_separator()
    
    @staticmethod
    def _render_messages_simple(messages: List[MessageData]) -> None:
        """Render messages without date grouping"""
        for i, message in enumerate(messages):
            MessageComponent.render_message(message)
            
            # Add separator between messages (except last)
            if i < len(messages) - 1:
                render_message_separator()
    
    @staticmethod
    def _render_date_separator(date: datetime.date) -> None:
        """Render a date separator"""
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        if date == today:
            date_text = "Today"
        elif date == yesterday:
            date_text = "Yesterday"
        else:
            date_text = date.strftime("%B %d, %Y")
        
        st.markdown(
            f"""
            <div style="
                text-align: center;
                margin: 1.5rem 0;
                color: #6c757d;
                font-size: 0.875rem;
                font-weight: 500;
            ">
                <div style="
                    display: inline-block;
                    padding: 0.25rem 1rem;
                    background-color: #f8f9fa;
                    border-radius: 16px;
                    border: 1px solid #dee2e6;
                ">
                    📅 {date_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def _render_scroll_to_bottom() -> None:
        """Render scroll to bottom indicator"""
        st.markdown(
            """
            <div style="
                text-align: center;
                margin: 1rem 0;
                opacity: 0.6;
            ">
                <small>⬇️ New messages appear here</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def _render_conversation_stats(
        original_messages: List[MessageData], 
        filtered_messages: List[MessageData]
    ) -> None:
        """Render conversation statistics"""
        with st.expander("📊 Conversation Stats", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Messages", len(original_messages))
                
                if filtered_messages != original_messages:
                    st.metric("Filtered Messages", len(filtered_messages))
            
            with col2:
                if original_messages:
                    user_msgs = len([m for m in original_messages if m.role == "user"])
                    assistant_msgs = len([m for m in original_messages if m.role == "assistant"])
                    
                    st.metric("Your Messages", user_msgs)
                    st.metric("AI Responses", assistant_msgs)
            
            with col3:
                if original_messages:
                    first_msg = min(original_messages, key=lambda m: m.timestamp)
                    duration = datetime.now() - first_msg.timestamp
                    
                    if duration.days > 0:
                        duration_text = f"{duration.days} days"
                    elif duration.seconds > 3600:
                        duration_text = f"{duration.seconds // 3600} hours"
                    else:
                        duration_text = f"{duration.seconds // 60} minutes"
                    
                    st.metric("Conversation Length", duration_text)


class MessageListActions:
    """Actions for the message list component"""
    
    @staticmethod
    def render_list_actions() -> None:
        """Render actions for the entire message list"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗑️ Clear All", help="Clear all messages"):
                MessageListActions._clear_all_messages()
        
        with col2:
            if st.button("📤 Export", help="Export conversation"):
                MessageListActions._export_conversation()
        
        with col3:
            if st.button("🔄 Refresh", help="Refresh message list"):
                st.rerun()
        
        with col4:
            if st.button("⬇️ Scroll Down", help="Scroll to bottom"):
                MessageListActions._scroll_to_bottom()
    
    @staticmethod
    def _clear_all_messages() -> None:
        """Clear all messages with confirmation"""
        if st.session_state.get("confirm_clear_all", False):
            ChatStateManager.clear_messages()
            st.session_state["confirm_clear_all"] = False
            st.success("🗑️ All messages cleared")
            st.rerun()
        else:
            st.session_state["confirm_clear_all"] = True
            st.warning("⚠️ Click 'Clear All' again to confirm")
    
    @staticmethod
    def _export_conversation() -> None:
        """Export conversation to markdown"""
        messages = ChatStateManager.get_messages()
        
        if not messages:
            st.warning("⚠️ No messages to export")
            return
        
        # Generate markdown content
        markdown_content = MessageListActions._generate_markdown_export(messages)
        
        # Provide download
        st.download_button(
            "💾 Download Conversation",
            data=markdown_content,
            file_name=f"nic_chat_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            help="Download conversation as Markdown file"
        )
    
    @staticmethod
    def _generate_markdown_export(messages: List[MessageData]) -> str:
        """Generate markdown export of conversation"""
        lines = [
            "# NIC Chat Conversation Export",
            f"\nExported on: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}",
            f"Total messages: {len(messages)}\n",
            "---\n"
        ]
        
        for message in messages:
            role_name = {
                "user": "**You**",
                "assistant": "**NIC Assistant**",
                "system": "**System**"
            }.get(message.role, f"**{message.role.title()}**")
            
            timestamp = message.timestamp.strftime('%H:%M:%S')
            
            lines.extend([
                f"## {role_name} ({timestamp})\n",
                f"{message.content}\n",
                "---\n"
            ])
        
        return "\n".join(lines)
    
    @staticmethod
    def _scroll_to_bottom() -> None:
        """Scroll to bottom of message list"""
        # This is handled client-side with JavaScript in a real implementation
        # For now, we'll use session state to indicate scroll preference
        st.session_state["scroll_to_bottom"] = True
        st.rerun()


# Utility functions
def get_message_list_height() -> str:
    """Get the optimal height for message list based on screen size"""
    # This would ideally use client-side detection
    # For now, return a reasonable default
    return "60vh"


def should_auto_scroll() -> bool:
    """Determine if messages should auto-scroll"""
    return UIStateManager.get_preference("auto_scroll", True)