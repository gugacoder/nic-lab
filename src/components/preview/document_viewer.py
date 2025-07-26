"""
Document Viewer Component

Enhanced document preview component that provides high-fidelity rendering
matching DOCX and PDF output formats with interactive controls and responsive design.
"""

import streamlit as st
import base64
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ViewMode(Enum):
    """Document view modes"""
    SINGLE_PAGE = "single"
    CONTINUOUS = "continuous"
    FACING_PAGES = "facing"
    PRINT_PREVIEW = "print"


@dataclass
class DocumentViewerConfig:
    """Configuration for document viewer"""
    show_zoom_controls: bool = True
    show_page_navigation: bool = True
    show_rulers: bool = True
    show_margins: bool = True
    initial_zoom: int = 100
    view_mode: ViewMode = ViewMode.CONTINUOUS
    enable_edit_mode: bool = False
    custom_css: Optional[str] = None
    page_size: str = "a4"  # "a4", "letter", "legal"


class DocumentViewer:
    """
    Main document viewer component with enhanced preview capabilities.
    
    Provides high-fidelity document rendering that accurately matches
    DOCX and PDF output formats.
    """
    
    @staticmethod
    def render_document_viewer(
        document_content: str,
        document_format: str = "html",
        config: Optional[DocumentViewerConfig] = None,
        on_edit: Optional[Callable[[str], None]] = None,
        container_key: str = "doc_viewer"
    ) -> Dict[str, Any]:
        """
        Render the complete document viewer interface.
        
        Args:
            document_content: The document content to display
            document_format: Format of the content ("html", "markdown", "docx", "pdf")
            config: Viewer configuration options
            on_edit: Callback for edit events
            container_key: Unique key for the container
            
        Returns:
            Dictionary with viewer state and user interactions
        """
        if config is None:
            config = DocumentViewerConfig()
        
        # Initialize session state for viewer
        viewer_state_key = f"{container_key}_state"
        if viewer_state_key not in st.session_state:
            st.session_state[viewer_state_key] = {
                "zoom_level": config.initial_zoom,
                "current_page": 1,
                "view_mode": config.view_mode,
                "show_rulers": config.show_rulers,
                "show_margins": config.show_margins,
                "edit_mode": False
            }
        
        viewer_state = st.session_state[viewer_state_key]
        
        # Load CSS styles
        DocumentViewer._load_document_styles()
        
        # Create main container
        with st.container():
            # Render header with controls
            interactions = DocumentViewer._render_viewer_header(
                config, viewer_state, container_key
            )
            
            # Render main document content
            DocumentViewer._render_document_content(
                document_content, document_format, config, viewer_state, container_key
            )
            
            # Handle edit mode if enabled
            if config.enable_edit_mode and viewer_state.get("edit_mode", False):
                edited_content = DocumentViewer._render_edit_interface(
                    document_content, container_key
                )
                if edited_content != document_content and on_edit:
                    on_edit(edited_content)
                    interactions["content_edited"] = edited_content
            
            return interactions
    
    @staticmethod
    def _load_document_styles():
        """Load document preview CSS styles"""
        try:
            with open("src/styles/document_preview.css", "r") as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        except FileNotFoundError:
            logger.warning("Document preview CSS not found, using basic styles")
            # Fallback basic styles
            st.markdown("""
            <style>
            .document-preview-container {
                width: 100%;
                background-color: #f5f5f5;
                padding: 2rem;
            }
            .document-page {
                background: white;
                box-shadow: 0 8px 24px rgba(0,0,0,0.15);
                margin: 0 auto 2rem;
                padding: 96px;
                width: 794px;
                min-height: 1123px;
            }
            </style>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def _render_viewer_header(
        config: DocumentViewerConfig,
        viewer_state: Dict[str, Any],
        container_key: str
    ) -> Dict[str, Any]:
        """Render the viewer header with controls"""
        interactions = {}
        
        with st.container():
            st.markdown('<div class="preview-header">', unsafe_allow_html=True)
            
            # Title and document info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<h3 class="preview-title">📄 Document Preview</h3>', 
                           unsafe_allow_html=True)
            
            with col2:
                # View mode selector
                if st.selectbox(
                    "View Mode",
                    options=list(ViewMode),
                    index=list(ViewMode).index(viewer_state["view_mode"]),
                    format_func=lambda x: x.value.title().replace("_", " "),
                    key=f"{container_key}_view_mode"
                ) != viewer_state["view_mode"]:
                    viewer_state["view_mode"] = st.session_state[f"{container_key}_view_mode"]
                    interactions["view_mode_changed"] = viewer_state["view_mode"]
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Controls row
            controls_col1, controls_col2, controls_col3 = st.columns([2, 2, 1])
            
            with controls_col1:
                if config.show_zoom_controls:
                    zoom_interactions = DocumentViewer._render_zoom_controls(
                        viewer_state, container_key
                    )
                    interactions.update(zoom_interactions)
            
            with controls_col2:
                if config.show_page_navigation:
                    nav_interactions = DocumentViewer._render_page_navigation(
                        viewer_state, container_key
                    )
                    interactions.update(nav_interactions)
            
            with controls_col3:
                # Display options
                display_options = DocumentViewer._render_display_options(
                    config, viewer_state, container_key
                )
                interactions.update(display_options)
        
        return interactions
    
    @staticmethod
    def _render_zoom_controls(viewer_state: Dict[str, Any], container_key: str) -> Dict[str, Any]:
        """Render zoom controls"""
        interactions = {}
        
        # Zoom control container
        zoom_col1, zoom_col2, zoom_col3, zoom_col4, zoom_col5 = st.columns([1, 1, 3, 1, 2])
        
        with zoom_col1:
            if st.button("🔍-", key=f"{container_key}_zoom_out", help="Zoom Out"):
                new_zoom = max(25, viewer_state["zoom_level"] - 25)
                viewer_state["zoom_level"] = new_zoom
                interactions["zoom_changed"] = new_zoom
        
        with zoom_col2:
            if st.button("🔍+", key=f"{container_key}_zoom_in", help="Zoom In"):
                new_zoom = min(400, viewer_state["zoom_level"] + 25)
                viewer_state["zoom_level"] = new_zoom
                interactions["zoom_changed"] = new_zoom
        
        with zoom_col3:
            new_zoom = st.slider(
                "Zoom",
                min_value=25,
                max_value=400,
                value=viewer_state["zoom_level"],
                step=25,
                key=f"{container_key}_zoom_slider",
                label_visibility="collapsed"
            )
            if new_zoom != viewer_state["zoom_level"]:
                viewer_state["zoom_level"] = new_zoom
                interactions["zoom_changed"] = new_zoom
        
        with zoom_col4:
            st.markdown(f'<div class="zoom-percentage">{viewer_state["zoom_level"]}%</div>', 
                       unsafe_allow_html=True)
        
        with zoom_col5:
            # Zoom presets
            preset_col1, preset_col2, preset_col3 = st.columns(3)
            
            with preset_col1:
                if st.button("Fit", key=f"{container_key}_zoom_fit", help="Fit to Width"):
                    viewer_state["zoom_level"] = 100
                    interactions["zoom_changed"] = 100
            
            with preset_col2:
                if st.button("100%", key=f"{container_key}_zoom_100", help="Actual Size"):
                    viewer_state["zoom_level"] = 100
                    interactions["zoom_changed"] = 100
            
            with preset_col3:
                if st.button("150%", key=f"{container_key}_zoom_150", help="150% Zoom"):
                    viewer_state["zoom_level"] = 150
                    interactions["zoom_changed"] = 150
        
        return interactions
    
    @staticmethod
    def _render_page_navigation(viewer_state: Dict[str, Any], container_key: str) -> Dict[str, Any]:
        """Render page navigation controls"""
        interactions = {}
        
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 2, 1, 1])
        
        with nav_col1:
            if st.button("⏮️", key=f"{container_key}_first_page", help="First Page"):
                viewer_state["current_page"] = 1
                interactions["page_changed"] = 1
        
        with nav_col2:
            page_col1, page_col2 = st.columns(2)
            
            with page_col1:
                if st.button("⏪", key=f"{container_key}_prev_page", help="Previous Page"):
                    new_page = max(1, viewer_state["current_page"] - 1)
                    viewer_state["current_page"] = new_page
                    interactions["page_changed"] = new_page
            
            with page_col2:
                if st.button("⏩", key=f"{container_key}_next_page", help="Next Page"):
                    new_page = viewer_state["current_page"] + 1  # Max will be handled by document
                    viewer_state["current_page"] = new_page
                    interactions["page_changed"] = new_page
        
        with nav_col3:
            new_page = st.number_input(
                "Page",
                min_value=1,
                value=viewer_state["current_page"],
                key=f"{container_key}_page_input",
                label_visibility="collapsed"
            )
            if new_page != viewer_state["current_page"]:
                viewer_state["current_page"] = new_page
                interactions["page_changed"] = new_page
        
        with nav_col4:
            st.markdown(f'<div class="page-info">of {viewer_state.get("total_pages", "?")}</div>', 
                       unsafe_allow_html=True)
        
        return interactions
    
    @staticmethod
    def _render_display_options(
        config: DocumentViewerConfig,
        viewer_state: Dict[str, Any],
        container_key: str
    ) -> Dict[str, Any]:
        """Render display options"""
        interactions = {}
        
        with st.expander("⚙️ Display Options"):
            # Rulers toggle
            show_rulers = st.checkbox(
                "Show Rulers",
                value=viewer_state["show_rulers"],
                key=f"{container_key}_rulers"
            )
            if show_rulers != viewer_state["show_rulers"]:
                viewer_state["show_rulers"] = show_rulers
                interactions["rulers_toggled"] = show_rulers
            
            # Margins toggle
            show_margins = st.checkbox(
                "Show Margins",
                value=viewer_state["show_margins"],
                key=f"{container_key}_margins"
            )
            if show_margins != viewer_state["show_margins"]:
                viewer_state["show_margins"] = show_margins
                interactions["margins_toggled"] = show_margins
            
            # Edit mode toggle (if enabled)
            if config.enable_edit_mode:
                edit_mode = st.checkbox(
                    "Edit Mode",
                    value=viewer_state.get("edit_mode", False),
                    key=f"{container_key}_edit_mode"
                )
                if edit_mode != viewer_state.get("edit_mode", False):
                    viewer_state["edit_mode"] = edit_mode
                    interactions["edit_mode_toggled"] = edit_mode
            
            # Print preview button
            if st.button("🖨️ Print Preview", key=f"{container_key}_print_preview"):
                viewer_state["view_mode"] = ViewMode.PRINT_PREVIEW
                interactions["print_preview_activated"] = True
        
        return interactions
    
    @staticmethod
    def _render_document_content(
        document_content: str,
        document_format: str,
        config: DocumentViewerConfig,
        viewer_state: Dict[str, Any],
        container_key: str
    ):
        """Render the main document content with styling"""
        
        # Determine zoom class
        zoom_level = viewer_state["zoom_level"]
        zoom_class = f"zoom-{zoom_level}"
        
        # Determine page size class
        page_size_class = f"{config.page_size}-size"
        
        # Build CSS classes
        container_classes = ["document-preview-container", zoom_class]
        if viewer_state["view_mode"] == ViewMode.PRINT_PREVIEW:
            container_classes.append("print-preview-mode")
        
        page_classes = ["document-page", page_size_class]
        
        # Start container
        st.markdown(f'<div class="{" ".join(container_classes)}">', unsafe_allow_html=True)
        
        # Render rulers if enabled
        if viewer_state["show_rulers"] and viewer_state["view_mode"] != ViewMode.PRINT_PREVIEW:
            st.markdown('<div class="ruler-horizontal"></div>', unsafe_allow_html=True)
        
        # Preview content area
        st.markdown('<div class="preview-content">', unsafe_allow_html=True)
        
        # Single page container
        st.markdown(f'<div class="{" ".join(page_classes)}">', unsafe_allow_html=True)
        
        # Render margin indicators if enabled
        if viewer_state["show_margins"] and viewer_state["view_mode"] != ViewMode.PRINT_PREVIEW:
            st.markdown('''
            <div class="margin-indicator top"></div>
            <div class="margin-indicator bottom"></div>
            <div class="margin-indicator left"></div>
            <div class="margin-indicator right"></div>
            ''', unsafe_allow_html=True)
        
        # Document content
        st.markdown('<div class="document-content">', unsafe_allow_html=True)
        
        # Render content based on format
        if document_format.lower() == "html":
            st.markdown(document_content, unsafe_allow_html=True)
        elif document_format.lower() == "markdown":
            st.markdown(document_content)
        else:
            # For other formats, display as text with basic formatting
            st.text(document_content)
        
        # Close document content
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Close page
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Close preview content
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Close container
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def _render_edit_interface(document_content: str, container_key: str) -> str:
        """Render edit interface for document content"""
        st.markdown("### ✏️ Edit Mode")
        st.info("📝 Make changes to the document content below. Changes will be reflected in the preview.")
        
        edited_content = st.text_area(
            "Document Content",
            value=document_content,
            height=400,
            key=f"{container_key}_editor",
            label_visibility="collapsed"
        )
        
        edit_col1, edit_col2 = st.columns(2)
        
        with edit_col1:
            if st.button("💾 Apply Changes", key=f"{container_key}_apply_edit"):
                st.success("✅ Changes applied to preview")
        
        with edit_col2:
            if st.button("🔄 Reset", key=f"{container_key}_reset_edit"):
                st.session_state[f"{container_key}_editor"] = document_content
                st.rerun()
        
        return edited_content
    
    @staticmethod
    def render_simple_preview(
        document_content: str,
        title: str = "Document Preview",
        zoom_level: int = 100
    ):
        """
        Render a simplified document preview without full controls.
        
        Args:
            document_content: HTML or markdown content to display
            title: Preview title
            zoom_level: Initial zoom level (25-400)
        """
        DocumentViewer._load_document_styles()
        
        # Simple header
        st.markdown(f'<h3 class="preview-title">📄 {title}</h3>', unsafe_allow_html=True)
        
        # Zoom class
        zoom_class = f"zoom-{zoom_level}"
        
        # Simple preview container
        st.markdown(f'''
        <div class="document-preview-container {zoom_class}">
            <div class="preview-content">
                <div class="document-page a4-size">
                    <div class="document-content">
                        {document_content}
                    </div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)


# Utility functions for common use cases

def render_document_from_file(
    file_path: str,
    config: Optional[DocumentViewerConfig] = None
) -> Dict[str, Any]:
    """Render document preview from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine format from file extension
        if file_path.endswith('.html'):
            format_type = "html"
        elif file_path.endswith('.md'):
            format_type = "markdown"
        else:
            format_type = "text"
        
        return DocumentViewer.render_document_viewer(content, format_type, config)
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
        return {}
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return {}


def render_document_from_bytes(
    document_bytes: bytes,
    document_format: str,
    config: Optional[DocumentViewerConfig] = None
) -> Dict[str, Any]:
    """Render document preview from bytes"""
    try:
        if document_format.lower() in ["html", "markdown", "text"]:
            content = document_bytes.decode('utf-8')
            return DocumentViewer.render_document_viewer(content, document_format, config)
        else:
            # For binary formats, show file info
            st.info(f"📄 Binary document ({document_format.upper()}) - {len(document_bytes)} bytes")
            st.markdown("Preview not available for binary formats")
            return {}
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        return {}