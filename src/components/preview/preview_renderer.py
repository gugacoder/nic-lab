"""
Preview Renderer Component

High-level rendering functions for document preview with enhanced styling capabilities.
Provides unified interface for rendering documents with accurate DOCX/PDF styling.
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import base64
import io

from .document_viewer import DocumentViewer, DocumentViewerConfig, ViewMode
from .zoom_controls import ZoomControls, ZoomControlsConfig, ZoomLevel
from .page_display import PageDisplay, PageDisplayConfig, PageContent, PageSize, PageDisplayMode
from src.utils import style_mapper

logger = logging.getLogger(__name__)


class RenderMode(Enum):
    """Document rendering modes"""
    PREVIEW = "preview"
    EDIT = "edit"
    COMPARE = "compare"
    PRINT = "print"


@dataclass
class RenderConfig:
    """Complete configuration for document rendering"""
    render_mode: RenderMode = RenderMode.PREVIEW
    view_mode: ViewMode = ViewMode.CONTINUOUS
    page_display_mode: PageDisplayMode = PageDisplayMode.CONTINUOUS
    page_size: PageSize = PageSize.A4
    zoom_level: int = 100
    show_controls: bool = True
    show_rulers: bool = True
    show_margins: bool = True
    enable_editing: bool = False
    custom_styles: Optional[Dict[str, str]] = None
    style_template: Optional[str] = None


class PreviewRenderer:
    """
    Unified document preview renderer with enhanced styling capabilities.
    
    Provides high-level interface for rendering documents with accurate
    styling that matches DOCX and PDF output formats.
    """
    
    @staticmethod
    def render_document_preview(
        content: Union[str, bytes, List[PageContent]],
        content_type: str = "html",
        config: Optional[RenderConfig] = None,
        container_key: str = "doc_preview"
    ) -> Dict[str, Any]:
        """
        Render complete document preview with all features.
        
        Args:
            content: Document content (HTML, markdown, bytes, or pages)
            content_type: Type of content ("html", "markdown", "docx", "pdf", "pages")
            config: Rendering configuration
            container_key: Unique key for the container
            
        Returns:
            Dictionary with rendering state and user interactions
        """
        if config is None:
            config = RenderConfig()
        
        interactions = {}
        
        # Load and apply styles
        PreviewRenderer._load_preview_styles(config)
        
        # Process content based on type
        processed_content = PreviewRenderer._process_content(content, content_type)
        
        if config.render_mode == RenderMode.PRINT:
            # Special print preview mode
            return PreviewRenderer._render_print_preview(
                processed_content, config, container_key
            )
        elif config.render_mode == RenderMode.COMPARE:
            # Document comparison mode
            return PreviewRenderer._render_comparison_view(
                processed_content, config, container_key
            )
        else:
            # Standard preview or edit mode
            return PreviewRenderer._render_standard_preview(
                processed_content, content_type, config, container_key
            )
    
    @staticmethod
    def _load_preview_styles(config: RenderConfig):
        """Load and apply preview styles"""
        try:
            # Load base document preview CSS
            with open("src/styles/document_preview.css", "r") as f:
                base_css = f.read()
            
            # Apply style template if specified
            if config.style_template:
                template_css = PreviewRenderer._get_template_styles(config.style_template)
                combined_css = f"{base_css}\n\n{template_css}"
            else:
                combined_css = base_css
            
            # Apply custom styles if provided
            if config.custom_styles:
                custom_css = "\n".join([
                    f".{class_name} {{ {styles} }}"
                    for class_name, styles in config.custom_styles.items()
                ])
                combined_css = f"{combined_css}\n\n{custom_css}"
            
            st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)
            
        except FileNotFoundError:
            logger.warning("Document preview CSS not found, using minimal styles")
            PreviewRenderer._load_fallback_styles()
    
    @staticmethod
    def _load_fallback_styles():
        """Load minimal fallback styles if main CSS is not available"""
        fallback_css = """
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
        .document-content {
            font-family: "Times New Roman", serif;
            font-size: 12pt;
            line-height: 1.15;
        }
        """
        st.markdown(f"<style>{fallback_css}</style>", unsafe_allow_html=True)
    
    @staticmethod
    def _get_template_styles(template_name: str) -> str:
        """Get CSS styles for a specific template"""
        templates = {
            "corporate": """
            .doc-heading-1 { 
                color: #1f4788; 
                font-family: "Arial", sans-serif;
                border-bottom: 2px solid #1f4788;
                padding-bottom: 4pt;
            }
            .doc-heading-2 { 
                color: #2f5496; 
                font-family: "Arial", sans-serif;
            }
            .document-content {
                font-family: "Arial", sans-serif;
                font-size: 11pt;
            }
            """,
            
            "academic": """
            .doc-heading-1 { 
                color: #000000; 
                font-family: "Times New Roman", serif;
                text-align: center;
                margin: 18pt 0 12pt 0;
            }
            .doc-heading-2 { 
                color: #000000; 
                font-family: "Times New Roman", serif;
                margin: 12pt 0 6pt 0;
            }
            .document-content {
                font-family: "Times New Roman", serif;
                font-size: 12pt;
                line-height: 2.0;
                text-align: justify;
            }
            """,
            
            "modern": """
            .doc-heading-1 { 
                color: #333333; 
                font-family: "Calibri", sans-serif;
                font-weight: 300;
                font-size: 24pt;
                margin: 16pt 0 8pt 0;
            }
            .doc-heading-2 { 
                color: #666666; 
                font-family: "Calibri", sans-serif;
                font-weight: 400;
                font-size: 16pt;
            }
            .document-content {
                font-family: "Calibri", sans-serif;
                font-size: 11pt;
                color: #333333;
            }
            """
        }
        
        return templates.get(template_name, "")
    
    @staticmethod
    def _process_content(
        content: Union[str, bytes, List[PageContent]], 
        content_type: str
    ) -> Union[str, List[PageContent]]:
        """Process content based on type"""
        
        if content_type == "pages" and isinstance(content, list):
            return content
        
        if isinstance(content, bytes):
            if content_type.lower() in ["docx", "pdf"]:
                # For binary formats, we'd need specialized processing
                # For now, return a placeholder
                return f"Binary {content_type.upper()} document ({len(content)} bytes)"
            else:
                # Decode text content
                try:
                    return content.decode('utf-8')
                except UnicodeDecodeError:
                    return "Unable to decode document content"
        
        if isinstance(content, str):
            if content_type.lower() == "markdown":
                # Convert markdown to HTML for better styling
                try:
                    import markdown
                    return markdown.markdown(content)
                except ImportError:
                    # Fallback: use Streamlit's markdown rendering
                    return content
            else:
                return content
        
        return str(content)
    
    @staticmethod
    def _render_standard_preview(
        content: Union[str, List[PageContent]],
        content_type: str,
        config: RenderConfig,
        container_key: str
    ) -> Dict[str, Any]:
        """Render standard preview mode"""
        interactions = {}
        
        # Create viewer configuration
        viewer_config = DocumentViewerConfig(
            show_zoom_controls=config.show_controls,
            show_page_navigation=config.show_controls,
            show_rulers=config.show_rulers,
            show_margins=config.show_margins,
            initial_zoom=config.zoom_level,
            view_mode=config.view_mode,
            enable_edit_mode=(config.render_mode == RenderMode.EDIT),
            page_size=config.page_size.value["name"].lower()
        )
        
        # Handle multi-page content
        if isinstance(content, list) and content_type == "pages":
            # Multi-page content
            page_config = PageDisplayConfig(
                display_mode=config.page_display_mode,
                page_size=config.page_size,
                show_page_numbers=True,
                show_page_breaks=True
            )
            
            # Use page display component
            page_interactions = PageDisplay.render_pages(
                content, page_config, 1, config.zoom_level, f"{container_key}_pages"
            )
            interactions.update(page_interactions)
        else:
            # Single content rendering
            viewer_interactions = DocumentViewer.render_document_viewer(
                content, 
                content_type if content_type != "pages" else "html",
                viewer_config,
                container_key=f"{container_key}_viewer"
            )
            interactions.update(viewer_interactions)
        
        return interactions
    
    @staticmethod
    def _render_print_preview(
        content: Union[str, List[PageContent]],
        config: RenderConfig,
        container_key: str
    ) -> Dict[str, Any]:
        """Render print preview mode"""
        interactions = {}
        
        st.markdown("### 🖨️ Print Preview")
        
        # Print options
        print_col1, print_col2, print_col3 = st.columns(3)
        
        with print_col1:
            paper_size = st.selectbox(
                "Paper Size",
                options=["A4", "Letter", "Legal"],
                key=f"{container_key}_paper_size"
            )
        
        with print_col2:
            orientation = st.selectbox(
                "Orientation",
                options=["Portrait", "Landscape"],
                key=f"{container_key}_orientation"
            )
        
        with print_col3:
            if st.button("🖨️ Print", key=f"{container_key}_print_btn"):
                interactions["print_requested"] = True
                st.success("Print dialog would open here")
        
        # Render content in print mode
        st.markdown('<div class="print-preview-mode">', unsafe_allow_html=True)
        
        if isinstance(content, list):
            for page in content:
                PreviewRenderer._render_print_page(page, config)
        else:
            # Single page print preview
            print_page = PageContent(1, content, "html", config.page_size)
            PreviewRenderer._render_print_page(print_page, config)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return interactions
    
    @staticmethod
    def _render_print_page(page: PageContent, config: RenderConfig):
        """Render a single page in print mode"""
        page_size_class = f"{config.page_size.value['name'].lower()}-size"
        
        st.markdown(
            f'<div class="document-page {page_size_class} print-page">',
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="document-content">', unsafe_allow_html=True)
        
        if page.content_type.lower() == "html":
            st.markdown(page.content, unsafe_allow_html=True)
        else:
            st.markdown(page.content)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def _render_comparison_view(
        content: Union[str, List[PageContent]],
        config: RenderConfig,
        container_key: str
    ) -> Dict[str, Any]:
        """Render document comparison view"""
        interactions = {}
        
        st.markdown("### 📊 Document Comparison")
        
        # Comparison would require two documents
        # For now, show placeholder
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Document**")
            # Render original content
            if isinstance(content, str):
                st.markdown(content, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Modified Document**")
            st.info("📝 Comparison feature would show changes here")
        
        return interactions
    
    @staticmethod
    def render_quick_preview(
        content: str,
        content_type: str = "html",
        zoom_level: int = 100,
        title: str = "Quick Preview"
    ):
        """
        Render a quick, simple document preview without full controls.
        
        Args:
            content: Document content to preview
            content_type: Type of content
            zoom_level: Zoom level percentage
            title: Preview title
        """
        DocumentViewer.render_simple_preview(content, title, zoom_level)
    
    @staticmethod
    def render_with_style_mapping(
        content: str,
        document_styles: Dict[str, Any],
        content_type: str = "html",
        container_key: str = "styled_preview"
    ) -> Dict[str, Any]:
        """
        Render document with automatic style mapping from generation system.
        
        Args:
            content: Document content
            document_styles: Style definitions from document generation
            content_type: Type of content
            container_key: Unique container key
            
        Returns:
            Rendering interactions
        """
        # Generate CSS from document styles
        mapper = style_mapper.get_style_mapper()
        custom_css = mapper.generate_style_sheet(document_styles)
        
        # Apply styles
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
        
        # Render with enhanced styles
        config = RenderConfig(custom_styles=document_styles)
        return PreviewRenderer.render_document_preview(
            content, content_type, config, container_key
        )
    
    @staticmethod
    def render_from_generation_output(
        generation_result: Dict[str, Any],
        container_key: str = "gen_preview"
    ) -> Dict[str, Any]:
        """
        Render preview directly from document generation system output.
        
        Args:
            generation_result: Output from document generation
            container_key: Unique container key
            
        Returns:
            Rendering interactions
        """
        # Extract content and styles from generation result
        content = generation_result.get("content", "")
        content_type = generation_result.get("format", "html")
        styles = generation_result.get("styles", {})
        metadata = generation_result.get("metadata", {})
        
        # Create configuration from metadata
        config = RenderConfig()
        if "page_size" in metadata:
            page_size_name = metadata["page_size"].upper()
            for size in PageSize:
                if size.value["name"].upper() == page_size_name:
                    config.page_size = size
                    break
        
        # Render with styles
        return PreviewRenderer.render_with_style_mapping(
            content, styles, content_type, container_key
        )


# Utility functions for common use cases

def render_document_from_file(
    file_path: str,
    config: Optional[RenderConfig] = None
) -> Dict[str, Any]:
    """Render document preview from file path"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Determine content type from extension
        if file_path.endswith('.html'):
            content_type = "html"
            content = content.decode('utf-8')
        elif file_path.endswith('.md'):
            content_type = "markdown"
            content = content.decode('utf-8')
        elif file_path.endswith('.docx'):
            content_type = "docx"
        elif file_path.endswith('.pdf'):
            content_type = "pdf"
        else:
            content_type = "text"
            content = content.decode('utf-8')
        
        return PreviewRenderer.render_document_preview(content, content_type, config)
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return {}


def render_document_with_template(
    content: str,
    template_name: str,
    content_type: str = "html"
) -> Dict[str, Any]:
    """Render document with a specific template"""
    config = RenderConfig(style_template=template_name)
    return PreviewRenderer.render_document_preview(content, content_type, config)


def create_preview_from_pages(
    pages: List[str],
    page_size: PageSize = PageSize.A4,
    content_type: str = "html"
) -> Dict[str, Any]:
    """Create preview from list of page contents"""
    page_objects = [
        PageContent(i + 1, content, content_type, page_size)
        for i, content in enumerate(pages)
    ]
    
    config = RenderConfig(
        page_display_mode=PageDisplayMode.CONTINUOUS,
        page_size=page_size
    )
    
    return PreviewRenderer.render_document_preview(page_objects, "pages", config)