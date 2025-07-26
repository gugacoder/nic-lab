"""
Streamlit Document Preview Component Example

This example demonstrates how to implement document preview with zoom controls,
page display, and styling that matches DOCX/PDF output formats.
"""

import streamlit as st
import base64
from typing import Optional, List, Dict, Any
from pathlib import Path

class DocumentPreviewComponent:
    """Enhanced document preview component with zoom and styling controls"""
    
    def __init__(self):
        """Initialize the preview component"""
        self.zoom_levels = [50, 75, 100, 125, 150, 200]
        self.default_zoom = 100
        
        # Initialize session state
        if 'preview_zoom' not in st.session_state:
            st.session_state.preview_zoom = self.default_zoom
        if 'preview_mode' not in st.session_state:
            st.session_state.preview_mode = 'document'  # 'document' or 'print'
    
    def load_preview_css(self) -> str:
        """Load CSS styles for document preview"""
        css = """
        <style>
        .document-preview-container {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: auto;
            max-height: 800px;
            position: relative;
        }
        
        .document-page {
            background: white;
            margin: 20px auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 96px; /* ~1 inch margins */
            width: 816px; /* ~8.5 inches at 96 DPI */
            min-height: 1056px; /* ~11 inches at 96 DPI */
            box-sizing: border-box;
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.5;
            color: #000;
        }
        
        .zoom-controls {
            position: sticky;
            top: 10px;
            right: 10px;
            z-index: 100;
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 8px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .page-break {
            border-bottom: 2px dashed #6c757d;
            margin: 20px 0;
            position: relative;
        }
        
        .page-break::after {
            content: "Page Break";
            position: absolute;
            right: 0;
            top: -10px;
            background: #f8f9fa;
            padding: 2px 8px;
            font-size: 12px;
            color: #6c757d;
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }
        
        .document-header, .document-footer {
            font-size: 10pt;
            color: #666;
            text-align: center;
            margin: 0;
        }
        
        .document-header {
            border-bottom: 1px solid #ddd;
            padding-bottom: 6pt;
            margin-bottom: 24pt;
        }
        
        .document-footer {
            border-top: 1px solid #ddd;
            padding-top: 6pt;
            margin-top: 24pt;
        }
        
        /* Apply zoom transformation */
        .zoom-50 .document-page { transform: scale(0.5); transform-origin: top center; }
        .zoom-75 .document-page { transform: scale(0.75); transform-origin: top center; }
        .zoom-100 .document-page { transform: scale(1.0); transform-origin: top center; }
        .zoom-125 .document-page { transform: scale(1.25); transform-origin: top center; }
        .zoom-150 .document-page { transform: scale(1.5); transform-origin: top center; }
        .zoom-200 .document-page { transform: scale(2.0); transform-origin: top center; }
        
        .document-page {
            transition: transform 0.3s ease;
        }
        </style>
        """
        return css
    
    def render_zoom_controls(self) -> None:
        """Render zoom control interface"""
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        
        with col1:
            if st.button("🔍-", help="Zoom out"):
                current_idx = self.zoom_levels.index(st.session_state.preview_zoom)
                if current_idx > 0:
                    st.session_state.preview_zoom = self.zoom_levels[current_idx - 1]
                    st.rerun()
        
        with col2:
            zoom_percentage = st.selectbox(
                "Zoom",
                self.zoom_levels,
                index=self.zoom_levels.index(st.session_state.preview_zoom),
                format_func=lambda x: f"{x}%",
                key="zoom_select"
            )
            if zoom_percentage != st.session_state.preview_zoom:
                st.session_state.preview_zoom = zoom_percentage
                st.rerun()
        
        with col3:
            if st.button("🔍+", help="Zoom in"):
                current_idx = self.zoom_levels.index(st.session_state.preview_zoom)
                if current_idx < len(self.zoom_levels) - 1:
                    st.session_state.preview_zoom = self.zoom_levels[current_idx + 1]
                    st.rerun()
        
        with col4:
            if st.button("📄", help="Fit to page"):
                st.session_state.preview_zoom = 100
                st.rerun()
        
        with col5:
            preview_mode = st.selectbox(
                "Mode",
                ["document", "print"],
                index=0 if st.session_state.preview_mode == "document" else 1,
                key="preview_mode_select"
            )
            st.session_state.preview_mode = preview_mode
    
    def render_document_content(self, content: str, pages: Optional[List[str]] = None) -> str:
        """Render document content with proper styling"""
        if pages:
            # Multi-page document
            html_content = ""
            for i, page_content in enumerate(pages):
                html_content += f'''
                <div class="document-page">
                    <div class="document-header">Page {i + 1}</div>
                    <div class="document-content">
                        {self._format_content(page_content)}
                    </div>
                    <div class="document-footer">© 2025 NIC Chat System</div>
                </div>
                '''
                if i < len(pages) - 1:
                    html_content += '<div class="page-break"></div>'
        else:
            # Single page document
            html_content = f'''
            <div class="document-page">
                <div class="document-header">Document Preview</div>
                <div class="document-content">
                    {self._format_content(content)}
                </div>
                <div class="document-footer">© 2025 NIC Chat System</div>
            </div>
            '''
        
        return html_content
    
    def _format_content(self, content: str) -> str:
        """Format content with proper HTML tags for preview"""
        # Convert markdown-like formatting to HTML
        import re
        
        # Headers
        content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
        content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
        content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
        
        # Bold and italic
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
        
        # Lists
        content = re.sub(r'^- (.+)$', r'<li>\1</li>', content, flags=re.MULTILINE)
        content = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', content, flags=re.DOTALL)
        
        # Paragraphs
        paragraphs = content.split('\n\n')
        formatted_paragraphs = []
        for para in paragraphs:
            if para.strip() and not para.startswith('<'):
                formatted_paragraphs.append(f'<p>{para.strip()}</p>')
            else:
                formatted_paragraphs.append(para)
        
        return '\n'.join(formatted_paragraphs)
    
    def render_preview(self, content: str, pages: Optional[List[str]] = None) -> None:
        """Render the complete document preview with controls"""
        # Load CSS
        st.markdown(self.load_preview_css(), unsafe_allow_html=True)
        
        # Render zoom controls
        with st.container():
            st.markdown("### Document Preview")
            self.render_zoom_controls()
        
        # Render document with zoom class
        zoom_class = f"zoom-{st.session_state.preview_zoom}"
        mode_class = f"preview-{st.session_state.preview_mode}"
        
        document_html = self.render_document_content(content, pages)
        
        preview_html = f'''
        <div class="document-preview-container {zoom_class} {mode_class}">
            {document_html}
        </div>
        '''
        
        st.markdown(preview_html, unsafe_allow_html=True)
    
    def render_with_comparison(self, content: str, docx_path: Optional[str] = None, pdf_path: Optional[str] = None):
        """Render preview with comparison to actual DOCX/PDF output"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Live Preview")
            self.render_preview(content)
        
        with col2:
            st.markdown("#### Actual Output")
            if docx_path and Path(docx_path).exists():
                st.markdown("**DOCX File:**")
                with open(docx_path, "rb") as file:
                    st.download_button(
                        label="📄 Download DOCX",
                        data=file.read(),
                        file_name=Path(docx_path).name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            
            if pdf_path and Path(pdf_path).exists():
                st.markdown("**PDF File:**")
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="📄 Download PDF",
                        data=file.read(),
                        file_name=Path(pdf_path).name,
                        mime="application/pdf"
                    )


def demo_document_preview():
    """Demo function showing how to use the document preview component"""
    st.title("Document Preview Demo")
    
    # Sample content
    sample_content = """
    # Project Report
    
    ## Executive Summary
    
    This document provides a comprehensive overview of the project implementation and results.
    
    ## Key Findings
    
    The analysis revealed several important insights:
    
    - **Performance Improvement**: 50% reduction in response time
    - **User Satisfaction**: 95% positive feedback
    - **Cost Efficiency**: 30% reduction in operational costs
    
    ## Detailed Analysis
    
    The implementation followed industry best practices and incorporated modern
    technologies to ensure scalability and maintainability.
    
    ### Technical Architecture
    
    The system utilizes a microservices architecture with the following components:
    
    - API Gateway for request routing
    - Authentication service for security
    - Database service for data persistence
    - Analytics service for monitoring
    
    ## Recommendations
    
    Based on the findings, we recommend:
    
    1. Continue monitoring performance metrics
    2. Implement additional security measures
    3. Plan for horizontal scaling
    4. Establish regular maintenance procedures
    
    ## Conclusion
    
    The project has successfully met all defined objectives and is ready for
    production deployment.
    """
    
    # Create preview component
    preview = DocumentPreviewComponent()
    
    # Allow user to edit content
    edited_content = st.text_area(
        "Document Content (Markdown-like)",
        value=sample_content,
        height=200
    )
    
    # Option to split into pages
    split_pages = st.checkbox("Split into multiple pages")
    
    if split_pages:
        # Simple page splitting by content length
        words = edited_content.split()
        words_per_page = 150
        pages = []
        for i in range(0, len(words), words_per_page):
            page_words = words[i:i + words_per_page]
            pages.append(' '.join(page_words))
        
        st.info(f"Content split into {len(pages)} pages")
        preview.render_preview(edited_content, pages)
    else:
        preview.render_preview(edited_content)
    
    # Show preview statistics
    with st.expander("Preview Statistics"):
        st.write(f"**Zoom Level:** {st.session_state.preview_zoom}%")
        st.write(f"**Preview Mode:** {st.session_state.preview_mode}")
        st.write(f"**Content Length:** {len(edited_content)} characters")
        st.write(f"**Word Count:** {len(edited_content.split())} words")


if __name__ == "__main__":
    # Run the demo
    demo_document_preview()