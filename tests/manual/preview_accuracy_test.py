"""
Manual Test for Document Preview Accuracy

This test script validates that the document preview accurately matches
DOCX and PDF output formats. Run with: streamlit run tests/manual/preview_accuracy_test.py
"""

import streamlit as st
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.components.preview.document_viewer import DocumentViewer, DocumentViewerConfig
from src.components.preview.preview_renderer import PreviewRenderer, RenderConfig, RenderMode
from src.components.preview.page_display import PageContent, PageSize
from src.utils.style_mapper import get_style_mapper, generate_preview_styles_from_template

def main():
    st.set_page_config(
        page_title="Document Preview Accuracy Test",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Preview Accuracy Test")
    st.markdown("Testing preview accuracy against DOCX and PDF output formats")
    
    # Test selection
    test_type = st.selectbox(
        "Select Test Type",
        options=[
            "Basic Typography Test",
            "Heading Styles Test",
            "Table Rendering Test",
            "List Formatting Test",
            "Page Layout Test",
            "Style Mapping Test",
            "Multi-Page Test",
            "Zoom Functionality Test"
        ]
    )
    
    if test_type == "Basic Typography Test":
        run_typography_test()
    elif test_type == "Heading Styles Test":
        run_heading_test()
    elif test_type == "Table Rendering Test":
        run_table_test()
    elif test_type == "List Formatting Test":
        run_list_test()
    elif test_type == "Page Layout Test":
        run_page_layout_test()
    elif test_type == "Style Mapping Test":
        run_style_mapping_test()
    elif test_type == "Multi-Page Test":
        run_multi_page_test()
    elif test_type == "Zoom Functionality Test":
        run_zoom_test()

def run_typography_test():
    st.subheader("📝 Basic Typography Test")
    st.markdown("Testing font families, sizes, and basic formatting")
    
    # Sample content with various typography
    content = """
    <div class="document-content">
        <p class="doc-paragraph">This is a standard paragraph with normal text formatting. 
        It should match the default Times New Roman font at 12pt size with 1.15 line height.</p>
        
        <p class="doc-paragraph">This paragraph contains <span class="doc-bold">bold text</span>, 
        <span class="doc-italic">italic text</span>, and 
        <span class="doc-underline">underlined text</span>.</p>
        
        <p class="doc-paragraph doc-size-14">This paragraph uses 14pt font size instead of the default 12pt.</p>
        
        <p class="doc-paragraph doc-size-10">This paragraph uses 10pt font size for comparison.</p>
        
        <p class="doc-paragraph" style="font-family: Arial, sans-serif;">
        This paragraph uses Arial font family instead of Times New Roman.</p>
    </div>
    """
    
    config = RenderConfig(zoom_level=100, show_rulers=True, show_margins=True)
    interactions = PreviewRenderer.render_document_preview(content, "html", config, "typography_test")
    
    # Results
    st.markdown("### ✅ Validation Checklist")
    st.checkbox("Font appears as Times New Roman (or similar serif font)", key="font_check")
    st.checkbox("Font size appears as 12pt for normal text", key="size_check")
    st.checkbox("Line height appears consistent at 1.15", key="line_height_check")
    st.checkbox("Bold text is visibly bolder", key="bold_check")
    st.checkbox("Italic text is properly slanted", key="italic_check")
    st.checkbox("Underlined text has proper underlines", key="underline_check")

def run_heading_test():
    st.subheader("📋 Heading Styles Test")
    st.markdown("Testing heading hierarchy and formatting")
    
    content = """
    <div class="document-content">
        <h1 class="doc-heading-1">Heading 1 - 16pt Bold Blue Calibri</h1>
        <p class="doc-paragraph">This is a paragraph following Heading 1.</p>
        
        <h2 class="doc-heading-2">Heading 2 - 13pt Bold Blue Calibri</h2>
        <p class="doc-paragraph">This is a paragraph following Heading 2.</p>
        
        <h3 class="doc-heading-3">Heading 3 - 12pt Bold Dark Blue Calibri</h3>
        <p class="doc-paragraph">This is a paragraph following Heading 3.</p>
        
        <h4 class="doc-heading-4">Heading 4 - 11pt Bold Italic Blue Calibri</h4>
        <p class="doc-paragraph">This is a paragraph following Heading 4.</p>
    </div>
    """
    
    config = RenderConfig(zoom_level=125, show_rulers=True)
    interactions = PreviewRenderer.render_document_preview(content, "html", config, "heading_test")
    
    st.markdown("### ✅ Validation Checklist")
    st.checkbox("Heading 1 is largest and uses blue color", key="h1_check")
    st.checkbox("Heading 2 is smaller than H1 but larger than body text", key="h2_check")
    st.checkbox("Heading 3 uses darker blue color", key="h3_check")
    st.checkbox("Heading 4 appears italic", key="h4_check")
    st.checkbox("All headings use Calibri or similar sans-serif font", key="heading_font_check")

def run_table_test():
    st.subheader("📊 Table Rendering Test")
    st.markdown("Testing table formatting and borders")
    
    content = """
    <div class="document-content">
        <h2 class="doc-heading-2">Sample Table</h2>
        
        <table class="doc-table">
            <thead>
                <tr>
                    <th>Header 1</th>
                    <th>Header 2</th>
                    <th>Header 3</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Row 1, Cell 1</td>
                    <td>Row 1, Cell 2</td>
                    <td>Row 1, Cell 3</td>
                </tr>
                <tr>
                    <td>Row 2, Cell 1 with longer content</td>
                    <td>Row 2, Cell 2</td>
                    <td>Row 2, Cell 3</td>
                </tr>
                <tr>
                    <td>Row 3, Cell 1</td>
                    <td>Row 3, Cell 2</td>
                    <td>Row 3, Cell 3</td>
                </tr>
            </tbody>
        </table>
        
        <p class="doc-paragraph">This is a paragraph after the table.</p>
    </div>
    """
    
    config = RenderConfig(zoom_level=100)
    interactions = PreviewRenderer.render_document_preview(content, "html", config, "table_test")
    
    st.markdown("### ✅ Validation Checklist")
    st.checkbox("Table has visible borders around all cells", key="table_borders_check")
    st.checkbox("Header row has gray background", key="table_header_check")
    st.checkbox("Cell padding appears appropriate (4pt 8pt)", key="table_padding_check")
    st.checkbox("Text aligns to top of cells", key="table_valign_check")
    st.checkbox("Table width uses full available space", key="table_width_check")

def run_list_test():
    st.subheader("📝 List Formatting Test")
    st.markdown("Testing bullet and numbered lists")
    
    content = """
    <div class="document-content">
        <h2 class="doc-heading-2">List Examples</h2>
        
        <h3 class="doc-heading-3">Bullet List</h3>
        <ul class="doc-list-bullet">
            <li>First bullet item</li>
            <li>Second bullet item with longer text that might wrap to multiple lines</li>
            <li>Third bullet item
                <ul class="doc-list-bullet">
                    <li>Nested bullet item</li>
                    <li>Another nested item</li>
                </ul>
            </li>
            <li>Fourth bullet item</li>
        </ul>
        
        <h3 class="doc-heading-3">Numbered List</h3>
        <ol class="doc-list-numbered">
            <li>First numbered item</li>
            <li>Second numbered item</li>
            <li>Third numbered item with sub-items
                <ol class="doc-list-numbered">
                    <li>Nested numbered item</li>
                    <li>Another nested numbered item</li>
                </ol>
            </li>
            <li>Fourth numbered item</li>
        </ol>
    </div>
    """
    
    config = RenderConfig(zoom_level=100)
    interactions = PreviewRenderer.render_document_preview(content, "html", config, "list_test")
    
    st.markdown("### ✅ Validation Checklist")
    st.checkbox("Bullet points appear as solid discs", key="bullet_style_check")
    st.checkbox("Nested bullets use different style (circles)", key="nested_bullet_check")
    st.checkbox("Numbers appear correctly in sequence", key="number_sequence_check")
    st.checkbox("List items have proper indentation (0.5 inch)", key="list_indent_check")
    st.checkbox("Line spacing within lists is appropriate", key="list_spacing_check")

def run_page_layout_test():
    st.subheader("📐 Page Layout Test")
    st.markdown("Testing page dimensions and margins")
    
    content = """
    <div class="document-content">
        <h1 class="doc-heading-1">Page Layout Test Document</h1>
        
        <p class="doc-paragraph">This document tests the page layout dimensions and margins. 
        The page should be A4 size (794px × 1123px at 96 DPI) with 1-inch margins on all sides.</p>
        
        <p class="doc-paragraph">The content area should be properly contained within the margins, 
        and the page should have a realistic appearance with shadow effects.</p>
        
        <h2 class="doc-heading-2">Margin Indicators</h2>
        <p class="doc-paragraph">When margin indicators are enabled, you should see dashed lines 
        marking the 1-inch margins on all four sides of the page.</p>
        
        <h2 class="doc-heading-2">Rulers</h2>
        <p class="doc-paragraph">When rulers are enabled, you should see horizontal and vertical 
        rulers that help with precise positioning and measurement.</p>
        
        <p class="doc-paragraph">This content fills the page to demonstrate the layout bounds 
        and ensure that text flows properly within the designated content area.</p>
    </div>
    """
    
    # Test different page sizes
    page_size_option = st.selectbox(
        "Page Size",
        options=["A4", "Letter", "Legal"],
        key="page_size_test"
    )
    
    page_sizes = {
        "A4": PageSize.A4,
        "Letter": PageSize.LETTER,
        "Legal": PageSize.LEGAL
    }
    
    config = RenderConfig(
        page_size=page_sizes[page_size_option],
        show_rulers=True,
        show_margins=True,
        zoom_level=100
    )
    
    interactions = PreviewRenderer.render_document_preview(content, "html", config, "layout_test")
    
    st.markdown("### ✅ Validation Checklist")
    st.checkbox("Page has correct dimensions for selected size", key="page_size_check")
    st.checkbox("Margins appear as 1-inch on all sides", key="margin_check")
    st.checkbox("Content is properly contained within margins", key="content_bounds_check")
    st.checkbox("Page shadow effect is visible", key="page_shadow_check")
    st.checkbox("Rulers and margin indicators display correctly", key="rulers_check")

def run_style_mapping_test():
    st.subheader("🎨 Style Mapping Test")
    st.markdown("Testing automatic style mapping from generation system")
    
    # Simulate document styles from generation system
    document_styles = {
        "corporate-heading": {
            "text_style": {
                "font_family": "Arial",
                "font_size": 18,
                "weight": "bold",
                "color": "#1f4788"
            },
            "content_type": "heading_1"
        },
        "corporate-body": {
            "text_style": {
                "font_family": "Arial",
                "font_size": 11,
                "weight": "normal",
                "color": "#000000"
            },
            "content_type": "paragraph"
        }
    }
    
    content = """
    <div class="document-content">
        <h1 class="doc-corporate-heading">Corporate Style Heading</h1>
        <p class="doc-corporate-body">This paragraph uses corporate body styling with Arial font at 11pt.</p>
        <p class="doc-corporate-body">The styles are automatically mapped from the document generation system to ensure consistency between preview and final output.</p>
    </div>
    """
    
    interactions = PreviewRenderer.render_with_style_mapping(
        content, document_styles, "html", "style_mapping_test"
    )
    
    st.markdown("### ✅ Validation Checklist")
    st.checkbox("Heading uses Arial font family", key="style_font_check")
    st.checkbox("Heading color matches specified blue", key="style_color_check")
    st.checkbox("Body text uses correct font size", key="style_size_check")
    st.checkbox("Styles are applied consistently", key="style_consistency_check")

def run_multi_page_test():
    st.subheader("📄 Multi-Page Test")
    st.markdown("Testing multi-page document rendering")
    
    # Create sample pages
    pages = [
        PageContent(1, """
        <h1 class="doc-heading-1">Document Title - Page 1</h1>
        <p class="doc-paragraph">This is the first page of a multi-page document. 
        It contains the document title and introduction.</p>
        <p class="doc-paragraph">The page should display with proper page numbering 
        and maintain consistent formatting across all pages.</p>
        """, "html", PageSize.A4),
        
        PageContent(2, """
        <h2 class="doc-heading-2">Chapter 1 - Page 2</h2>
        <p class="doc-paragraph">This is the second page with chapter content. 
        Page breaks should be clearly visible between pages.</p>
        <p class="doc-paragraph">Navigation controls should allow moving between 
        pages easily.</p>
        """, "html", PageSize.A4),
        
        PageContent(3, """
        <h2 class="doc-heading-2">Chapter 2 - Page 3</h2>
        <p class="doc-paragraph">This is the third and final page of the document. 
        It demonstrates the multi-page rendering capability.</p>
        <p class="doc-paragraph">The page numbering should be accurate and 
        the layout consistent.</p>
        """, "html", PageSize.A4)
    ]
    
    config = RenderConfig(
        page_display_mode=PageDisplayMode.CONTINUOUS,
        show_controls=True,
        zoom_level=100
    )
    
    from src.components.preview.page_display import PageDisplayMode
    interactions = PreviewRenderer.render_document_preview(pages, "pages", config, "multi_page_test")
    
    st.markdown("### ✅ Validation Checklist")
    st.checkbox("All three pages display correctly", key="page_count_check")
    st.checkbox("Page breaks are visible between pages", key="page_breaks_check")
    st.checkbox("Page numbering is accurate", key="page_numbers_check")
    st.checkbox("Navigation controls work properly", key="navigation_check")
    st.checkbox("Formatting is consistent across pages", key="multi_page_consistency_check")

def run_zoom_test():
    st.subheader("🔍 Zoom Functionality Test")
    st.markdown("Testing zoom controls and scaling")
    
    content = """
    <div class="document-content">
        <h1 class="doc-heading-1">Zoom Test Document</h1>
        <p class="doc-paragraph">This document tests the zoom functionality. 
        Use the zoom controls to test different zoom levels.</p>
        
        <h2 class="doc-heading-2">Zoom Levels to Test</h2>
        <ul class="doc-list-bullet">
            <li>50% - Should be readable but small</li>
            <li>75% - Reduced size but clear</li>
            <li>100% - Standard size</li>
            <li>125% - Slightly enlarged</li>
            <li>150% - Noticeably larger</li>
            <li>200% - Double size</li>
        </ul>
        
        <p class="doc-paragraph">Text should remain crisp and readable at all zoom levels. 
        The layout should scale proportionally without breaking.</p>
    </div>
    """
    
    # Test different zoom levels
    zoom_level = st.slider("Test Zoom Level", 25, 400, 100, 25, key="zoom_test_slider")
    
    config = RenderConfig(zoom_level=zoom_level, show_controls=True)
    interactions = PreviewRenderer.render_document_preview(content, "html", config, "zoom_test")
    
    st.markdown("### ✅ Validation Checklist")
    st.checkbox("Zoom controls respond smoothly", key="zoom_controls_check")
    st.checkbox("Text remains crisp at all zoom levels", key="zoom_quality_check")
    st.checkbox("Layout scales proportionally", key="zoom_layout_check")
    st.checkbox("Page boundaries scale correctly", key="zoom_bounds_check")
    st.checkbox("Performance remains smooth during zoom", key="zoom_performance_check")

if __name__ == "__main__":
    main()