"""
Default PDF Template

This module provides the default template configuration for PDF generation,
including standard styles, layouts, and formatting options.
"""

from src.generation.models import (
    DocumentTemplate, 
    TemplateStyle, 
    TemplateLayout,
    TextStyle,
    StyleWeight,
    StyleEmphasis,
    ContentType
)
from datetime import datetime


def create_default_pdf_template() -> DocumentTemplate:
    """
    Create the default PDF template with standard corporate styling.
    
    Returns:
        DocumentTemplate configured for PDF generation
    """
    
    # Define layout configuration
    layout = TemplateLayout(
        page_size="A4",
        page_orientation="portrait",
        margins={
            "top": 2.54,
            "bottom": 2.54,
            "left": 2.54,
            "right": 2.54
        },
        header_height=1.27,
        footer_height=1.27,
        column_count=1,
        column_spacing=1.27
    )
    
    # Define template styles
    styles = [
        # Document title style
        TemplateStyle(
            name="document_title",
            style=TextStyle(
                font_family="Helvetica",
                font_size=24,
                weight=StyleWeight.BOLD,
                color="#000000"
            ),
            applies_to=[ContentType.HEADING],
            conditions={"level": 0}
        ),
        
        # Heading 1 style
        TemplateStyle(
            name="heading_1",
            style=TextStyle(
                font_family="Helvetica",
                font_size=20,
                weight=StyleWeight.BOLD,
                color="#000000",
                line_height=1.2
            ),
            applies_to=[ContentType.HEADING],
            conditions={"level": 1}
        ),
        
        # Heading 2 style
        TemplateStyle(
            name="heading_2",
            style=TextStyle(
                font_family="Helvetica",
                font_size=16,
                weight=StyleWeight.BOLD,
                color="#333333",
                line_height=1.2
            ),
            applies_to=[ContentType.HEADING],
            conditions={"level": 2}
        ),
        
        # Heading 3 style
        TemplateStyle(
            name="heading_3",
            style=TextStyle(
                font_family="Helvetica",
                font_size=14,
                weight=StyleWeight.BOLD,
                color="#333333",
                line_height=1.2
            ),
            applies_to=[ContentType.HEADING],
            conditions={"level": 3}
        ),
        
        # Normal paragraph style
        TemplateStyle(
            name="paragraph",
            style=TextStyle(
                font_family="Helvetica",
                font_size=11,
                weight=StyleWeight.NORMAL,
                color="#000000",
                line_height=1.15
            ),
            applies_to=[ContentType.PARAGRAPH, ContentType.TEXT]
        ),
        
        # List item style
        TemplateStyle(
            name="list_item",
            style=TextStyle(
                font_family="Helvetica",
                font_size=11,
                weight=StyleWeight.NORMAL,
                color="#000000",
                line_height=1.15
            ),
            applies_to=[ContentType.LIST]
        ),
        
        # Code block style
        TemplateStyle(
            name="code_block",
            style=TextStyle(
                font_family="Courier",
                font_size=9,
                weight=StyleWeight.NORMAL,
                color="#000000",
                background_color="#f5f5f5",
                line_height=1.2
            ),
            applies_to=[ContentType.CODE]
        ),
        
        # Quote style
        TemplateStyle(
            name="quote",
            style=TextStyle(
                font_family="Helvetica",
                font_size=11,
                weight=StyleWeight.NORMAL,
                emphasis=StyleEmphasis.ITALIC,
                color="#555555",
                line_height=1.15
            ),
            applies_to=[ContentType.QUOTE]
        ),
        
        # Table header style
        TemplateStyle(
            name="table_header",
            style=TextStyle(
                font_family="Helvetica",
                font_size=10,
                weight=StyleWeight.BOLD,
                color="#ffffff",
                background_color="#666666"
            ),
            applies_to=[ContentType.TABLE],
            conditions={"is_header": True}
        ),
        
        # Table cell style
        TemplateStyle(
            name="table_cell",
            style=TextStyle(
                font_family="Helvetica",
                font_size=9,
                weight=StyleWeight.NORMAL,
                color="#000000"
            ),
            applies_to=[ContentType.TABLE],
            conditions={"is_header": False}
        )
    ]
    
    # Create the template
    template = DocumentTemplate(
        name="Default PDF Template",
        format_type="pdf",
        description="Standard PDF template with clean, professional styling",
        styles=styles,
        layout=layout,
        
        # Branding
        company_name="Document Generator",
        brand_colors={
            "primary": "#000000",
            "secondary": "#666666",
            "accent": "#333333",
            "background": "#ffffff"
        },
        
        # Header/Footer templates
        header_template="Generated PDF Document",
        footer_template="Page {page_num} | {date}",
        
        # Metadata
        version="1.0",
        created_date=datetime.now(),
        author="PDF Generator System",
        tags=["default", "pdf", "professional"],
        is_default=True
    )
    
    return template


def create_corporate_pdf_template() -> DocumentTemplate:
    """
    Create a corporate PDF template with formal styling.
    
    Returns:
        DocumentTemplate configured for corporate documents
    """
    
    # Define layout with narrower margins for more content
    layout = TemplateLayout(
        page_size="A4",
        page_orientation="portrait",
        margins={
            "top": 2.0,
            "bottom": 2.0,
            "left": 2.0,
            "right": 2.0
        },
        header_height=1.5,
        footer_height=1.0,
        column_count=1
    )
    
    # Corporate styling with professional colors
    styles = [
        TemplateStyle(
            name="corporate_title",
            style=TextStyle(
                font_family="Times",
                font_size=26,
                weight=StyleWeight.BOLD,
                color="#1f4e79"  # Corporate blue
            ),
            applies_to=[ContentType.HEADING],
            conditions={"level": 0}
        ),
        
        TemplateStyle(
            name="corporate_heading_1",
            style=TextStyle(
                font_family="Times",
                font_size=18,
                weight=StyleWeight.BOLD,
                color="#1f4e79",
                line_height=1.3
            ),
            applies_to=[ContentType.HEADING],
            conditions={"level": 1}
        ),
        
        TemplateStyle(
            name="corporate_paragraph",
            style=TextStyle(
                font_family="Times",
                font_size=12,
                weight=StyleWeight.NORMAL,
                color="#000000",
                line_height=1.25
            ),
            applies_to=[ContentType.PARAGRAPH, ContentType.TEXT]
        )
    ]
    
    template = DocumentTemplate(
        name="Corporate PDF Template",
        format_type="pdf",
        description="Formal corporate template with Times font and professional styling",
        styles=styles,
        layout=layout,
        company_name="Corporate Organization",
        brand_colors={
            "primary": "#1f4e79",
            "secondary": "#4472c4",
            "accent": "#70ad47",
            "background": "#ffffff"
        },
        header_template="CONFIDENTIAL - Corporate Document",
        footer_template="© 2025 Corporate Organization | Page {page_num}",
        version="1.0",
        created_date=datetime.now(),
        author="Corporate Templates",
        tags=["corporate", "formal", "professional"],
        is_default=False
    )
    
    return template


def create_report_pdf_template() -> DocumentTemplate:
    """
    Create a report-style PDF template optimized for technical documents.
    
    Returns:
        DocumentTemplate configured for reports
    """
    
    layout = TemplateLayout(
        page_size="A4",
        page_orientation="portrait",
        margins={
            "top": 3.0,
            "bottom": 2.5,
            "left": 2.5,
            "right": 2.5
        },
        header_height=2.0,
        footer_height=1.5,
        column_count=1
    )
    
    styles = [
        TemplateStyle(
            name="report_title",
            style=TextStyle(
                font_family="Helvetica",
                font_size=22,
                weight=StyleWeight.BOLD,
                color="#2c3e50"
            ),
            applies_to=[ContentType.HEADING],
            conditions={"level": 0}
        ),
        
        TemplateStyle(
            name="report_section",
            style=TextStyle(
                font_family="Helvetica",
                font_size=16,
                weight=StyleWeight.BOLD,
                color="#34495e",
                line_height=1.4
            ),
            applies_to=[ContentType.HEADING],
            conditions={"level": 1}
        ),
        
        TemplateStyle(
            name="report_body",
            style=TextStyle(
                font_family="Helvetica",
                font_size=10,
                weight=StyleWeight.NORMAL,
                color="#2c3e50",
                line_height=1.4
            ),
            applies_to=[ContentType.PARAGRAPH, ContentType.TEXT]
        ),
        
        TemplateStyle(
            name="report_code",
            style=TextStyle(
                font_family="Courier",
                font_size=8,
                weight=StyleWeight.NORMAL,
                color="#e74c3c",
                background_color="#ecf0f1"
            ),
            applies_to=[ContentType.CODE]
        )
    ]
    
    template = DocumentTemplate(
        name="Technical Report Template",
        format_type="pdf",
        description="Template optimized for technical reports and documentation",
        styles=styles,
        layout=layout,
        company_name="Technical Documentation",
        brand_colors={
            "primary": "#2c3e50",
            "secondary": "#34495e",
            "accent": "#3498db",
            "background": "#ffffff"
        },
        header_template="Technical Report | {title}",
        footer_template="Generated: {date} | Page {page_num} of {total_pages}",
        version="1.0",
        created_date=datetime.now(),
        author="Report Generator",
        tags=["report", "technical", "documentation"],
        is_default=False
    )
    
    return template


# Available templates
AVAILABLE_TEMPLATES = {
    "default": create_default_pdf_template,
    "corporate": create_corporate_pdf_template,
    "report": create_report_pdf_template
}


def get_template(template_name: str = "default") -> DocumentTemplate:
    """
    Get a PDF template by name.
    
    Args:
        template_name: Name of the template to retrieve
        
    Returns:
        DocumentTemplate instance
        
    Raises:
        KeyError: If template name is not found
    """
    if template_name not in AVAILABLE_TEMPLATES:
        raise KeyError(f"Template '{template_name}' not found. Available: {list(AVAILABLE_TEMPLATES.keys())}")
    
    return AVAILABLE_TEMPLATES[template_name]()


def list_available_templates() -> list:
    """Get list of available template names."""
    return list(AVAILABLE_TEMPLATES.keys())