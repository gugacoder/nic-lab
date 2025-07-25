"""
Default DOCX Template Configuration

This module defines the default template for DOCX document generation
with corporate styling and formatting standards.
"""

from typing import List

from src.generation.models import (
    DocumentTemplate, TemplateStyle, TemplateLayout, TextStyle, 
    StyleWeight, StyleEmphasis, ContentType
)


def create_default_docx_template() -> DocumentTemplate:
    """
    Create the default DOCX template with corporate styling.
    
    Returns:
        DocumentTemplate configured for professional documents
    """
    
    # Define template styles
    styles = [
        # Heading styles
        TemplateStyle(
            name="Corporate Heading 1",
            style=TextStyle(
                font_family="Arial",
                font_size=18,
                weight=StyleWeight.BOLD,
                color="#1F4E79"
            ),
            applies_to=[ContentType.HEADING]
        ),
        TemplateStyle(
            name="Corporate Heading 2", 
            style=TextStyle(
                font_family="Arial",
                font_size=16,
                weight=StyleWeight.BOLD,
                color="#2E75B5"
            ),
            applies_to=[ContentType.HEADING]
        ),
        TemplateStyle(
            name="Corporate Heading 3",
            style=TextStyle(
                font_family="Arial", 
                font_size=14,
                weight=StyleWeight.BOLD,
                color="#5B9BD5"
            ),
            applies_to=[ContentType.HEADING]
        ),
        
        # Body text styles
        TemplateStyle(
            name="Corporate Body",
            style=TextStyle(
                font_family="Calibri",
                font_size=11,
                weight=StyleWeight.NORMAL,
                line_height=1.15
            ),
            applies_to=[ContentType.PARAGRAPH, ContentType.TEXT]
        ),
        
        # List styles
        TemplateStyle(
            name="Corporate List",
            style=TextStyle(
                font_family="Calibri",
                font_size=11,
                weight=StyleWeight.NORMAL
            ),
            applies_to=[ContentType.LIST]
        ),
        
        # Quote style
        TemplateStyle(
            name="Corporate Quote",
            style=TextStyle(
                font_family="Calibri",
                font_size=11,
                emphasis=StyleEmphasis.ITALIC,
                color="#666666"
            ),
            applies_to=[ContentType.QUOTE]
        ),
        
        # Code style
        TemplateStyle(
            name="Corporate Code",
            style=TextStyle(
                font_family="Consolas",
                font_size=9,
                background_color="#F5F5F5"
            ),
            applies_to=[ContentType.CODE]
        )
    ]
    
    # Define layout
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
        footer_height=1.27
    )
    
    # Create template
    template = DocumentTemplate(
        name="Corporate Default",
        format_type="docx",
        description="Default corporate template for professional documents",
        styles=styles,
        layout=layout,
        brand_colors={
            "primary": "#1F4E79",
            "secondary": "#2E75B5", 
            "accent": "#5B9BD5",
            "text": "#000000",
            "light_gray": "#F5F5F5"
        },
        company_name="Your Company Name",
        header_template="{{company_name}} - {{title}}",
        footer_template="Page {{page_number}} of {{total_pages}} | {{date}}",
        is_default=True
    )
    
    return template


def create_minimal_docx_template() -> DocumentTemplate:
    """
    Create a minimal DOCX template with basic styling.
    
    Returns:
        DocumentTemplate with minimal formatting
    """
    
    # Minimal styles
    styles = [
        TemplateStyle(
            name="Simple Heading",
            style=TextStyle(
                font_family="Arial",
                font_size=14,
                weight=StyleWeight.BOLD
            ),
            applies_to=[ContentType.HEADING]
        ),
        TemplateStyle(
            name="Simple Text",
            style=TextStyle(
                font_family="Calibri",
                font_size=11
            ),
            applies_to=[ContentType.PARAGRAPH, ContentType.TEXT]
        )
    ]
    
    # Minimal layout
    layout = TemplateLayout(
        page_size="Letter",
        page_orientation="portrait",
        margins={
            "top": 2.54,
            "bottom": 2.54,
            "left": 2.54, 
            "right": 2.54
        }
    )
    
    template = DocumentTemplate(
        name="Minimal",
        format_type="docx",
        description="Minimal template for simple documents",
        styles=styles,
        layout=layout,
        brand_colors={
            "primary": "#000000",
            "text": "#000000"
        }
    )
    
    return template


# Template registry
DOCX_TEMPLATES = {
    "default": create_default_docx_template,
    "corporate": create_default_docx_template,  # Alias
    "minimal": create_minimal_docx_template
}


def get_docx_template(template_name: str = "default") -> DocumentTemplate:
    """
    Get a DOCX template by name.
    
    Args:
        template_name: Name of template to retrieve
        
    Returns:
        DocumentTemplate instance
        
    Raises:
        ValueError: If template name not found
    """
    if template_name not in DOCX_TEMPLATES:
        available = ", ".join(DOCX_TEMPLATES.keys())
        raise ValueError(f"Template '{template_name}' not found. Available: {available}")
    
    return DOCX_TEMPLATES[template_name]()


def list_docx_templates() -> List[str]:
    """
    List available DOCX template names.
    
    Returns:
        List of template names
    """
    return list(DOCX_TEMPLATES.keys())


# Test template creation
if __name__ == "__main__":
    # Test default template
    default_template = create_default_docx_template()
    print(f"✅ Created default template: {default_template.name}")
    print(f"   - Styles: {len(default_template.styles)}")
    print(f"   - Brand colors: {len(default_template.brand_colors)}")
    
    # Test minimal template
    minimal_template = create_minimal_docx_template()
    print(f"✅ Created minimal template: {minimal_template.name}")
    
    # Test template registry
    templates = list_docx_templates()
    print(f"✅ Available templates: {templates}")
    
    # Test retrieval
    retrieved = get_docx_template("default")
    print(f"✅ Retrieved template: {retrieved.name}")