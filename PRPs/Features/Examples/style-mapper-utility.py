"""
Style Mapper Utility Example

This example demonstrates how to map document generation styles to CSS
for accurate preview rendering that matches DOCX and PDF output.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class DocumentFormat(Enum):
    """Supported document formats"""
    DOCX = "docx"
    PDF = "pdf"
    HTML = "html"

@dataclass
class FontStyle:
    """Font styling information"""
    family: str = "Times New Roman"
    size: int = 12  # in points
    bold: bool = False
    italic: bool = False
    underline: bool = False
    color: str = "#000000"
    
class DocumentStyleMapper:
    """Maps document generation styles to CSS for preview rendering"""
    
    def __init__(self):
        """Initialize style mapper with format-specific mappings"""
        self.docx_font_mapping = {
            "Times New Roman": "'Times New Roman', Times, serif",
            "Arial": "Arial, sans-serif", 
            "Calibri": "Calibri, sans-serif",
            "Helvetica": "Helvetica, Arial, sans-serif",
            "Georgia": "Georgia, serif",
            "Courier New": "'Courier New', monospace"
        }
        
        self.pdf_font_mapping = {
            "Times-Roman": "'Times New Roman', Times, serif",
            "Helvetica": "Helvetica, Arial, sans-serif",
            "Courier": "'Courier New', monospace"
        }
        
        # Standard document margins (in CSS units)
        self.page_margins = {
            "normal": {"top": "1in", "right": "1in", "bottom": "1in", "left": "1in"},
            "narrow": {"top": "0.5in", "right": "0.5in", "bottom": "0.5in", "left": "0.5in"},
            "wide": {"top": "1in", "right": "2in", "bottom": "1in", "left": "2in"}
        }
    
    def map_font_to_css(self, font_style: FontStyle, target_format: DocumentFormat) -> Dict[str, str]:
        """Convert font style to CSS properties"""
        css_props = {}
        
        # Font family mapping
        if target_format == DocumentFormat.DOCX:
            css_props["font-family"] = self.docx_font_mapping.get(
                font_style.family, 
                f"'{font_style.family}', serif"
            )
        elif target_format == DocumentFormat.PDF:
            css_props["font-family"] = self.pdf_font_mapping.get(
                font_style.family,
                f"'{font_style.family}', serif"  
            )
        else:
            css_props["font-family"] = f"'{font_style.family}', serif"
        
        # Font size (convert points to CSS)
        css_props["font-size"] = f"{font_style.size}pt"
        
        # Font weight
        css_props["font-weight"] = "bold" if font_style.bold else "normal"
        
        # Font style
        css_props["font-style"] = "italic" if font_style.italic else "normal"
        
        # Text decoration
        decorations = []
        if font_style.underline:
            decorations.append("underline")
        css_props["text-decoration"] = " ".join(decorations) if decorations else "none"
        
        # Color
        css_props["color"] = font_style.color
        
        return css_props
    
    def generate_paragraph_css(self, 
                             style_name: str,
                             font_style: FontStyle,
                             line_spacing: float = 1.15,
                             space_before: int = 0,
                             space_after: int = 0,
                             alignment: str = "left",
                             indent: int = 0) -> str:
        """Generate CSS for paragraph styles"""
        
        font_props = self.map_font_to_css(font_style, DocumentFormat.DOCX)
        
        css_rules = []
        css_rules.append(f".{style_name} {{")
        
        # Font properties
        for prop, value in font_props.items():
            css_rules.append(f"    {prop}: {value};")
        
        # Line spacing
        css_rules.append(f"    line-height: {line_spacing};")
        
        # Spacing
        if space_before > 0:
            css_rules.append(f"    margin-top: {space_before}pt;")
        if space_after > 0:
            css_rules.append(f"    margin-bottom: {space_after}pt;")
        
        # Text alignment
        css_rules.append(f"    text-align: {alignment};")
        
        # Indentation
        if indent > 0:
            css_rules.append(f"    text-indent: {indent}pt;")
        
        css_rules.append("}")
        
        return "\n".join(css_rules)
    
    def generate_heading_styles(self) -> str:
        """Generate CSS for standard heading styles"""
        styles = []
        
        # Heading 1
        h1_font = FontStyle(family="Arial", size=16, bold=True, color="#2F5597")
        styles.append(self.generate_paragraph_css(
            "heading-1", h1_font, 
            line_spacing=1.0, space_before=12, space_after=3
        ))
        
        # Heading 2
        h2_font = FontStyle(family="Arial", size=13, bold=True, color="#2F5597")
        styles.append(self.generate_paragraph_css(
            "heading-2", h2_font,
            line_spacing=1.0, space_before=10, space_after=3
        ))
        
        # Heading 3
        h3_font = FontStyle(family="Arial", size=12, bold=True, color="#1F497D")
        styles.append(self.generate_paragraph_css(
            "heading-3", h3_font,
            line_spacing=1.0, space_before=8, space_after=2
        ))
        
        return "\n\n".join(styles)
    
    def generate_table_css(self, 
                          border_width: int = 1,
                          border_color: str = "#000000",
                          cell_padding: int = 6,
                          header_bg: str = "#D9D9D9") -> str:
        """Generate CSS for table styles matching document output"""
        
        css = f"""
.document-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12pt 0;
    font-size: 11pt;
    font-family: 'Arial', sans-serif;
}}

.document-table th,
.document-table td {{
    border: {border_width}px solid {border_color};
    padding: {cell_padding}pt;
    text-align: left;
    vertical-align: top;
}}

.document-table th {{
    background-color: {header_bg};
    font-weight: bold;
}}

.document-table tr:nth-child(even) {{
    background-color: #F2F2F2;
}}
"""
        return css
    
    def generate_page_css(self, 
                         page_size: str = "A4",
                         margin_style: str = "normal",
                         orientation: str = "portrait") -> str:
        """Generate CSS for page layout matching document formats"""
        
        # Page dimensions (approximate for CSS)
        page_dimensions = {
            "A4": {"width": "210mm", "height": "297mm"},
            "Letter": {"width": "8.5in", "height": "11in"},
            "Legal": {"width": "8.5in", "height": "14in"}
        }
        
        if orientation == "landscape":
            dims = page_dimensions[page_size]
            page_dimensions[page_size] = {"width": dims["height"], "height": dims["width"]}
        
        margins = self.page_margins[margin_style]
        page_dims = page_dimensions[page_size]
        
        css = f"""
.document-page {{
    width: {page_dims['width']};
    min-height: {page_dims['height']};
    max-width: 100%;
    margin: 20px auto;
    padding: {margins['top']} {margins['right']} {margins['bottom']} {margins['left']};
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    box-sizing: border-box;
    position: relative;
}}

@media (max-width: 768px) {{
    .document-page {{
        width: 95%;
        margin: 10px auto;
        padding: 15mm;
    }}
}}

.document-page {{
    font-family: 'Times New Roman', serif;
    font-size: 12pt;
    line-height: 1.15;
    color: #000000;
}}
"""
        return css
    
    def generate_complete_stylesheet(self, 
                                   include_headings: bool = True,
                                   include_tables: bool = True,
                                   include_page_layout: bool = True) -> str:
        """Generate complete CSS stylesheet for document preview"""
        
        stylesheets = []
        
        # Page layout
        if include_page_layout:
            stylesheets.append("/* Page Layout */")
            stylesheets.append(self.generate_page_css())
        
        # Heading styles  
        if include_headings:
            stylesheets.append("/* Heading Styles */")
            stylesheets.append(self.generate_heading_styles())
        
        # Table styles
        if include_tables:
            stylesheets.append("/* Table Styles */")
            stylesheets.append(self.generate_table_css())
        
        # Normal paragraph style
        normal_font = FontStyle(family="Times New Roman", size=12)
        stylesheets.append("/* Normal Text */")
        stylesheets.append(self.generate_paragraph_css(
            "normal", normal_font,
            line_spacing=1.15, space_after=6
        ))
        
        # Zoom and responsive styles
        stylesheets.append("""
/* Zoom Controls and Responsive */
.zoom-controls {
    position: sticky;
    top: 10px;
    right: 10px;
    z-index: 100;
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.document-page {
    transition: transform 0.3s ease;
}

.zoom-50 .document-page { transform: scale(0.5); transform-origin: top center; }
.zoom-75 .document-page { transform: scale(0.75); transform-origin: top center; }
.zoom-100 .document-page { transform: scale(1.0); transform-origin: top center; }
.zoom-125 .document-page { transform: scale(1.25); transform-origin: top center; }
.zoom-150 .document-page { transform: scale(1.5); transform-origin: top center; }
.zoom-200 .document-page { transform: scale(2.0); transform-origin: top center; }
""")
        
        return "\n\n".join(stylesheets)
    
    def map_docx_style_to_css(self, docx_style: Dict[str, Any]) -> Dict[str, str]:
        """Convert python-docx style object to CSS properties"""
        css_props = {}
        
        # Font properties from docx style
        if 'font_name' in docx_style:
            css_props['font-family'] = self.docx_font_mapping.get(
                docx_style['font_name'],
                f"'{docx_style['font_name']}', serif"
            )
        
        if 'font_size' in docx_style:
            # docx font size is in Pt (points)
            css_props['font-size'] = f"{docx_style['font_size']}pt"
        
        if 'bold' in docx_style:
            css_props['font-weight'] = 'bold' if docx_style['bold'] else 'normal'
        
        if 'italic' in docx_style:
            css_props['font-style'] = 'italic' if docx_style['italic'] else 'normal'
        
        if 'color' in docx_style:
            css_props['color'] = docx_style['color']
        
        # Paragraph properties
        if 'alignment' in docx_style:
            alignment_map = {
                'left': 'left',
                'center': 'center', 
                'right': 'right',
                'justify': 'justify'
            }
            css_props['text-align'] = alignment_map.get(docx_style['alignment'], 'left')
        
        if 'line_spacing' in docx_style:
            css_props['line-height'] = str(docx_style['line_spacing'])
        
        if 'space_before' in docx_style:
            css_props['margin-top'] = f"{docx_style['space_before']}pt"
        
        if 'space_after' in docx_style:
            css_props['margin-bottom'] = f"{docx_style['space_after']}pt"
        
        return css_props


def demo_style_mapping():
    """Demonstrate style mapping functionality"""
    mapper = DocumentStyleMapper()
    
    print("=== Document Style Mapper Demo ===\n")
    
    # Demo font mapping
    print("1. Font Style Mapping:")
    font = FontStyle(family="Arial", size=14, bold=True, color="#1F497D")
    css_props = mapper.map_font_to_css(font, DocumentFormat.DOCX)
    print(f"Font Style: {font}")
    print("CSS Properties:")
    for prop, value in css_props.items():
        print(f"  {prop}: {value};")
    print()
    
    # Demo paragraph CSS generation
    print("2. Paragraph CSS Generation:")
    para_css = mapper.generate_paragraph_css(
        "custom-heading", font,
        line_spacing=1.2, space_before=12, space_after=6, alignment="center"
    )
    print(para_css)
    print()
    
    # Demo complete stylesheet
    print("3. Complete Stylesheet Preview:")
    stylesheet = mapper.generate_complete_stylesheet()
    print(stylesheet[:500] + "...\n[truncated]")
    
    # Demo DOCX style conversion
    print("4. DOCX Style Conversion:")
    docx_style = {
        'font_name': 'Calibri',
        'font_size': 11,
        'bold': False,
        'italic': True,
        'color': '#000000',
        'alignment': 'justify',
        'line_spacing': 1.15,
        'space_after': 6
    }
    css_props = mapper.map_docx_style_to_css(docx_style)
    print("DOCX Style:", docx_style)
    print("Converted CSS:")
    for prop, value in css_props.items():
        print(f"  {prop}: {value};")


if __name__ == "__main__":
    demo_style_mapping()