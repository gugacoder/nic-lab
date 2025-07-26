"""
Style Mapper Utility

Maps document styles from generation system (DOCX/PDF) to CSS for accurate preview rendering.
Ensures visual fidelity between preview and final output formats.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import colorsys

logger = logging.getLogger(__name__)

# Import generation system models if available
try:
    from src.generation.models import TextStyle, StyleWeight, StyleEmphasis, ContentType
    GENERATION_MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Generation models not available, using fallback enums")
    GENERATION_MODELS_AVAILABLE = False
    
    # Fallback enums if generation models aren't available
    class StyleWeight(Enum):
        NORMAL = "normal"
        BOLD = "bold"
        LIGHT = "light"
    
    class StyleEmphasis(Enum):
        NONE = "none"
        ITALIC = "italic"
        UNDERLINE = "underline"
        STRIKETHROUGH = "strikethrough"
    
    class ContentType(Enum):
        PARAGRAPH = "paragraph"
        HEADING = "heading"
        LIST = "list"
        TABLE = "table"
        TEXT = "text"


@dataclass
class CSSStyle:
    """CSS style representation"""
    font_family: Optional[str] = None
    font_size: Optional[str] = None
    font_weight: Optional[str] = None
    font_style: Optional[str] = None
    text_decoration: Optional[str] = None
    color: Optional[str] = None
    background_color: Optional[str] = None
    text_align: Optional[str] = None
    line_height: Optional[str] = None
    margin: Optional[str] = None
    padding: Optional[str] = None
    border: Optional[str] = None
    additional_properties: Optional[Dict[str, str]] = None
    
    def to_css_string(self) -> str:
        """Convert to CSS style string"""
        styles = []
        
        if self.font_family:
            styles.append(f"font-family: {self.font_family}")
        if self.font_size:
            styles.append(f"font-size: {self.font_size}")
        if self.font_weight:
            styles.append(f"font-weight: {self.font_weight}")
        if self.font_style:
            styles.append(f"font-style: {self.font_style}")
        if self.text_decoration:
            styles.append(f"text-decoration: {self.text_decoration}")
        if self.color:
            styles.append(f"color: {self.color}")
        if self.background_color:
            styles.append(f"background-color: {self.background_color}")
        if self.text_align:
            styles.append(f"text-align: {self.text_align}")
        if self.line_height:
            styles.append(f"line-height: {self.line_height}")
        if self.margin:
            styles.append(f"margin: {self.margin}")
        if self.padding:
            styles.append(f"padding: {self.padding}")
        if self.border:
            styles.append(f"border: {self.border}")
        
        if self.additional_properties:
            for prop, value in self.additional_properties.items():
                styles.append(f"{prop}: {value}")
        
        return "; ".join(styles)


class StyleMapper:
    """
    Maps document generation styles to CSS for preview rendering.
    
    Ensures visual fidelity between document preview and final output
    by translating style definitions from DOCX and PDF generators.
    """
    
    def __init__(self):
        """Initialize style mapper with default mappings"""
        self._font_mappings = {
            # DOCX default fonts
            "Calibri": '"Calibri", "Helvetica Neue", Arial, sans-serif',
            "Times New Roman": '"Times New Roman", "Liberation Serif", Times, serif',
            "Arial": 'Arial, "Helvetica Neue", Helvetica, sans-serif',
            "Helvetica": '"Helvetica Neue", Helvetica, Arial, sans-serif',
            
            # PDF default fonts (ReportLab)
            "Helvetica": '"Helvetica Neue", Helvetica, Arial, sans-serif',
            "Times-Roman": '"Times New Roman", "Liberation Serif", Times, serif',
            "Courier": '"Courier New", Courier, monospace',
        }
        
        self._weight_mappings = {
            StyleWeight.NORMAL: "normal",
            StyleWeight.BOLD: "bold",
            StyleWeight.LIGHT: "300"
        }
        
        self._emphasis_mappings = {
            StyleEmphasis.NONE: {"font-style": "normal", "text-decoration": "none"},
            StyleEmphasis.ITALIC: {"font-style": "italic", "text-decoration": "none"},
            StyleEmphasis.UNDERLINE: {"font-style": "normal", "text-decoration": "underline"},
            StyleEmphasis.STRIKETHROUGH: {"font-style": "normal", "text-decoration": "line-through"}
        }
        
        self._content_type_mappings = {
            ContentType.PARAGRAPH: {
                "margin": "0 0 8pt 0",
                "line-height": "1.15"
            },
            ContentType.HEADING: {
                "font-size": "16pt",
                "font-weight": "bold",
                "color": "#2f5496",
                "margin": "12pt 0 6pt 0",
                "line-height": "1.15",
                "page-break-after": "avoid"
            },
            ContentType.LIST: {
                "margin": "0 0 4pt 0",
                "line-height": "1.15",
                "padding-left": "0.5in"
            },
            ContentType.TABLE: {
                "padding": "4pt 8pt",
                "border": "1pt solid #000000",
                "vertical-align": "top",
                "line-height": "1.15"
            },
            ContentType.TEXT: {
                "margin": "0",
                "line-height": "1.15"
            }
        }
    
    def map_text_style_to_css(self, text_style: Union[Dict[str, Any], Any]) -> CSSStyle:
        """
        Map a text style to CSS.
        
        Args:
            text_style: TextStyle object or dictionary with style properties
            
        Returns:
            CSSStyle object with mapped properties
        """
        css_style = CSSStyle()
        
        # Handle both dict and object formats
        if isinstance(text_style, dict):
            style_dict = text_style
        else:
            # Convert object to dict
            style_dict = {}
            if hasattr(text_style, 'font_family'):
                style_dict['font_family'] = text_style.font_family
            if hasattr(text_style, 'font_size'):
                style_dict['font_size'] = text_style.font_size
            if hasattr(text_style, 'weight'):
                style_dict['weight'] = text_style.weight
            if hasattr(text_style, 'emphasis'):
                style_dict['emphasis'] = text_style.emphasis
            if hasattr(text_style, 'color'):
                style_dict['color'] = text_style.color
            if hasattr(text_style, 'alignment'):
                style_dict['alignment'] = text_style.alignment
        
        # Map font family
        if 'font_family' in style_dict:
            css_style.font_family = self._map_font_family(style_dict['font_family'])
        
        # Map font size
        if 'font_size' in style_dict:
            css_style.font_size = self._map_font_size(style_dict['font_size'])
        
        # Map font weight
        if 'weight' in style_dict:
            css_style.font_weight = self._map_font_weight(style_dict['weight'])
        
        # Map emphasis (italic, underline, etc.)
        if 'emphasis' in style_dict:
            emphasis_styles = self._map_emphasis(style_dict['emphasis'])
            css_style.font_style = emphasis_styles.get('font-style')
            css_style.text_decoration = emphasis_styles.get('text-decoration')
        
        # Map color
        if 'color' in style_dict:
            css_style.color = self._map_color(style_dict['color'])
        
        # Map alignment
        if 'alignment' in style_dict:
            css_style.text_align = self._map_alignment(style_dict['alignment'])
        
        return css_style
    
    def map_content_type_to_css(self, content_type: ContentType) -> CSSStyle:
        """
        Map content type to default CSS styles.
        
        Args:
            content_type: Type of content element
            
        Returns:
            CSSStyle with default styles for the content type
        """
        css_style = CSSStyle()
        
        if content_type in self._content_type_mappings:
            mapping = self._content_type_mappings[content_type]
            
            css_style.font_family = mapping.get('font-family')
            css_style.font_size = mapping.get('font-size')
            css_style.font_weight = mapping.get('font-weight')
            css_style.font_style = mapping.get('font-style')
            css_style.color = mapping.get('color')
            css_style.text_align = mapping.get('text-align')
            css_style.line_height = mapping.get('line-height')
            css_style.margin = mapping.get('margin')
            css_style.padding = mapping.get('padding')
            css_style.border = mapping.get('border')
            
            # Additional properties
            additional = {}
            for key, value in mapping.items():
                if key not in ['font-family', 'font-size', 'font-weight', 'font-style', 
                              'color', 'text-align', 'line-height', 'margin', 'padding', 'border']:
                    additional[key] = value
            
            if additional:
                css_style.additional_properties = additional
        
        return css_style
    
    def _map_font_family(self, font_family: str) -> str:
        """Map font family name to CSS font stack"""
        return self._font_mappings.get(font_family, f'"{font_family}", sans-serif')
    
    def _map_font_size(self, font_size: Union[int, str]) -> str:
        """Map font size to CSS format"""
        if isinstance(font_size, (int, float)):
            return f"{font_size}pt"
        elif isinstance(font_size, str):
            if font_size.endswith(('pt', 'px', 'em', 'rem', '%')):
                return font_size
            else:
                # Assume it's a number without unit
                try:
                    size_num = float(font_size)
                    return f"{size_num}pt"
                except ValueError:
                    return font_size
        return str(font_size)
    
    def _map_font_weight(self, weight: Union[StyleWeight, str]) -> str:
        """Map font weight to CSS format"""
        if isinstance(weight, StyleWeight):
            return self._weight_mappings.get(weight, "normal")
        elif isinstance(weight, str):
            weight_lower = weight.lower()
            if weight_lower in ["bold", "normal", "lighter", "bolder"]:
                return weight_lower
            elif weight_lower in ["100", "200", "300", "400", "500", "600", "700", "800", "900"]:
                return weight_lower
            else:
                return "normal"
        return "normal"
    
    def _map_emphasis(self, emphasis: Union[StyleEmphasis, str]) -> Dict[str, str]:
        """Map text emphasis to CSS properties"""
        if isinstance(emphasis, StyleEmphasis):
            return self._emphasis_mappings.get(emphasis, {"font-style": "normal", "text-decoration": "none"})
        elif isinstance(emphasis, str):
            emphasis_lower = emphasis.lower()
            if emphasis_lower == "italic":
                return {"font-style": "italic", "text-decoration": "none"}
            elif emphasis_lower == "underline":
                return {"font-style": "normal", "text-decoration": "underline"}
            elif emphasis_lower == "strikethrough":
                return {"font-style": "normal", "text-decoration": "line-through"}
            else:
                return {"font-style": "normal", "text-decoration": "none"}
        return {"font-style": "normal", "text-decoration": "none"}
    
    def _map_color(self, color: Union[str, tuple, Any]) -> str:
        """Map color value to CSS color"""
        if isinstance(color, str):
            if color.startswith('#'):
                return color
            elif color.startswith('rgb'):
                return color
            elif color.lower() in ['black', 'white', 'red', 'green', 'blue', 'yellow', 'orange', 'purple']:
                return color.lower()
            else:
                return color
        elif isinstance(color, (tuple, list)) and len(color) >= 3:
            # RGB tuple
            r, g, b = color[:3]
            if all(isinstance(c, (int, float)) for c in [r, g, b]):
                if all(c <= 1.0 for c in [r, g, b]):
                    # Normalized RGB (0-1)
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                return f"rgb({r}, {g}, {b})"
        
        # Try to get color from object attributes
        if hasattr(color, 'red') and hasattr(color, 'green') and hasattr(color, 'blue'):
            return f"rgb({color.red}, {color.green}, {color.blue})"
        
        return "#000000"  # Default to black
    
    def _map_alignment(self, alignment: Union[str, Any]) -> str:
        """Map text alignment to CSS"""
        if isinstance(alignment, str):
            alignment_lower = alignment.lower()
            if alignment_lower in ['left', 'center', 'right', 'justify']:
                return alignment_lower
        
        # Try to handle enum-like objects
        if hasattr(alignment, 'value'):
            return self._map_alignment(alignment.value)
        
        return "left"  # Default alignment
    
    def generate_css_class(
        self,
        class_name: str,
        text_style: Optional[Union[Dict[str, Any], Any]] = None,
        content_type: Optional[ContentType] = None,
        additional_styles: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate complete CSS class definition.
        
        Args:
            class_name: CSS class name
            text_style: Text style to map
            content_type: Content type for default styles
            additional_styles: Additional CSS properties
            
        Returns:
            Complete CSS class definition string
        """
        css_style = CSSStyle()
        
        # Apply content type defaults first
        if content_type:
            content_css = self.map_content_type_to_css(content_type)
            css_style = self._merge_css_styles(css_style, content_css)
        
        # Apply text style overrides
        if text_style:
            text_css = self.map_text_style_to_css(text_style)
            css_style = self._merge_css_styles(css_style, text_css)
        
        # Apply additional styles
        if additional_styles:
            if css_style.additional_properties:
                css_style.additional_properties.update(additional_styles)
            else:
                css_style.additional_properties = additional_styles.copy()
        
        # Generate CSS class
        css_properties = css_style.to_css_string()
        
        formatted_properties = css_properties.replace('; ', ';\n    ')
        return f".{class_name} {{\n    {formatted_properties}\n}}"
    
    def _merge_css_styles(self, base: CSSStyle, override: CSSStyle) -> CSSStyle:
        """Merge two CSS styles, with override taking precedence"""
        merged = CSSStyle()
        
        # Copy base properties
        for attr in ['font_family', 'font_size', 'font_weight', 'font_style', 
                     'text_decoration', 'color', 'background_color', 'text_align',
                     'line_height', 'margin', 'padding', 'border']:
            base_value = getattr(base, attr)
            override_value = getattr(override, attr)
            setattr(merged, attr, override_value if override_value is not None else base_value)
        
        # Merge additional properties
        merged.additional_properties = {}
        if base.additional_properties:
            merged.additional_properties.update(base.additional_properties)
        if override.additional_properties:
            merged.additional_properties.update(override.additional_properties)
        
        if not merged.additional_properties:
            merged.additional_properties = None
        
        return merged
    
    def generate_style_sheet(
        self,
        style_definitions: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate complete CSS stylesheet from style definitions.
        
        Args:
            style_definitions: Dictionary mapping class names to style properties
            
        Returns:
            Complete CSS stylesheet string
        """
        css_classes = []
        
        for class_name, style_def in style_definitions.items():
            text_style = style_def.get('text_style')
            content_type = style_def.get('content_type')
            additional_styles = style_def.get('additional_styles')
            
            css_class = self.generate_css_class(
                class_name, text_style, content_type, additional_styles
            )
            css_classes.append(css_class)
        
        return "\n\n".join(css_classes)


# Global style mapper instance
_style_mapper = None

def get_style_mapper() -> StyleMapper:
    """Get global style mapper instance"""
    global _style_mapper
    if _style_mapper is None:
        _style_mapper = StyleMapper()
    return _style_mapper


# Utility functions

def map_docx_style_to_css(docx_style: Dict[str, Any]) -> str:
    """Quick utility to map DOCX style to CSS string"""
    mapper = get_style_mapper()
    css_style = mapper.map_text_style_to_css(docx_style)
    return css_style.to_css_string()


def map_pdf_style_to_css(pdf_style: Dict[str, Any]) -> str:
    """Quick utility to map PDF style to CSS string"""
    mapper = get_style_mapper()
    css_style = mapper.map_text_style_to_css(pdf_style)
    return css_style.to_css_string()


def generate_preview_styles_from_template(template_styles: Dict[str, Any]) -> str:
    """Generate CSS stylesheet from document template styles"""
    mapper = get_style_mapper()
    
    style_definitions = {}
    
    for style_name, style_props in template_styles.items():
        style_definitions[f"doc-{style_name}"] = {
            'text_style': style_props,
            'additional_styles': {
                'page-break-inside': 'avoid'
            }
        }
    
    return mapper.generate_style_sheet(style_definitions)


def create_responsive_preview_css(base_styles: str) -> str:
    """Add responsive design rules to base preview styles"""
    responsive_css = f"""
{base_styles}

/* Responsive Design for Document Preview */
@media (max-width: 1024px) {{
    .document-page {{
        width: 100%;
        max-width: 794px;
        padding: 48px;
    }}
    
    .doc-heading-1 {{ font-size: 14pt; }}
    .doc-heading-2 {{ font-size: 12pt; }}
    .doc-heading-3 {{ font-size: 11pt; }}
    .doc-heading-4 {{ font-size: 10pt; }}
}}

@media (max-width: 768px) {{
    .document-page {{
        padding: 24px;
    }}
    
    .document-content {{
        font-size: 11pt;
    }}
    
    .doc-table {{
        font-size: 10pt;
    }}
}}

@media (max-width: 480px) {{
    .document-page {{
        padding: 16px;
    }}
    
    .document-content {{
        font-size: 10pt;
    }}
    
    .doc-table td,
    .doc-table th {{
        padding: 2pt 4pt;
    }}
}}
"""
    return responsive_css