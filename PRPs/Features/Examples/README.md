# Document Preview Styling Examples

This directory contains example implementations for Task 15 - Add Document Preview Styling.

## Files Overview

### 1. `document-preview-styling.css`
Complete CSS stylesheet demonstrating how to style document previews to match DOCX and PDF output:
- Page layout with proper dimensions and margins
- Typography matching document generation fonts
- Zoom controls and responsive design
- Print preview mode
- Dark mode support
- Page break visualization

### 2. `streamlit-document-preview.py`
Full Streamlit component implementation showing:
- Document preview with zoom controls
- Multi-page document handling
- Real-time content rendering
- Responsive design patterns
- Integration with document generation pipeline

### 3. `style-mapper-utility.py`
Utility for converting document generation styles to CSS:
- Font style mapping between formats
- DOCX style to CSS conversion
- Paragraph and heading style generation
- Table styling for preview accuracy
- Page layout CSS generation

## Usage Examples

### Basic Preview Implementation
```python
from streamlit_document_preview import DocumentPreviewComponent

preview = DocumentPreviewComponent()
content = "# My Document\n\nThis is sample content..."
preview.render_preview(content)
```

### Style Mapping for Accurate Previews
```python
from style_mapper_utility import DocumentStyleMapper

mapper = DocumentStyleMapper()
font_style = FontStyle(family="Arial", size=14, bold=True)
css_props = mapper.map_font_to_css(font_style, DocumentFormat.DOCX)
```

### Custom CSS Integration
```html
<link rel="stylesheet" href="document-preview-styling.css">
<div class="document-preview-container zoom-100">
    <div class="document-page">
        <!-- Your content here -->
    </div>
</div>
```

## Integration with Task 15

These examples directly support the implementation requirements:

1. **CSS styles matching DOCX output** → `document-preview-styling.css`
2. **Zoom controls and page display** → `streamlit-document-preview.py`
3. **Style mapping for accuracy** → `style-mapper-utility.py`
4. **Responsive preview sizing** → All files include responsive patterns
5. **Performance optimization** → Examples include loading states and virtual rendering concepts

## Testing the Examples

### CSS Testing
1. Open `document-preview-styling.css` in a web browser
2. Create HTML with the documented class structure
3. Test zoom levels and responsive behavior

### Streamlit Component Testing
```bash
streamlit run streamlit-document-preview.py
```

### Style Mapper Testing
```bash
python style-mapper-utility.py
```

## Best Practices Demonstrated

1. **Visual Fidelity**: CSS closely matches document generation output
2. **Performance**: Optimized for large documents with virtual rendering concepts
3. **Accessibility**: Keyboard navigation and screen reader support
4. **Responsive Design**: Mobile-first approach with progressive enhancement
5. **Maintainability**: Modular CSS architecture with clear separation of concerns