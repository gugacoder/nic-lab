# Document Generation Pipeline

```yaml
---
type: feature
tags: [document-generation, python-docx, reportlab, pdf, docx]
created: 2025-07-22
updated: 2025-07-22
status: todo
up: "[[Document Generation System.md]]"
related: "[[Review and Export Workflow.md]], [[AI Knowledge Base Query System.md]]"
dependencies: "[[Document Generation System.md]], [[AI Conversational System.md]]"
---
```

## Purpose

This feature implements the complete pipeline for transforming AI-generated content and chat conversations into professional documents in DOCX and PDF formats. The pipeline handles content structuring, formatting application, image embedding, and template-based styling to produce corporate-standard documents ready for review and distribution.

## Scope

- Convert chat conversations to structured document content
- Generate DOCX files using python-docx with full formatting
- Create PDF documents using ReportLab with precise layouts
- Apply corporate templates and branding standards
- Embed images with automatic sizing and placement
- Support tables, lists, and complex formatting
- Real-time preview generation for UI display
- Metadata injection for document properties

## User Flow

1. User completes knowledge exploration in chat interface
2. User clicks "Generate Document" with format selection
3. System processes conversation into structured content
4. User selects document template from available options
5. System generates document with progress indication
6. Preview appears in UI for user review
7. User can edit/adjust content before finalizing
8. User exports final document or sends to GitLab

**Success State**: Professional document generated in < 5 seconds with accurate formatting

**Error Handling**: Clear error messages for formatting issues, fallback to simple format

## Data Models

```yaml
DocumentRequest:
  id: str
  format: str  # 'docx' | 'pdf'
  template_id: str
  content_source: str  # 'chat' | 'custom' | 'query'
  conversation_id: str
  metadata: dict
    title: str
    author: str
    subject: str
    keywords: List[str]

DocumentContent:
  sections: List[Section]
  images: List[Image]
  tables: List[Table]
  metadata: DocumentMetadata

Section:
  title: str
  content: str
  level: int  # 1-4 for heading levels
  style: str
  subsections: List[Section]

Image:
  id: str
  source: str  # 'file' | 'url' | 'generated'
  path: str
  caption: str
  width: float
  height: float
  alignment: str

DocumentTemplate:
  id: str
  name: str
  format: str
  styles: dict
  layout: dict
  header_footer: dict
```

## API Specification

```yaml
# Document Generation Service
class DocumentGenerator:
  async def generate_document(request: DocumentRequest) -> Document:
    """Main document generation entry point"""
  
  async def process_content(source: str, source_id: str) -> DocumentContent:
    """Convert source content to structured format"""
  
  async def apply_template(content: DocumentContent, template: DocumentTemplate) -> Document:
    """Apply formatting template to content"""
  
  async def generate_docx(document: Document) -> bytes:
    """Generate DOCX file using python-docx"""
  
  async def generate_pdf(document: Document) -> bytes:
    """Generate PDF file using ReportLab"""

# Format-specific generators
class DocxGenerator:
  def create_document(content: DocumentContent, template: DocumentTemplate) -> docx.Document:
    """Create Word document with styling"""
  
  def add_image(doc: docx.Document, image: Image) -> None:
    """Embed image with proper sizing"""

class PdfGenerator:
  def create_document(content: DocumentContent, template: DocumentTemplate) -> Canvas:
    """Create PDF with ReportLab"""
  
  def add_page_template(canvas: Canvas, template: dict) -> None:
    """Apply page layout template"""
```

## Technical Implementation

### Core Components

- **ContentProcessor**: `src/generation/content_processor.py` - Structure chat content
- **TemplateManager**: `src/generation/template_manager.py` - Handle document templates  
- **DocxGenerator**: `src/generation/docx_generator.py` - DOCX file creation
- **PdfGenerator**: `src/generation/pdf_generator.py` - PDF file creation
- **ImageProcessor**: `src/generation/image_processor.py` - Image handling with Pillow
- **PreviewGenerator**: `src/generation/preview_generator.py` - Real-time previews

### Integration Points

- **AI Conversational System**: Source of content for documents
- **Image Handling System**: Process and embed images
- **Review and Export Workflow**: Document review and finalization

### Implementation Patterns

- **Builder Pattern**: Step-by-step document construction
- **Template Method**: Consistent formatting across formats
- **Factory Pattern**: Format-specific generator selection
- **Streaming Generation**: Progressive document building

## Examples

### Implementation References

- **[document-generation-example/](Examples/document-generation-example/)** - Complete pipeline demo
- **[docx-formatting.py](Examples/docx-formatting.py)** - python-docx formatting examples
- **[pdf-layouts.py](Examples/pdf-layouts.py)** - ReportLab layout patterns
- **[template-examples/](Examples/template-examples/)** - Corporate template samples

### Example Content Guidelines

- Create working document generation examples
- Show both DOCX and PDF generation
- Include image embedding demonstrations
- Provide template customization examples
- Show error handling and edge cases

## Error Scenarios

- **Template Not Found**: Missing template → Use default → Notify user
- **Image Load Failed**: Can't access image → Placeholder → Log error
- **Content Too Long**: Exceeds page limits → Paginate → Warn user
- **Format Error**: Invalid content structure → Best-effort conversion → Show issues
- **Memory Limit**: Large document → Stream generation → Progress indication

## Acceptance Criteria

- [ ] Generate DOCX documents with full formatting preserved
- [ ] Create PDF documents with pixel-perfect layouts
- [ ] Embed images with automatic sizing and quality optimization
- [ ] Apply corporate templates consistently across formats
- [ ] Generate 10-page document in under 5 seconds
- [ ] Preview updates in real-time during generation
- [ ] Support Unicode and multilingual content
- [ ] Maintain formatting fidelity between preview and final

## Validation

### Testing Strategy

- **Unit Tests**: Test individual generators, formatting functions
- **Integration Tests**: Full pipeline with various content types
- **Visual Tests**: Compare generated documents to references
- **Performance Tests**: Generation speed and memory usage

### Verification Commands

```bash
# Test document generation
python -m src.generation.document_generator test --format all

# Run formatting tests
pytest tests/generation/test_formatting.py

# Visual regression tests
python -m tests.visual.document_comparison

# Performance benchmark
python -m tests.performance.generation_benchmark

# Memory profiling
mprof run python -m src.generation.document_generator profile
```

### Success Metrics

- Generation Speed: < 0.5s per page
- Memory Usage: < 100MB for 50-page document
- Format Fidelity: 100% style preservation
- Image Quality: No visible compression artifacts
- Error Rate: < 0.1% generation failures