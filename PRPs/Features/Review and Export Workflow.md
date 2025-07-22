# Review and Export Workflow

```yaml
---
type: feature
tags: [workflow, review, export, gitlab, ui]
created: 2025-07-22
updated: 2025-07-22
status: todo
up: "[[Streamlit Interface.md]]"
related: "[[Document Generation Pipeline.md]], [[GitLab Repository Integration.md]]"
dependencies: "[[Document Generation System.md]], [[GitLab Integration.md]], [[Streamlit Interface.md]]"
---
```

## Purpose

This feature implements the complete workflow for reviewing generated documents within the Streamlit interface and exporting them to GitLab repositories. It provides professionals with the ability to preview documents, make adjustments, approve content, and seamlessly commit finalized documents to version control with appropriate metadata and commit messages.

## Scope

- Real-time document preview in the UI with accurate rendering
- In-line editing capabilities for last-minute adjustments
- Version comparison showing changes from original
- GitLab destination selection with project/path browser
- Commit message generation with AI assistance
- Branch selection and creation for document updates
- Export confirmation with GitLab URL provision
- Rollback capability for recent exports

## User Flow

1. User generates document from chat conversation
2. Document preview appears in dedicated pane
3. User reviews content and formatting in preview
4. User makes edits directly in preview if needed
5. User selects GitLab destination (project/path)
6. System suggests appropriate commit message
7. User confirms export with optional message edit
8. System commits document and provides GitLab link
9. User receives confirmation with direct link

**Success State**: Document successfully committed to GitLab with clear confirmation

**Error Handling**: Clear permission errors, retry options for failures, rollback capability

## Data Models

```yaml
ReviewSession:
  id: str
  document_id: str
  original_content: str
  edited_content: str
  changes: List[Change]
  created_at: datetime
  status: str  # 'reviewing' | 'approved' | 'exported'

Change:
  type: str  # 'edit' | 'format' | 'delete' | 'add'
  location: str
  original: str
  modified: str
  timestamp: datetime

ExportRequest:
  document_id: str
  format: str
  gitlab_project_id: int
  target_path: str
  branch: str
  commit_message: str
  author: dict
    name: str
    email: str

ExportResult:
  request_id: str
  success: bool
  gitlab_url: str
  commit_sha: str
  error_message: str
  timestamp: datetime
```

## API Specification

```yaml
# Review and Export Service
class ReviewExportService:
  async def start_review(document_id: str) -> ReviewSession:
    """Initialize document review session"""
  
  async def update_content(session_id: str, changes: List[Change]) -> None:
    """Apply edits to document content"""
  
  async def preview_changes(session_id: str) -> bytes:
    """Generate preview with changes applied"""
  
  async def suggest_commit_message(document: Document) -> str:
    """AI-generated commit message suggestion"""
  
  async def export_to_gitlab(request: ExportRequest) -> ExportResult:
    """Commit document to GitLab repository"""

# UI Components
class DocumentReviewer:
  def render_preview(document: bytes, format: str) -> None:
    """Display document preview in Streamlit"""
  
  def render_editor(content: str) -> str:
    """Provide editing interface"""
  
  def render_export_dialog() -> ExportRequest:
    """GitLab destination selection UI"""
```

## Technical Implementation

### Core Components

- **ReviewManager**: `src/workflows/review_manager.py` - Review session handling
- **PreviewRenderer**: `src/ui/preview_renderer.py` - Document preview display
- **EditorComponent**: `src/ui/editor_component.py` - In-line editing interface
- **ExportDialog**: `src/ui/export_dialog.py` - GitLab destination selection
- **CommitService**: `src/workflows/commit_service.py` - GitLab commit operations
- **ChangeTracker**: `src/workflows/change_tracker.py` - Edit history management

### Integration Points

- **Document Generation Pipeline**: Receives generated documents for review
- **GitLab Repository Integration**: Commits approved documents
- **AI Conversational System**: Generates commit message suggestions

### Implementation Patterns

- **Preview Isolation**: Sandbox preview rendering for security
- **Optimistic Updates**: Show changes immediately, sync in background
- **Atomic Commits**: Ensure document integrity during export
- **Progressive Enhancement**: Basic functionality works without JavaScript

## Examples

### Implementation References

- **[review-workflow-example/](Examples/review-workflow-example/)** - Complete workflow implementation
- **[streamlit-preview.py](Examples/streamlit-preview.py)** - Document preview in Streamlit
- **[inline-editor.py](Examples/inline-editor.py)** - Editing component example
- **[gitlab-export-flow.py](Examples/gitlab-export-flow.py)** - Export process demo

### Example Content Guidelines

- Show complete review to export flow
- Demonstrate preview rendering techniques
- Include editing UI examples
- Show GitLab destination browser
- Provide error handling examples

## Error Scenarios

- **Preview Failed**: Rendering error → Fallback to text → Download option
- **Edit Conflict**: Concurrent edits → Merge dialog → Manual resolution  
- **Export Permission**: No write access → Show requirements → Alternative options
- **Network Failure**: GitLab unreachable → Retry options → Local save
- **Commit Failed**: GitLab rejection → Show reason → Correction guidance

## Acceptance Criteria

- [ ] Document preview renders accurately within 2 seconds
- [ ] Edits apply immediately with visual feedback
- [ ] GitLab project browser shows only accessible projects
- [ ] Commit messages are descriptive and follow conventions
- [ ] Export completes within 5 seconds for typical documents
- [ ] Success confirmation includes clickable GitLab link
- [ ] Failed exports can be retried without data loss
- [ ] Recent exports can be rolled back if needed

## Validation

### Testing Strategy

- **Unit Tests**: Test preview rendering, change tracking, commit building
- **Integration Tests**: Full workflow from review to GitLab commit
- **UI Tests**: Streamlit component interactions and responsiveness
- **User Tests**: Workflow usability with real users

### Verification Commands

```bash
# Test review workflow
python -m src.workflows.review_manager test

# Run UI component tests
pytest tests/ui/test_review_components.py

# Integration test with GitLab
pytest tests/integration/test_export_workflow.py --gitlab-test

# Manual UI testing
streamlit run tests/manual/review_workflow_test.py
```

### Success Metrics

- Preview Load Time: < 2s for 20-page document
- Edit Responsiveness: < 100ms for text changes
- Export Success Rate: > 99% for valid requests
- User Completion Rate: > 95% start-to-finish
- Error Recovery Rate: > 90% successful retries