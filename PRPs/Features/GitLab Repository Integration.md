# GitLab Repository Integration

```yaml
---
type: feature
tags: [gitlab, api, integration, python-gitlab]
created: 2025-07-22
updated: 2025-07-22
status: todo
up: "[[GitLab Integration.md]]"
related: "[[Knowledge Base Architecture.md]], [[Document Generation Pipeline.md]]"
dependencies: "[[GitLab Integration.md]], [[Knowledge Base Architecture.md]]"
---
```

## Purpose

This feature establishes secure, authenticated connectivity between the NIC Chat system and self-hosted GitLab instances using the python-gitlab library. It enables the system to read from repositories and wikis for knowledge base queries, and write generated documents back to appropriate locations, maintaining full version control and access control compliance.

## Scope

- GitLab API authentication with token management
- Repository browsing and file retrieval capabilities
- Wiki page reading and parsing functionality
- Document commit operations with meaningful messages
- Multi-project search and aggregation
- Branch management for document updates
- Rate limiting and connection pooling
- Caching layer for performance optimization

## User Flow

1. Administrator configures GitLab connection with API token
2. System validates connection and available permissions
3. User queries knowledge base through chat interface
4. System searches across configured GitLab projects
5. Relevant content retrieved and processed for AI
6. User generates document from conversation
7. System commits document to specified GitLab location
8. User receives confirmation with GitLab link

**Success State**: Seamless GitLab operations transparent to end user with fast response times

**Error Handling**: Clear permission errors, automatic retry for transient failures, offline mode fallback

## Data Models

```yaml
GitLabConfig:
  url: str  # GitLab instance URL
  private_token: str  # API access token
  default_branch: str  # Default branch for operations
  projects: List[int]  # Project IDs to access
  search_scope: str  # 'projects' | 'groups' | 'all'
  cache_ttl: int  # Cache time-to-live in seconds

GitLabProject:
  id: int
  name: str
  path_with_namespace: str
  default_branch: str
  web_url: str
  wikis_enabled: bool
  last_activity_at: datetime

GitLabContent:
  project_id: int
  file_path: str
  content: str
  commit_id: str
  last_modified: datetime
  author: str
  size: int
```

## API Specification

```yaml
# GitLab integration service interface
class GitLabService:
  def authenticate(token: str, url: str) -> bool:
    """Validate GitLab credentials"""
  
  def get_projects() -> List[GitLabProject]:
    """List accessible projects"""
  
  def search_content(query: str, projects: List[int] = None) -> List[GitLabContent]:
    """Search across repositories and wikis"""
  
  def get_file(project_id: int, file_path: str, ref: str = None) -> GitLabContent:
    """Retrieve specific file content"""
  
  def create_file(project_id: int, file_path: str, content: str, 
                  commit_message: str, branch: str = None) -> dict:
    """Create new file in repository"""
  
  def update_file(project_id: int, file_path: str, content: str,
                  commit_message: str, branch: str = None) -> dict:
    """Update existing file"""
```

## Technical Implementation

### Core Components

- **GitLabClient**: `src/integrations/gitlab_client.py` - Main GitLab API wrapper
- **AuthManager**: `src/integrations/auth_manager.py` - Token storage and validation
- **ContentSearcher**: `src/integrations/content_searcher.py` - Multi-project search
- **WikiProcessor**: `src/integrations/wiki_processor.py` - Wiki content handling
- **CacheManager**: `src/integrations/cache_manager.py` - Response caching layer
- **RateLimiter**: `src/integrations/rate_limiter.py` - API rate limit handling

### Integration Points

- **Knowledge Base Architecture**: Provides content for indexing and search
- **AI Conversational System**: Supplies context for AI responses
- **Document Generation Pipeline**: Receives generated documents for storage

### Implementation Patterns

- **Connection Pooling**: Reuse HTTP connections for performance
- **Exponential Backoff**: Retry failed requests with increasing delays
- **Lazy Loading**: Fetch content only when needed
- **Batch Operations**: Group API calls when possible

## Examples

### Implementation References

- **[gitlab-integration-example/](Examples/gitlab-integration-example/)** - Complete integration setup
- **[gitlab-search-pattern.py](Examples/gitlab-search-pattern.py)** - Efficient search implementation
- **[gitlab-auth-flow.py](Examples/gitlab-auth-flow.py)** - Authentication best practices

### Example Content Guidelines

- Demonstrate secure credential management
- Show error handling for common GitLab API errors
- Include performance optimization techniques
- Provide migration path from different GitLab versions
- Example multi-project search implementation

## Error Scenarios

- **Authentication Failed**: Invalid token → Clear error message → Re-prompt for token
- **Permission Denied**: Insufficient access → List required permissions → Graceful degradation
- **Rate Limited**: Too many requests → Implement backoff → Show progress to user
- **Network Timeout**: Connection issues → Retry with cache → Offline mode if available
- **Project Not Found**: Invalid project ID → Validate configuration → Suggest alternatives

## Acceptance Criteria

- [ ] Successfully authenticate with GitLab API using provided token
- [ ] Search across multiple projects returns relevant results within 3 seconds
- [ ] File retrieval works for all supported formats (md, txt, json, yaml)
- [ ] Document commits include author info and descriptive messages
- [ ] Rate limiting never causes user-facing errors
- [ ] Cache improves response time by >50% for repeated queries
- [ ] Support for GitLab CE versions 14.x and above

## Validation

### Testing Strategy

- **Unit Tests**: Test individual API methods, error handling, caching
- **Integration Tests**: Test against GitLab test instance, various permissions
- **Performance Tests**: Measure search speed, cache effectiveness

### Verification Commands

```bash
# Test GitLab connection
python -m src.integrations.gitlab_client test-connection

# Run integration tests
pytest tests/integration/test_gitlab_integration.py

# Performance benchmark
python -m tests.performance.gitlab_benchmark

# Verify cache behavior
python -m src.integrations.cache_manager stats
```

### Success Metrics

- Authentication Time: < 500ms
- Search Performance: < 3s for 1000 files
- Cache Hit Rate: > 80% after warmup
- API Error Rate: < 0.1%
- Commit Success Rate: > 99.9%