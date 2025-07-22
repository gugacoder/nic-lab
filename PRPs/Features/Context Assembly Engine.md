# Context Assembly Engine

```yaml
---
type: feature
tags: [ai, context, rag, optimization]
created: 2025-07-22
updated: 2025-07-22
status: todo
up: "[[AI Conversational System.md]]"
related: "[[AI Knowledge Base Query System.md]], [[Knowledge Base Architecture.md]]"
dependencies: "[[AI Conversational System.md]], [[Knowledge Base Architecture.md]]"
---
```

## Purpose

This feature implements an intelligent context assembly system that dynamically constructs optimal context windows for AI queries by selecting, ranking, and combining relevant information from multiple sources. The engine ensures that the Groq API receives the most pertinent information within token limits while maintaining coherence and avoiding redundancy.

## Scope

- Dynamic context window sizing based on query complexity
- Intelligent chunk selection from search results
- Deduplication of overlapping content
- Context relevance scoring and ranking
- Token counting and limit management
- Semantic coherence preservation
- Metadata inclusion for source attribution
- Context caching for repeated queries

## User Flow

1. Search results arrive from knowledge base query
2. System analyzes query intent and complexity
3. Content is chunked into semantic units
4. Chunks are scored for relevance to query
5. Optimal chunks selected within token budget
6. Related chunks merged to avoid fragmentation
7. Context assembled with source metadata
8. Final context passed to AI for response generation

**Success State**: Highly relevant context that fits token limits and produces accurate responses

**Error Handling**: Graceful degradation when content exceeds limits, priority-based selection

## Data Models

```yaml
ContextRequest:
  query_id: str
  query_text: str
  search_results: List[SearchResult]
  max_tokens: int
  strategy: str  # 'balanced' | 'comprehensive' | 'focused'
  include_metadata: bool

ContentChunk:
  id: str
  source_id: str
  text: str
  tokens: int
  relevance_score: float
  position: int  # Position in source document
  semantic_group: str
  metadata: dict

AssembledContext:
  request_id: str
  chunks: List[ContentChunk]
  total_tokens: int
  sources: List[str]
  assembly_strategy: str
  quality_score: float

ContextCache:
  query_hash: str
  context: AssembledContext
  created_at: datetime
  hit_count: int
  ttl: int
```

## API Specification

```yaml
# Context Assembly Service
class ContextAssembler:
  async def assemble_context(request: ContextRequest) -> AssembledContext:
    """Main context assembly entry point"""
  
  async def chunk_content(results: List[SearchResult]) -> List[ContentChunk]:
    """Break content into semantic chunks"""
  
  async def score_chunks(chunks: List[ContentChunk], query: str) -> List[ContentChunk]:
    """Calculate relevance scores for chunks"""
  
  async def select_optimal_chunks(chunks: List[ContentChunk], limit: int) -> List[ContentChunk]:
    """Select best chunks within token limit"""
  
  async def merge_related_chunks(chunks: List[ContentChunk]) -> List[ContentChunk]:
    """Combine semantically related chunks"""

# Scoring and optimization
class RelevanceScorer:
  def calculate_relevance(chunk: str, query: str) -> float:
    """Score chunk relevance to query"""
  
  def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity"""

class TokenOptimizer:
  def count_tokens(text: str) -> int:
    """Count tokens for Llama model"""
  
  def optimize_selection(chunks: List[ContentChunk], limit: int) -> List[ContentChunk]:
    """Optimal chunk selection algorithm"""
```

## Technical Implementation

### Core Components

- **ChunkProcessor**: `src/ai/chunk_processor.py` - Semantic chunking logic
- **RelevanceScorer**: `src/ai/relevance_scorer.py` - Multi-factor scoring
- **TokenCounter**: `src/ai/token_counter.py` - Accurate token counting
- **ChunkSelector**: `src/ai/chunk_selector.py` - Optimization algorithms
- **ContextBuilder**: `src/ai/context_builder.py` - Final assembly logic
- **ContextCache**: `src/ai/context_cache.py` - Caching implementation

### Integration Points

- **AI Knowledge Base Query System**: Receives search results for processing
- **AI Conversational System**: Provides assembled context for generation
- **Knowledge Base Architecture**: Uses content structure information

### Implementation Patterns

- **Sliding Window**: Overlapping chunks for context preservation
- **Greedy Selection**: Highest relevance first within limits
- **Semantic Grouping**: Keep related content together
- **Hierarchical Assembly**: Important content first, details later

## Examples

### Implementation References

- **[context-assembly-example/](Examples/context-assembly-example/)** - Complete assembly pipeline
- **[chunking-strategies.py](Examples/chunking-strategies.py)** - Different chunking approaches
- **[relevance-scoring.py](Examples/relevance-scoring.py)** - Scoring algorithms
- **[token-optimization.py](Examples/token-optimization.py)** - Token limit handling

### Example Content Guidelines

- Show different assembly strategies
- Demonstrate chunking techniques
- Include relevance scoring examples
- Show token optimization in action
- Provide performance comparisons

## Error Scenarios

- **Token Overflow**: Content exceeds limits → Prioritize by relevance → Truncate gracefully
- **No Relevant Content**: Low scores → Include best available → Flag low confidence
- **Chunking Failed**: Malformed content → Fallback chunking → Best effort
- **Cache Miss**: No cached context → Full assembly → Update cache
- **Scoring Error**: Algorithm failure → Default scoring → Log for analysis

## Acceptance Criteria

- [ ] Assemble context within 500ms for typical queries
- [ ] Stay within token limits 100% of the time
- [ ] Achieve 90%+ relevance accuracy in user testing
- [ ] Cache hit rate > 40% for repeated queries
- [ ] Support context windows from 1K to 8K tokens
- [ ] Preserve semantic coherence in assembled context
- [ ] Include clear source attribution for all chunks
- [ ] Handle 50+ search results efficiently

## Validation

### Testing Strategy

- **Unit Tests**: Test chunking, scoring, selection algorithms
- **Integration Tests**: Full assembly pipeline with real data
- **Performance Tests**: Assembly speed and memory usage
- **Quality Tests**: Human evaluation of context relevance

### Verification Commands

```bash
# Test context assembly
python -m src.ai.context_assembler test --verbose

# Benchmark assembly performance
python -m tests.performance.context_benchmark

# Evaluate context quality
python -m tests.quality.context_evaluator --human-eval

# Cache effectiveness test
python -m tests.integration.cache_test --duration 1h

# Token accuracy verification
python -m tests.accuracy.token_counter_test
```

### Success Metrics

- Assembly Speed: < 500ms for 50 chunks
- Token Accuracy: ±5 tokens of actual
- Relevance Score: > 0.85 average
- Cache Hit Rate: > 40% in production
- Memory Usage: < 50MB per assembly