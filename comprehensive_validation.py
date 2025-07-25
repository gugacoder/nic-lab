#!/usr/bin/env python3
"""
Comprehensive validation test for Content Chunking implementation
Tests all acceptance criteria from Task 12
"""

import asyncio
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.chunker import ContentChunker, ChunkingConfig, ChunkingStrategy

# Test content with various elements
TEST_CONTENT = """
# Document Processing System

This document describes a comprehensive document processing system that handles various content types with intelligent chunking strategies.

## Overview

The document processing system is designed to handle complex documents containing multiple content types including text, code, tables, and lists. The system uses advanced algorithms to preserve document structure while creating semantically meaningful chunks.

### Key Features

The system provides the following capabilities:

- **Intelligent chunking** that preserves semantic boundaries
- **Structure preservation** for headers, code blocks, and tables  
- **Overlap management** to maintain context between chunks
- **Quality scoring** based on multiple metrics
- **Metadata preservation** throughout the processing pipeline

## Implementation Details

### Core Components

The system consists of several key components working together:

```python
from src.preprocessing.chunker import ContentChunker, ChunkingConfig

# Initialize chunker with hybrid strategy
config = ChunkingConfig(
    strategy=ChunkingStrategy.HYBRID,
    target_chunk_size=1500,
    overlap_ratio=0.1,
    preserve_structure=True
)

chunker = ContentChunker(config)

# Process document
async def process_document(content):
    chunks = await chunker.chunk_document(
        content, 
        document_id="test_doc",
        file_path="document.md"
    )
    return chunks

# Quality assessment
def assess_quality(chunks):
    total_quality = sum(chunk.quality_score for chunk in chunks)
    return total_quality / len(chunks) if chunks else 0
```

### Processing Pipeline

The processing pipeline follows these steps:

1. **Document Analysis** - Analyze structure and content type
2. **Strategy Selection** - Choose optimal chunking approach
3. **Initial Chunking** - Apply selected strategy
4. **Overlap Application** - Add intelligent overlap
5. **Quality Enhancement** - Apply post-processing optimizations
6. **Metadata Preservation** - Maintain context information

## Data Structures

The system uses several important data structures:

| Component | Purpose | Key Fields |
|-----------|---------|------------|
| ContentChunk | Represents processed chunk | content, metadata, quality metrics |
| DocumentStructure | Document analysis results | sections, complexity, content_type |
| ChunkingConfig | Processing parameters | strategy, sizes, overlap settings |
| OverlapStrategy | Overlap configuration | ratios, boundaries, quality thresholds |

### Performance Characteristics

The system is optimized for performance with these characteristics:

- **Processing Speed**: > 100 pages per minute for typical documents
- **Quality Metrics**: 95% semantic completeness target
- **Memory Efficiency**: Streaming processing for large documents
- **Cache Utilization**: Analysis results cached for repeated processing

## Advanced Features

### Semantic Analysis

The semantic analysis component provides:

- Topic boundary detection using sliding window analysis
- Transition word recognition for natural breaks
- Lexical similarity scoring between content sections
- Quality assessment based on semantic coherence

### Code Block Handling

Special handling for code blocks ensures:

```javascript
function processCodeBlock(code) {
    // Preserve function boundaries
    const functions = extractFunctions(code);
    
    // Maintain syntax integrity  
    const validatedChunks = validateSyntax(functions);
    
    // Add language-specific metadata
    return enhanceWithMetadata(validatedChunks);
}

class DocumentProcessor {
    constructor(config) {
        this.config = config;
        this.stats = new ProcessingStats();
    }
    
    async processDocument(content) {
        const analysis = await this.analyzeDocument(content);
        const chunks = await this.createChunks(content, analysis);
        return this.optimizeChunks(chunks);
    }
}
```

### List Processing

Lists are handled with special care:

- Nested lists maintain their hierarchy
- Item boundaries are preserved  
- Indentation levels are maintained
- Related items are kept together when possible

Example list structures:

1. **Primary Items**
   - Sub-item with details
   - Another sub-item
     - Nested sub-sub-item
     - Additional nested content

2. **Secondary Items** 
   - Different category items
   - Specialized handling

## Conclusion

This document processing system provides robust, efficient document chunking with semantic preservation and structural integrity. The implementation meets all quality and performance requirements for production use.
"""

async def run_comprehensive_validation():
    """Run comprehensive validation of all acceptance criteria"""
    
    print("🔍 Starting Comprehensive Content Chunking Validation")
    print("=" * 60)
    
    # Test configurations
    configs = [
        ("Semantic Strategy", ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC)),
        ("Structural Strategy", ChunkingConfig(strategy=ChunkingStrategy.STRUCTURAL)),
        ("Hybrid Strategy", ChunkingConfig(strategy=ChunkingStrategy.HYBRID)),
        ("Adaptive Strategy", ChunkingConfig(strategy=ChunkingStrategy.ADAPTIVE))
    ]
    
    results = {}
    
    for strategy_name, config in configs:
        print(f"\n🧪 Testing {strategy_name}")
        print("-" * 40)
        
        chunker = ContentChunker(config)
        start_time = time.time()
        
        # Process test document
        chunks = await chunker.chunk_document(
            TEST_CONTENT,
            document_id=f"test_{strategy_name.lower().replace(' ', '_')}",
            file_path="test_document.md"
        )
        
        processing_time = time.time() - start_time
        
        # Collect metrics
        result = await analyze_chunks(chunks, processing_time, strategy_name)
        results[strategy_name] = result
        
        # Print results
        print_strategy_results(result)
    
    # Overall validation
    print("\n" + "=" * 60)
    print("📊 OVERALL VALIDATION RESULTS")
    print("=" * 60)
    
    validate_acceptance_criteria(results)
    
    print("\n✅ Comprehensive validation completed successfully!")

async def analyze_chunks(chunks, processing_time, strategy_name):
    """Analyze chunks against acceptance criteria"""
    
    if not chunks:
        return {
            'chunks_count': 0,
            'avg_chunk_size': 0,
            'semantic_completeness': 0,
            'processing_time': processing_time,
            'has_code_blocks': False,
            'structure_preserved': False,
            'overlap_quality': 0,
            'token_compliance': False
        }
    
    # Basic metrics
    total_size = sum(chunk.size for chunk in chunks)
    avg_chunk_size = total_size / len(chunks)
    
    # Quality metrics
    avg_quality_score = sum(chunk.quality_score for chunk in chunks) / len(chunks)
    avg_semantic_coherence = sum(chunk.semantic_coherence_score for chunk in chunks) / len(chunks)
    avg_structural_completeness = sum(chunk.structural_completeness for chunk in chunks) / len(chunks)
    
    # Check code block preservation
    code_chunks = [c for c in chunks if c.chunk_type in ['code', 'section'] and '```' in c.content]
    has_intact_code_blocks = all('```' in chunk.content and chunk.content.count('```') % 2 == 0 
                                for chunk in code_chunks) if code_chunks else True
    
    # Check structure preservation
    has_headers = any('section_title' in (chunk.metadata or {}) for chunk in chunks)
    structure_preserved = has_headers or any(chunk.section_title for chunk in chunks)
    
    # Check overlap quality
    chunks_with_overlap = [c for c in chunks if c.metadata and c.metadata.get('has_overlap', False)]
    avg_overlap_quality = 0.7 if chunks_with_overlap else 0  # Simplified estimate
    
    # Token compliance (rough estimate)
    max_tokens = max(chunk.estimated_tokens for chunk in chunks) if chunks else 0
    token_compliant = max_tokens <= 2000  # Reasonable token limit
    
    return {
        'chunks_count': len(chunks),
        'avg_chunk_size': avg_chunk_size,
        'total_size': total_size,
        'avg_quality_score': avg_quality_score,
        'semantic_completeness': avg_semantic_coherence * 100,  # Convert to percentage
        'structural_completeness': avg_structural_completeness * 100,
        'processing_time': processing_time,
        'has_intact_code_blocks': has_intact_code_blocks,
        'structure_preserved': structure_preserved,
        'overlap_quality': avg_overlap_quality,
        'token_compliance': token_compliant,
        'processing_speed_pages_per_min': (len(TEST_CONTENT) / 1000) / (processing_time / 60) if processing_time > 0 else 0
    }

def print_strategy_results(result):
    """Print results for a single strategy"""
    
    print(f"  📄 Chunks created: {result['chunks_count']}")
    print(f"  📏 Average chunk size: {result['avg_chunk_size']:.0f} characters")
    print(f"  🎯 Quality score: {result['avg_quality_score']:.3f}")
    print(f"  🧠 Semantic completeness: {result['semantic_completeness']:.1f}%")
    print(f"  🏗️  Structural completeness: {result['structural_completeness']:.1f}%")
    print(f"  ⚡ Processing time: {result['processing_time']:.3f}s")
    print(f"  📊 Processing speed: {result['processing_speed_pages_per_min']:.1f} pages/min")
    print(f"  💻 Code blocks intact: {'✅' if result['has_intact_code_blocks'] else '❌'}")
    print(f"  📋 Structure preserved: {'✅' if result['structure_preserved'] else '❌'}")
    print(f"  🔗 Token compliance: {'✅' if result['token_compliance'] else '❌'}")

def validate_acceptance_criteria(results):
    """Validate against the specific acceptance criteria from Task 12"""
    
    criteria = [
        ("Chunks maintain semantic coherence", "semantic_completeness", 95, ">="),
        ("Document structure is preserved in metadata", "structure_preserved", True, "=="),
        ("Code blocks remain intact", "has_intact_code_blocks", True, "=="),
        ("Chunk sizes fit within token limits", "token_compliance", True, "=="),
        ("Performance handles large documents efficiently", "processing_speed_pages_per_min", 100, ">="),
        ("Processing completes in reasonable time", "processing_time", 1.0, "<=")
    ]
    
    print("\n📋 Acceptance Criteria Validation:")
    print("-" * 50)
    
    all_passed = True
    
    for criterion, metric, threshold, operator in criteria:
        print(f"\n✓ {criterion}")
        
        strategy_results = []
        for strategy_name, result in results.items():
            value = result.get(metric, 0)
            
            if operator == ">=":
                passed = value >= threshold
            elif operator == "<=":
                passed = value <= threshold
            elif operator == "==":
                passed = value == threshold
            else:
                passed = False
            
            strategy_results.append((strategy_name, value, passed))
            
            if isinstance(value, bool):
                value_str = "✅" if value else "❌"
            elif isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            
            status = "✅" if passed else "❌"
            print(f"    {strategy_name:20} {value_str:>10} {status}")
            
            if not passed:
                all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL ACCEPTANCE CRITERIA PASSED!")
    else:
        print("⚠️  Some acceptance criteria need attention")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(run_comprehensive_validation())