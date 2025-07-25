"""
Tests for Content Chunker

This module contains comprehensive tests for the main content chunking
functionality, including different strategies and edge cases.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass
from typing import List, Optional

# Import the components to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.chunker import (
    ContentChunker, ChunkingConfig, ContentChunk, ChunkingStrategy,
    create_content_chunker, chunk_document
)


@dataclass
class MockDocumentStructure:
    """Mock document structure for testing"""
    content_type: str = "markdown"
    language: Optional[str] = None
    is_highly_structured: bool = True
    has_clear_sections: bool = True
    sections: List = None
    
    def __post_init__(self):
        if self.sections is None:
            self.sections = []
    
    def get_section_at_position(self, position: int):
        return None


class TestChunkingConfig:
    """Test chunking configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ChunkingConfig()
        
        assert config.target_chunk_size == 1500
        assert config.max_chunk_size == 2000
        assert config.min_chunk_size == 200
        assert config.overlap_ratio == 0.1
        assert config.strategy == ChunkingStrategy.HYBRID
        assert config.preserve_structure is True
        assert config.include_metadata is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ChunkingConfig(
            target_chunk_size=1000,
            strategy=ChunkingStrategy.SEMANTIC,
            overlap_ratio=0.15,
            preserve_structure=False
        )
        
        assert config.target_chunk_size == 1000
        assert config.strategy == ChunkingStrategy.SEMANTIC
        assert config.overlap_ratio == 0.15
        assert config.preserve_structure is False


class TestContentChunk:
    """Test content chunk functionality"""
    
    def test_chunk_creation(self):
        """Test basic chunk creation"""
        chunk = ContentChunk(
            content="This is test content.",
            chunk_id="test_1",
            chunk_index=0,
            start_char=0,
            end_char=21
        )
        
        assert chunk.content == "This is test content."
        assert chunk.chunk_id == "test_1"
        assert chunk.size == 21
        assert chunk.estimated_tokens == 5  # Rough estimate
    
    def test_chunk_properties(self):
        """Test chunk computed properties"""
        chunk = ContentChunk(
            content="This is a longer piece of test content with multiple words.",
            chunk_id="test_2",
            chunk_index=1,
            start_char=0,
            end_char=59,
            semantic_coherence_score=0.8,
            structural_completeness=0.7,
            information_density=0.6
        )
        
        assert chunk.size == 59
        assert chunk.estimated_tokens == 14
        assert abs(chunk.quality_score - 0.7) < 0.01  # Average of the three scores
    
    def test_chunk_to_dict(self):
        """Test chunk serialization"""
        chunk = ContentChunk(
            content="Test content",
            chunk_id="test_3",
            chunk_index=2,
            start_char=10,
            end_char=22,
            chunk_type="text",
            section_title="Test Section"
        )
        
        chunk_dict = chunk.to_dict()
        
        assert chunk_dict['content'] == "Test content"
        assert chunk_dict['chunk_id'] == "test_3"
        assert chunk_dict['size'] == 12
        assert chunk_dict['chunk_type'] == "text"
        assert chunk_dict['section_title'] == "Test Section"


class TestContentChunker:
    """Test main content chunker functionality"""
    
    @pytest.fixture
    def chunker(self):
        """Create a test chunker"""
        config = ChunkingConfig(target_chunk_size=500, max_chunk_size=800)
        return ContentChunker(config)
    
    @pytest.fixture
    def sample_markdown(self):
        """Sample markdown content for testing"""
        return """
# Main Title

This is the introduction paragraph that explains the overall purpose
of this document and provides context for the reader.

## Section 1: Overview

Here's some content with important information. This section covers
the basic concepts and introduces key terminology.

### Subsection 1.1

More detailed content here with specific examples and explanations
that help illustrate the concepts discussed above.

## Section 2: Implementation

This section provides implementation details and code examples.

```python
def example_function():
    return "Hello, World!"
```

## Conclusion

This concludes our example document with a summary of key points.
"""
    
    @pytest.fixture
    def sample_code(self):
        """Sample code content for testing"""
        return """
import os
import sys
from typing import List, Dict

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.processed_count = 0
    
    def process_document(self, content: str) -> Dict:
        '''Process a document and return analysis'''
        result = {
            'length': len(content),
            'word_count': len(content.split())
        }
        self.processed_count += 1
        return result
    
    def get_stats(self):
        return {'processed': self.processed_count}

def main():
    processor = DocumentProcessor({'debug': True})
    content = "Sample document content"
    result = processor.process_document(content)
    print(f"Processed: {result}")

if __name__ == "__main__":
    main()
"""
    
    @pytest.mark.asyncio
    async def test_chunk_empty_content(self, chunker):
        """Test chunking empty content"""
        chunks = await chunker.chunk_document("")
        assert len(chunks) == 0
        
        chunks = await chunker.chunk_document("   ")
        assert len(chunks) == 0
    
    @pytest.mark.asyncio
    async def test_chunk_markdown_content(self, chunker, sample_markdown):
        """Test chunking markdown content"""
        chunks = await chunker.chunk_document(
            sample_markdown,
            document_id="test_md",
            file_path="test.md"
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, ContentChunk) for chunk in chunks)
        assert all(chunk.chunk_id.startswith("test_md") for chunk in chunks)
        assert all(chunk.size <= chunker.config.max_chunk_size for chunk in chunks)
        
        # Check that chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    @pytest.mark.asyncio
    async def test_chunk_code_content(self, chunker, sample_code):
        """Test chunking code content"""
        chunks = await chunker.chunk_document(
            sample_code,
            document_id="test_code",
            file_path="test.py"
        )
        
        assert len(chunks) > 0
        assert all(chunk.size <= chunker.config.max_chunk_size for chunk in chunks)
        
        # Check that code chunks have appropriate metadata
        code_chunks = [c for c in chunks if c.chunk_type in ['code', 'code_block']]
        assert len(code_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_different_strategies(self, sample_markdown):
        """Test different chunking strategies"""
        strategies = [
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.STRUCTURAL,
            ChunkingStrategy.HYBRID
        ]
        
        results = {}
        
        for strategy in strategies:
            config = ChunkingConfig(strategy=strategy, target_chunk_size=500)
            chunker = ContentChunker(config)
            
            chunks = await chunker.chunk_document(
                sample_markdown,
                document_id=f"test_{strategy.value}"
            )
            
            results[strategy] = {
                'chunk_count': len(chunks),
                'avg_size': sum(c.size for c in chunks) / len(chunks) if chunks else 0,
                'chunks': chunks
            }
        
        # All strategies should produce chunks
        for strategy in strategies:
            assert results[strategy]['chunk_count'] > 0
            assert results[strategy]['avg_size'] > 0
    
    @pytest.mark.asyncio
    async def test_chunk_size_limits(self, chunker):
        """Test chunk size limits are respected"""
        # Create content that should be split
        long_content = "This is a sentence. " * 100  # ~2000 characters
        
        chunks = await chunker.chunk_document(
            long_content,
            document_id="test_long"
        )
        
        assert len(chunks) > 1  # Should be split
        assert all(chunk.size <= chunker.config.max_chunk_size for chunk in chunks)
        assert all(chunk.size >= chunker.config.min_chunk_size or len(chunks) == 1 for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_metadata_preservation(self, chunker, sample_markdown):
        """Test that metadata is preserved during chunking"""
        source_metadata = {
            'source_file_path': 'test.md',
            'author': 'Test Author',
            'created_date': '2023-01-01'
        }
        
        chunks = await chunker.chunk_document(
            sample_markdown,
            document_id="test_metadata",
            source_metadata=source_metadata
        )
        
        assert len(chunks) > 0
        
        # Check that metadata is included
        for chunk in chunks:
            assert hasattr(chunk, 'metadata')
            assert chunk.metadata is not None
            assert 'chunk_id' in chunk.metadata
            assert 'processed_at' in chunk.metadata
    
    @pytest.mark.asyncio
    async def test_error_handling(self, chunker):
        """Test error handling in chunking"""
        # Test with None content
        chunks = await chunker.chunk_document(None)
        assert len(chunks) == 0
        
        # Test with very large content (should not crash)
        huge_content = "Large content. " * 10000  # ~150,000 characters
        chunks = await chunker.chunk_document(huge_content, document_id="test_huge")
        
        # Should handle gracefully
        assert len(chunks) > 0
        assert len(chunks) <= chunker.config.max_chunks_per_document
    
    def test_chunk_statistics(self, chunker):
        """Test chunk statistics collection"""
        stats = chunker.get_stats()
        
        assert 'documents_processed' in stats
        assert 'chunks_created' in stats
        assert 'total_processing_time' in stats
        assert 'strategy_usage' in stats
        
        # Initially should be zero
        assert stats['documents_processed'] == 0
        assert stats['chunks_created'] == 0


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_content_chunker(self):
        """Test chunker factory function"""
        chunker = create_content_chunker(
            strategy=ChunkingStrategy.SEMANTIC,
            target_chunk_size=1000,
            overlap_ratio=0.15
        )
        
        assert isinstance(chunker, ContentChunker)
        assert chunker.config.strategy == ChunkingStrategy.SEMANTIC
        assert chunker.config.target_chunk_size == 1000
        assert chunker.config.overlap_ratio == 0.15
    
    @pytest.mark.asyncio
    async def test_chunk_document_function(self):
        """Test convenience chunk_document function"""
        content = "This is test content for the convenience function."
        
        chunks = await chunk_document(
            content,
            document_id="test_convenience",
            file_path="test.txt"
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, ContentChunk) for chunk in chunks)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.mark.asyncio
    async def test_single_word_content(self):
        """Test with single word content"""
        chunker = create_content_chunker()
        chunks = await chunker.chunk_document("word", document_id="single")
        
        assert len(chunks) == 1
        assert chunks[0].content == "word"
    
    @pytest.mark.asyncio
    async def test_whitespace_only_content(self):
        """Test with whitespace-only content"""
        chunker = create_content_chunker()
        chunks = await chunker.chunk_document("   \n\n   ", document_id="whitespace")
        
        assert len(chunks) == 0
    
    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test with special characters"""
        content = "Content with émojis 🚀 and spéciål characters: ñáéíóú"
        chunker = create_content_chunker()
        
        chunks = await chunker.chunk_document(content, document_id="special")
        
        assert len(chunks) > 0
        assert chunks[0].content == content
    
    @pytest.mark.asyncio
    async def test_very_small_chunk_size(self):
        """Test with very small chunk size"""
        config = ChunkingConfig(target_chunk_size=50, max_chunk_size=100)
        chunker = ContentChunker(config)
        
        content = "This is a test with multiple sentences. Each sentence should be handled properly. The chunker should work even with small sizes."
        
        chunks = await chunker.chunk_document(content, document_id="small")
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(chunk.size <= 100 for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_mixed_content_types(self):
        """Test with mixed content (markdown + code + tables)"""
        mixed_content = """
# Title

Regular paragraph text.

```python
def function():
    return True
```

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |

More text after the table.
"""
        
        chunker = create_content_chunker()
        chunks = await chunker.chunk_document(mixed_content, document_id="mixed")
        
        assert len(chunks) > 0
        # Should handle different content types appropriately
        chunk_types = {chunk.chunk_type for chunk in chunks}
        assert len(chunk_types) > 1  # Should have different types


if __name__ == "__main__":
    # Run tests manually for debugging
    import asyncio
    
    async def run_manual_tests():
        """Run some tests manually for debugging"""
        chunker = create_content_chunker()
        
        sample_content = """
        # Test Document
        
        This is a test document with multiple paragraphs and sections.
        It should be chunked appropriately based on the configured strategy.
        
        ## Section 1
        
        Here's some content in section 1. This section contains important
        information that should be preserved during chunking.
        
        ## Section 2
        
        And here's section 2 with different content. The chunker should
        respect section boundaries when possible.
        """
        
        print("Testing content chunker...")
        chunks = await chunker.chunk_document(
            sample_content,
            document_id="manual_test",
            file_path="test.md"
        )
        
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Type: {chunk.chunk_type}")
            print(f"  Size: {chunk.size} chars")
            print(f"  Quality: {chunk.quality_score:.3f}")
            print(f"  Content preview: {chunk.content[:100]}...")
        
        # Test statistics
        stats = chunker.get_stats()
        print(f"\nChunker statistics: {stats}")
    
    asyncio.run(run_manual_tests())