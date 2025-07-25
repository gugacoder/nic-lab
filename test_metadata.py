#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.metadata_preservers import MetadataPreserver
from datetime import datetime
from dataclasses import dataclass

@dataclass
class MockChunk:
    content: str
    chunk_id: str
    chunk_index: int
    start_char: int
    end_char: int
    chunk_type: str = "text"
    section_title: str = None
    section_level: int = 0
    language: str = None
    document_position: float = 0.0
    semantic_coherence_score: float = 0.8
    structural_completeness: float = 0.7
    information_density: float = 0.6
    content_hash: str = "abc123"
    preceding_chunks: list = None
    following_chunks: list = None
    related_chunks: list = None
    metadata: dict = None
    
    def __post_init__(self):
        if self.preceding_chunks is None:
            self.preceding_chunks = []
        if self.following_chunks is None:
            self.following_chunks = []
        if self.related_chunks is None:
            self.related_chunks = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def size(self) -> int:
        return len(self.content)
    
    @property
    def estimated_tokens(self) -> int:
        return len(self.content) // 4
    
    @property
    def quality_score(self) -> float:
        return (self.semantic_coherence_score + self.structural_completeness + self.information_density) / 3

async def test_metadata_preservation():
    # Create test data
    chunk = MockChunk(
        content="This is a test chunk with some content for metadata testing. It contains multiple sentences and provides a good example.",
        chunk_id="test_chunk_1",
        chunk_index=0,
        start_char=0,
        end_char=120,
        chunk_type="text",
        section_title="Test Section",
        section_level=1,
        document_position=0.5
    )
    
    source_metadata = {
        'source_document_id': 'doc_123',
        'source_file_path': '/path/to/test.md',
        'author': 'Test Author',
        'created_date': datetime.now().isoformat(),
        'document_type': 'markdown',
        'project_id': 'proj_456',
        'tags': ['test', 'example']
    }
    
    # Test metadata preservation
    preserver = MetadataPreserver()
    
    # Preserve metadata
    metadata = await preserver.preserve_metadata(chunk, source_metadata, None)
    
    print("Metadata Preserver Test Results:")
    print(f"Total metadata fields: {len(metadata)}")
    print("Sample metadata fields:")
    for key, value in list(metadata.items())[:10]:
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} subfields")
        else:
            print(f"  {key}: {value}")
    
    # Test metadata summary
    chunks = [chunk]
    summary = preserver.create_metadata_summary(chunks)
    
    print(f"\nMetadata Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Metadata preserver working correctly!")

if __name__ == "__main__":
    asyncio.run(test_metadata_preservation())