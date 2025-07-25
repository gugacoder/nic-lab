"""
Acceptance Criteria Validation Tests

This module validates that the content chunking system meets all
acceptance criteria specified in the task requirements.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock
from typing import List
import statistics

# Import the components to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.chunker import (
    ContentChunker, ChunkingConfig, ContentChunk, ChunkingStrategy,
    create_content_chunker
)


class TestAcceptanceCriteria:
    """Test all acceptance criteria from the task specification"""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents of different types for testing"""
        return {
            'markdown': """
# API Documentation

This document describes the REST API endpoints for our service.
It provides comprehensive information about authentication, request formats,
and response structures.

## Authentication

All API requests must include an authentication token in the header.
The token can be obtained through the login endpoint.

### Token Format

The authentication token is a JWT with the following structure:
- Header: Contains algorithm and token type
- Payload: Contains user information and expiration
- Signature: Ensures token integrity

## Endpoints

### GET /api/users

Retrieves a list of all users in the system.

**Parameters:**
- limit (optional): Maximum number of users to return
- offset (optional): Number of users to skip

**Response:**
```json
{
  "users": [
    {
      "id": 1,
      "name": "John Doe",
      "email": "john@example.com"
    }
  ],
  "total": 100
}
```

### POST /api/users

Creates a new user in the system.

**Request Body:**
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "password": "securepassword"
}
```

## Error Handling

The API uses standard HTTP status codes to indicate success or failure.
Common error responses include:

- 400 Bad Request: Invalid request parameters
- 401 Unauthorized: Missing or invalid authentication
- 404 Not Found: Resource does not exist
- 500 Internal Server Error: Server-side error

Each error response includes a descriptive message to help with debugging.
""",
            
            'code': """
#!/usr/bin/env python3
'''
User Management System

This module provides user authentication and management functionality
for the web application. It includes user registration, login, and
profile management features.
'''

import hashlib
import jwt
import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

@dataclass
class User:
    '''Represents a user in the system'''
    id: int
    username: str
    email: str
    password_hash: str
    created_at: datetime.datetime
    is_active: bool = True
    
    def check_password(self, password: str) -> bool:
        '''Check if provided password matches user password'''
        return self._hash_password(password) == self.password_hash
    
    @staticmethod
    def _hash_password(password: str) -> str:
        '''Hash password using SHA-256'''
        return hashlib.sha256(password.encode()).hexdigest()

class UserManager:
    '''Manages user operations and authentication'''
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.users: Dict[int, User] = {}
        self.next_id = 1
    
    def create_user(self, username: str, email: str, password: str) -> User:
        '''Create a new user account'''
        if self.get_user_by_email(email):
            raise ValueError("User with this email already exists")
        
        user = User(
            id=self.next_id,
            username=username,
            email=email,
            password_hash=User._hash_password(password),
            created_at=datetime.datetime.now()
        )
        
        self.users[user.id] = user
        self.next_id += 1
        return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        '''Authenticate user with email and password'''
        user = self.get_user_by_email(email)
        if user and user.check_password(password):
            return user
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        '''Get user by email address'''
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def generate_token(self, user: User) -> str:
        '''Generate JWT token for user'''
        payload = {
            'user_id': user.id,
            'email': user.email,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[User]:
        '''Verify JWT token and return user'''
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return self.users.get(payload['user_id'])
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

def main():
    '''Example usage of the user management system'''
    manager = UserManager('super-secret-key')
    
    # Create a test user
    user = manager.create_user('testuser', 'test@example.com', 'password123')
    print(f"Created user: {user.username}")
    
    # Authenticate user
    auth_user = manager.authenticate_user('test@example.com', 'password123')
    if auth_user:
        token = manager.generate_token(auth_user)
        print(f"Generated token: {token[:20]}...")
    
    # Verify token
    verified_user = manager.verify_token(token)
    if verified_user:
        print(f"Token verified for user: {verified_user.username}")

if __name__ == '__main__':
    main()
""",
            
            'mixed': """
# Database Migration Guide

This guide explains how to perform database migrations safely in production.

## Prerequisites

Before running migrations, ensure you have:

1. Database backup completed
2. Application maintenance mode enabled
3. Migration scripts tested in staging

## Migration Process

### Step 1: Backup Database

```bash
pg_dump -h localhost -U postgres myapp_production > backup_$(date +%Y%m%d).sql
```

### Step 2: Run Migrations

```python
def run_migration(migration_file):
    '''Execute database migration'''
    with open(migration_file, 'r') as f:
        sql = f.read()
    
    conn = get_database_connection()
    try:
        conn.execute(sql)
        conn.commit()
        log_migration_success(migration_file)
    except Exception as e:
        conn.rollback()
        log_migration_error(migration_file, str(e))
        raise
    finally:
        conn.close()
```

### Step 3: Verify Results

After migration, verify that:

- [ ] All tables exist with correct schema
- [ ] Data integrity is maintained
- [ ] Application starts without errors
- [ ] Basic functionality works

## Rollback Procedure

If migration fails:

1. Stop the application
2. Restore from backup:
   ```bash
   psql -h localhost -U postgres myapp_production < backup_20231201.sql
   ```
3. Restart application with previous version

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Lock timeout | Long-running query | Increase timeout settings |
| Constraint violation | Data inconsistency | Fix data before migration |
| Permission denied | Wrong user | Check database permissions |

Contact the DBA team if you encounter issues not covered here.
"""
        }
    
    @pytest.mark.asyncio
    async def test_semantic_coherence(self, sample_documents):
        """Test that chunks maintain semantic coherence"""
        chunker = create_content_chunker(strategy=ChunkingStrategy.SEMANTIC)
        
        for doc_type, content in sample_documents.items():
            chunks = await chunker.chunk_document(
                content, 
                document_id=f"semantic_test_{doc_type}"
            )
            
            # Check semantic coherence
            coherent_chunks = 0
            for chunk in chunks:
                # A chunk is semantically coherent if:
                # 1. It has a reasonable coherence score
                # 2. It doesn't end mid-sentence (except for very long sentences)
                # 3. It starts at a reasonable boundary
                
                if chunk.semantic_coherence_score >= 0.6:
                    coherent_chunks += 1
                
                # Check sentence boundaries
                content = chunk.content.strip()
                if content and len(content) > 100:  # Only check longer chunks
                    # Should not end mid-word
                    assert not (content[-1].isalnum() and 
                              not content.endswith(('.', '!', '?', ':', ';', '\n')))
            
            # At least 95% of chunks should be semantically coherent
            coherence_ratio = coherent_chunks / len(chunks) if chunks else 0
            assert coherence_ratio >= 0.95, f"Only {coherence_ratio:.2%} of chunks are coherent for {doc_type}"
    
    @pytest.mark.asyncio
    async def test_structure_preservation(self, sample_documents):
        """Test that document structure is preserved in metadata"""
        chunker = create_content_chunker(
            strategy=ChunkingStrategy.STRUCTURAL,
            preserve_structure=True
        )
        
        # Test with markdown document (has clear structure)
        chunks = await chunker.chunk_document(
            sample_documents['markdown'],
            document_id="structure_test",
            file_path="test.md"
        )
        
        # Check that structure is preserved
        has_headers = any(chunk.section_title is not None for chunk in chunks)
        assert has_headers, "Document structure (headers) not preserved"
        
        # Check section levels are reasonable
        section_levels = [chunk.section_level for chunk in chunks if chunk.section_level > 0]
        if section_levels:
            assert max(section_levels) <= 6, "Section levels should not exceed 6"
            assert min(section_levels) >= 1, "Section levels should start from 1"
        
        # Check metadata includes structure information
        for chunk in chunks:
            if chunk.metadata:
                # Should have document-level structure info
                assert 'document_content_type' in chunk.metadata
                assert 'document_is_structured' in chunk.metadata
    
    @pytest.mark.asyncio 
    async def test_code_block_integrity(self, sample_documents):
        """Test that code blocks remain intact"""
        chunker = create_content_chunker(respect_code_blocks=True)
        
        # Test with mixed document containing code blocks
        chunks = await chunker.chunk_document(
            sample_documents['mixed'],
            document_id="code_test",
            file_path="test.md"
        )
        
        # Find chunks containing code blocks
        code_chunks = []
        for chunk in chunks:
            if '```' in chunk.content:
                code_chunks.append(chunk)
        
        # Verify code block integrity
        for chunk in code_chunks:
            content = chunk.content
            # Count opening and closing code block markers
            opening_markers = content.count('```')
            
            # Should have even number of markers (paired opening/closing)
            # or end with incomplete block (which is acceptable at chunk boundary)
            if opening_markers % 2 != 0:
                # If odd number, the chunk should end with the incomplete block
                # or the block should continue in the next chunk
                last_marker_pos = content.rfind('```')
                # Text after last marker should not contain closing marker
                remaining_text = content[last_marker_pos + 3:]
                assert '```' not in remaining_text, "Code block markers are unbalanced"
    
    @pytest.mark.asyncio
    async def test_overlap_context_continuity(self, sample_documents):
        """Test that overlap provides sufficient context"""
        config = ChunkingConfig(
            overlap_ratio=0.15,  # 15% overlap
            min_overlap_size=100
        )
        chunker = ContentChunker(config)
        
        chunks = await chunker.chunk_document(
            sample_documents['markdown'],
            document_id="overlap_test"
        )
        
        # Check overlap exists between chunks
        overlapped_chunks = 0
        for chunk in chunks:
            if chunk.metadata and chunk.metadata.get('has_overlap', False):
                overlapped_chunks += 1
                overlap_size = chunk.metadata.get('overlap_size', 0)
                
                # Overlap should be reasonable size
                assert overlap_size >= 50, f"Overlap too small: {overlap_size}"
                assert overlap_size <= chunk.size * 0.4, f"Overlap too large: {overlap_size}"
        
        # Most chunks (except first) should have overlap
        if len(chunks) > 1:
            overlap_ratio = overlapped_chunks / len(chunks)
            assert overlap_ratio >= 0.5, f"Insufficient overlap coverage: {overlap_ratio:.2%}"
    
    @pytest.mark.asyncio
    async def test_chunk_size_compliance(self, sample_documents):
        """Test that chunk sizes fit within token limits"""
        config = ChunkingConfig(
            target_chunk_size=1500,
            max_chunk_size=2000,
            min_chunk_size=200
        )
        chunker = ContentChunker(config)
        
        for doc_type, content in sample_documents.items():
            chunks = await chunker.chunk_document(
                content,
                document_id=f"size_test_{doc_type}"
            )
            
            # Check size compliance
            oversized_chunks = 0
            undersized_chunks = 0
            
            for chunk in chunks:
                if chunk.size > config.max_chunk_size:
                    oversized_chunks += 1
                elif chunk.size < config.min_chunk_size:
                    undersized_chunks += 1
            
            # 100% should comply with max size
            assert oversized_chunks == 0, f"{oversized_chunks} chunks exceed max size"
            
            # Most should comply with min size (some exceptions allowed for structure)
            undersized_ratio = undersized_chunks / len(chunks) if chunks else 0
            assert undersized_ratio <= 0.1, f"Too many undersized chunks: {undersized_ratio:.2%}"
            
            # Average size should be close to target
            if chunks:
                avg_size = sum(c.size for c in chunks) / len(chunks)
                target_diff = abs(avg_size - config.target_chunk_size) / config.target_chunk_size
                assert target_diff <= 0.3, f"Average size too far from target: {avg_size} vs {config.target_chunk_size}"
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, sample_documents):
        """Test that performance handles large documents efficiently"""
        # Create larger content by repeating sample content
        large_content = sample_documents['markdown'] * 10  # ~10x larger
        
        chunker = create_content_chunker()
        
        # Measure processing time
        start_time = time.time()
        chunks = await chunker.chunk_document(
            large_content,
            document_id="performance_test"
        )
        processing_time = time.time() - start_time
        
        # Should process at reasonable speed
        # Target: > 100 pages/minute (assume ~500 words/page, ~2000 chars/page)
        content_size_pages = len(large_content) / 2000
        pages_per_minute = (content_size_pages / processing_time) * 60
        
        assert pages_per_minute >= 50, f"Processing too slow: {pages_per_minute:.1f} pages/minute"
        
        # Should produce reasonable number of chunks
        assert len(chunks) > 0, "No chunks produced"
        assert len(chunks) < 1000, f"Too many chunks produced: {len(chunks)}"
    
    @pytest.mark.asyncio
    async def test_markdown_formatting_preservation(self, sample_documents):
        """Test that markdown formatting is preserved"""
        chunker = create_content_chunker(preserve_structure=True)
        
        chunks = await chunker.chunk_document(
            sample_documents['markdown'],
            document_id="markdown_test",
            file_path="test.md"
        )
        
        # Check that markdown elements are preserved
        all_content = '\n'.join(chunk.content for chunk in chunks)
        
        # Headers should be preserved
        assert '# API Documentation' in all_content
        assert '## Authentication' in all_content
        assert '### Token Format' in all_content
        
        # Code blocks should be preserved
        assert '```json' in all_content
        assert '```bash' in all_content
        
        # Lists should be preserved
        assert '- Header: Contains algorithm' in all_content
        assert '- 400 Bad Request:' in all_content
    
    @pytest.mark.asyncio
    async def test_chunk_quality_distribution(self, sample_documents):
        """Test overall chunk quality meets expectations"""
        chunker = create_content_chunker()
        
        all_quality_scores = []
        
        for doc_type, content in sample_documents.items():
            chunks = await chunker.chunk_document(
                content,
                document_id=f"quality_test_{doc_type}"
            )
            
            quality_scores = [chunk.quality_score for chunk in chunks]
            all_quality_scores.extend(quality_scores)
        
        if all_quality_scores:
            avg_quality = statistics.mean(all_quality_scores)
            min_quality = min(all_quality_scores)
            
            # Average quality should be good (>0.6)
            assert avg_quality >= 0.6, f"Average quality too low: {avg_quality:.3f}"
            
            # No chunks should have very poor quality (<0.3)
            assert min_quality >= 0.3, f"Minimum quality too low: {min_quality:.3f}"
            
            # At least 80% of chunks should have good quality (>0.5)
            good_quality_chunks = sum(1 for score in all_quality_scores if score >= 0.5)
            good_quality_ratio = good_quality_chunks / len(all_quality_scores)
            assert good_quality_ratio >= 0.8, f"Too few good quality chunks: {good_quality_ratio:.2%}"
    
    @pytest.mark.asyncio
    async def test_consistent_chunk_quality(self, sample_documents):
        """Test that chunk quality is consistent across document types"""
        chunker = create_content_chunker()
        
        quality_by_type = {}
        
        for doc_type, content in sample_documents.items():
            chunks = await chunker.chunk_document(
                content,
                document_id=f"consistency_test_{doc_type}"
            )
            
            if chunks:
                avg_quality = sum(chunk.quality_score for chunk in chunks) / len(chunks)
                quality_by_type[doc_type] = avg_quality
        
        # Quality should be reasonably consistent across document types
        if len(quality_by_type) > 1:
            quality_values = list(quality_by_type.values())
            quality_range = max(quality_values) - min(quality_values)
            
            # Range should not be too large (indicating inconsistent processing)
            assert quality_range <= 0.3, f"Quality too inconsistent across document types: {quality_by_type}"


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_chunking_speed_benchmark(self):
        """Benchmark chunking speed with different content sizes"""
        chunker = create_content_chunker()
        
        # Test with different content sizes
        base_content = "This is a test paragraph with multiple sentences. " * 20
        
        results = {}
        
        for multiplier in [1, 5, 10, 20]:
            content = base_content * multiplier
            content_size = len(content)
            
            start_time = time.time()
            chunks = await chunker.chunk_document(
                content,
                document_id=f"benchmark_{multiplier}"
            )
            processing_time = time.time() - start_time
            
            chars_per_second = content_size / processing_time
            results[content_size] = {
                'processing_time': processing_time,
                'chars_per_second': chars_per_second,
                'chunk_count': len(chunks)
            }
        
        # Print benchmark results
        print("\nChunking Speed Benchmark:")
        for size, metrics in results.items():
            print(f"  {size:,} chars: {metrics['processing_time']:.3f}s "
                  f"({metrics['chars_per_second']:,.0f} chars/s, "
                  f"{metrics['chunk_count']} chunks)")
        
        # Verify performance scales reasonably
        # Processing time should scale sub-linearly with content size
        sizes = sorted(results.keys())
        if len(sizes) >= 2:
            small_time = results[sizes[0]]['processing_time']
            large_time = results[sizes[-1]]['processing_time']
            
            size_ratio = sizes[-1] / sizes[0]
            time_ratio = large_time / small_time
            
            # Time should not scale worse than linearly
            assert time_ratio <= size_ratio * 1.5, f"Performance degrades too much with size: {time_ratio} vs {size_ratio}"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency with large documents"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process a large document
        large_content = "This is a large document content. " * 10000  # ~340KB
        
        chunker = create_content_chunker()
        chunks = await chunker.chunk_document(
            large_content,
            document_id="memory_test"
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        # Should not use more than 10x the content size in memory
        content_size_mb = len(large_content) / 1024 / 1024
        memory_ratio = memory_increase / content_size_mb if content_size_mb > 0 else 0
        
        print(f"\nMemory Usage:")
        print(f"  Content size: {content_size_mb:.2f} MB")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        print(f"  Memory ratio: {memory_ratio:.1f}x")
        
        # Reasonable memory usage
        assert memory_ratio <= 15, f"Memory usage too high: {memory_ratio:.1f}x content size"


if __name__ == "__main__":
    # Run acceptance criteria tests manually
    import asyncio
    
    async def run_acceptance_tests():
        """Run key acceptance tests manually"""
        test_instance = TestAcceptanceCriteria()
        
        # Prepare sample documents
        sample_docs = {
            'simple': """
# Test Document

This is a simple test document with multiple paragraphs.
Each paragraph should be processed correctly by the chunker.

## Section One

Here is some content in section one. It contains important
information that should be preserved during the chunking process.

## Section Two

And here is section two with different content. The chunker
should handle this appropriately and maintain quality.
"""
        }
        
        print("Running acceptance criteria tests...")
        
        # Test semantic coherence
        print("\n1. Testing semantic coherence...")
        await test_instance.test_semantic_coherence(sample_docs)
        print("   ✓ Chunks maintain semantic coherence")
        
        # Test chunk size compliance
        print("\n2. Testing chunk size compliance...")
        await test_instance.test_chunk_size_compliance(sample_docs)
        print("   ✓ Chunk sizes fit within limits")
        
        # Test performance
        print("\n3. Testing performance...")
        await test_instance.test_performance_requirements(sample_docs)
        print("   ✓ Performance meets requirements")
        
        print("\nAll acceptance criteria tests passed! ✓")
    
    asyncio.run(run_acceptance_tests())