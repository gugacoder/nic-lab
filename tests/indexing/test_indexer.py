"""
Tests for the Search Indexing System

This module provides comprehensive tests for the indexing functionality,
including schema, text processing, metadata extraction, and search operations.
"""

import os
import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any

from src.indexing.schema import IndexSchema, ContentType
from src.indexing.text_processor import TextProcessor, ProcessedText
from src.indexing.metadata_extractor import MetadataExtractor, FileMetadata
from src.indexing.indexer import SearchIndexer, IndexConfig, IndexStatus
from src.indexing.storage.index_store import IndexStore, StorageConfig, IndexMetadata
from src.indexing.scheduler import IndexScheduler, ScheduleConfig, ScheduleType


class TestIndexSchema:
    """Test index schema functionality"""
    
    def test_schema_creation(self):
        """Test that schema can be created successfully"""
        schema = IndexSchema.get_schema()
        assert schema is not None
        
        # Check required fields exist
        required_fields = [
            'doc_id', 'project_id', 'project_name', 'file_path',
            'content', 'content_type', 'indexed_at'
        ]
        
        for field in required_fields:
            assert field in schema.names()
    
    def test_content_type_detection(self):
        """Test content type detection for various file types"""
        test_cases = [
            ('README.md', ContentType.MARKDOWN),
            ('script.py', ContentType.CODE),
            ('config.yaml', ContentType.YAML),
            ('data.json', ContentType.JSON),
            ('notes.txt', ContentType.TEXT),
            ('unknown.xyz', ContentType.OTHER)
        ]
        
        for file_path, expected_type in test_cases:
            detected_type = IndexSchema.determine_content_type(file_path)
            assert detected_type == expected_type
    
    def test_document_preparation(self):
        """Test document preparation for indexing"""
        doc_data = {
            'doc_id': 'test/file.py',
            'project_id': 1,
            'project_name': 'Test Project',
            'file_path': 'src/test.py',
            'content': 'def hello(): pass',
            'title': 'Test File',
            'author_name': 'Test Author'
        }
        
        prepared = IndexSchema.prepare_document(doc_data)
        
        assert prepared['doc_id'] == 'test/file.py'
        assert prepared['content_type'] == ContentType.CODE.value
        assert prepared['file_extension'] == '.py'
        assert 'indexed_at' in prepared
        assert prepared['code_content'] == prepared['content']  # Code files get code_content


class TestTextProcessor:
    """Test text processing functionality"""
    
    @pytest.fixture
    def processor(self):
        return TextProcessor()
    
    def test_markdown_processing(self, processor):
        """Test processing of markdown content"""
        markdown_content = """
# Main Title

This is a **bold** paragraph with [a link](https://example.com).

## Section 1

Here's some `inline code` and a list:
- Item 1
- Item 2

```python
def example():
    return "Hello"
```
        """
        
        result = processor.process(markdown_content, 'test.md')
        
        assert result.title == "Main Title"
        assert "Section 1" in result.headings
        assert result.word_count > 10
        assert len(result.code_blocks) == 1
        assert result.code_blocks[0]['language'] == 'python'
        assert "[CODE_BLOCK]" in result.content  # Code blocks replaced
        assert "**bold**" not in result.content  # Markdown formatting removed
    
    def test_code_processing(self, processor):
        """Test processing of code files"""
        code_content = """
def calculate_sum(a, b):
    # TODO: Add validation
    return a + b

# FIXME: Handle edge cases
class Calculator:
    pass
        """
        
        result = processor.process(code_content, 'calculator.py')
        
        assert result.language == 'python'
        assert result.word_count > 5
        
        # Test metadata extraction
        metadata = processor.extract_metadata(code_content, 'calculator.py')
        assert metadata['has_todo'] is True
        assert metadata['has_fixme'] is True
    
    def test_text_chunking(self, processor):
        """Test text chunking functionality"""
        long_text = " ".join(["This is sentence number {}.".format(i) for i in range(50)])
        
        chunks = processor.extract_chunks(long_text, chunk_size=200, overlap=50)
        
        assert len(chunks) > 1
        # Check overlap exists
        for i in range(len(chunks) - 1):
            assert any(word in chunks[i+1] for word in chunks[i].split()[-5:])


class TestMetadataExtractor:
    """Test metadata extraction functionality"""
    
    @pytest.fixture
    def extractor(self):
        # Mock GitLab client
        class MockGitLabClient:
            async def get_project(self, project_id):
                return {'name': f'Project-{project_id}', 'visibility': 'private'}
            
            async def get_file_info(self, project_id, file_path, branch):
                return {'size': 1000, 'commit_id': 'abc123'}
            
            async def get_commit(self, project_id, commit_sha):
                return {
                    'author_name': 'Test Author',
                    'author_email': 'test@example.com',
                    'committed_date': '2024-01-01T00:00:00Z'
                }
            
            async def get_file_blame(self, project_id, file_path, branch):
                return []
        
        return MetadataExtractor(MockGitLabClient())
    
    def test_tag_extraction(self, extractor):
        """Test tag extraction from content"""
        content = """
        # Documentation
        Tags: [python, testing, documentation]
        #development #coding
        
        This is a test file.
        """
        
        tags = extractor._extract_tags(content, 'docs/test.md')
        
        assert 'python' in tags
        assert 'testing' in tags
        assert 'documentation' in tags
        assert 'development' in tags
        assert 'docs' in tags  # From file path
        assert 'md' in tags  # From file extension
    
    def test_mime_type_detection(self, extractor):
        """Test MIME type detection"""
        test_cases = [
            ('test.py', 'text/x-python'),
            ('script.js', 'application/javascript'),
            ('config.json', 'application/json'),
            ('README.md', 'text/markdown'),
            ('unknown.xyz', 'text/plain')
        ]
        
        for file_path, expected_mime in test_cases:
            mime_type = extractor._determine_mime_type(file_path)
            assert mime_type == expected_mime
    
    @pytest.mark.asyncio
    async def test_file_metadata_extraction(self, extractor):
        """Test complete file metadata extraction"""
        metadata = await extractor.extract_file_metadata(
            project_id=1,
            file_path='src/example.py',
            branch='main',
            content='# Example Python File\n# Tags: [example, test]'
        )
        
        assert metadata.project_id == 1
        assert metadata.project_name == 'Project-1'
        assert metadata.file_path == 'src/example.py'
        assert metadata.file_extension == '.py'
        assert metadata.commit_sha == 'abc123'
        assert metadata.author_name == 'Test Author'
        assert 'example' in metadata.tags


class TestSearchIndexer:
    """Test main indexer functionality"""
    
    @pytest.fixture
    async def indexer(self, tmp_path):
        """Create indexer with temporary directory"""
        config = IndexConfig(
            index_dir=str(tmp_path / "test_index"),
            batch_size=10,
            commit_interval=50
        )
        
        # Mock GitLab client
        class MockGitLabClient:
            async def get_all_projects(self):
                return [{'id': 1, 'name': 'Test Project'}]
            
            async def get_project(self, project_id):
                return {
                    'id': project_id,
                    'name': 'Test Project',
                    'default_branch': 'main'
                }
            
            async def get_repository_tree(self, project_id, ref, recursive):
                return [
                    {'type': 'blob', 'path': 'README.md', 'id': '1'},
                    {'type': 'blob', 'path': 'src/main.py', 'id': '2'},
                    {'type': 'tree', 'path': 'docs', 'id': '3'}
                ]
            
            async def get_file_content(self, project_id, file_path, branch):
                contents = {
                    'README.md': '# Test Project\n\nThis is a test project for indexing.',
                    'src/main.py': 'def main():\n    print("Hello, World!")'
                }
                return contents.get(file_path, '')
        
        indexer = SearchIndexer(config, MockGitLabClient())
        yield indexer
        
        # Cleanup
        if os.path.exists(config.index_dir):
            shutil.rmtree(config.index_dir)
    
    @pytest.mark.asyncio
    async def test_index_building(self, indexer):
        """Test building a new index"""
        stats = await indexer.build_index(force_rebuild=True)
        
        assert stats.status == IndexStatus.IDLE
        assert stats.total_documents > 0
        assert len(stats.indexed_projects) > 0
        assert stats.indexing_time_seconds > 0
    
    @pytest.mark.asyncio
    async def test_incremental_update(self, indexer):
        """Test incremental index updates"""
        # Build initial index
        await indexer.build_index(force_rebuild=True)
        
        # Run incremental update
        stats = await indexer.update_incremental()
        
        assert stats.status == IndexStatus.IDLE
        # Should be fast since no changes
        assert stats.indexing_time_seconds < 5.0
    
    def test_search_functionality(self, indexer):
        """Test search operations"""
        # Note: This requires an index to be built first
        results = indexer.search("test", limit=10)
        
        assert results['success'] is True
        assert 'results' in results
        assert 'total' in results
    
    @pytest.mark.asyncio
    async def test_index_optimization(self, indexer):
        """Test index optimization"""
        # Build index first
        await indexer.build_index(force_rebuild=True)
        
        # Optimize
        await indexer.optimize_index()
        
        # Check status
        stats = await indexer.get_stats()
        assert stats.status == IndexStatus.IDLE


class TestIndexStore:
    """Test index storage functionality"""
    
    @pytest.fixture
    async def store(self, tmp_path):
        """Create store with temporary directory"""
        config = StorageConfig(
            base_dir=str(tmp_path / "test_indexes"),
            backup_dir=str(tmp_path / "test_backups"),
            max_backups=3
        )
        return IndexStore(config)
    
    @pytest.mark.asyncio
    async def test_metadata_persistence(self, store):
        """Test saving and loading metadata"""
        metadata = IndexMetadata(
            total_documents=100,
            indexed_projects=[1, 2, 3],
            index_size_bytes=1024 * 1024
        )
        
        # Save
        await store.save_metadata(metadata)
        
        # Load
        loaded = await store.load_metadata()
        
        assert loaded is not None
        assert loaded.total_documents == 100
        assert loaded.indexed_projects == [1, 2, 3]
    
    @pytest.mark.asyncio
    async def test_state_persistence(self, store):
        """Test saving and loading state"""
        state = {
            'doc1': {'project_id': 1, 'file_path': 'test.py'},
            'doc2': {'project_id': 2, 'file_path': 'readme.md'}
        }
        
        # Save
        await store.save_state(state)
        
        # Load
        loaded = await store.load_state()
        
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded['doc1']['file_path'] == 'test.py'
    
    @pytest.mark.asyncio
    async def test_backup_creation(self, store):
        """Test index backup functionality"""
        # Create dummy index
        index_dir = os.path.join(store.config.base_dir, "main")
        os.makedirs(index_dir, exist_ok=True)
        
        with open(os.path.join(index_dir, "test.dat"), "w") as f:
            f.write("test data")
        
        # Create backup
        backup_path = await store.backup_index()
        
        assert backup_path is not None
        assert os.path.exists(backup_path)
    
    @pytest.mark.asyncio
    async def test_index_validation(self, store):
        """Test index validation"""
        is_valid, issues = await store.validate_index()
        
        # Should be invalid since no index exists
        assert is_valid is False
        assert len(issues) > 0


class TestScheduler:
    """Test scheduler functionality"""
    
    @pytest.fixture
    async def scheduler(self, tmp_path):
        """Create scheduler with test configuration"""
        indexer = SearchIndexer(IndexConfig(index_dir=str(tmp_path / "test_index")))
        store = IndexStore(StorageConfig(base_dir=str(tmp_path / "test_indexes")))
        
        config = ScheduleConfig(
            incremental_interval=10,  # 10 seconds for testing
            backup_interval=20,
            enable_full_rebuild=False,
            enable_optimization=False
        )
        
        return IndexScheduler(indexer, store, config)
    
    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization"""
        assert scheduler is not None
        assert len(scheduler._tasks) > 0
        
        # Check that enabled tasks are scheduled
        assert ScheduleType.INCREMENTAL_UPDATE in scheduler._tasks
        assert ScheduleType.BACKUP in scheduler._tasks
    
    def test_scheduler_status(self, scheduler):
        """Test getting scheduler status"""
        status = scheduler.get_status()
        
        assert 'running' in status
        assert 'tasks' in status
        assert len(status['tasks']) > 0
    
    @pytest.mark.asyncio
    async def test_manual_task_trigger(self, scheduler):
        """Test manually triggering a task"""
        triggered = await scheduler.trigger_task(ScheduleType.BACKUP)
        
        assert triggered is True
        
        # Try triggering non-existent task
        triggered = await scheduler.trigger_task(ScheduleType.OPTIMIZATION)
        assert triggered is False  # Should be disabled in test config


# Performance and integration tests
class TestPerformance:
    """Performance tests for the indexing system"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_indexing_performance(self, tmp_path):
        """Test indexing performance with multiple files"""
        # This test is marked for optional execution
        config = IndexConfig(
            index_dir=str(tmp_path / "perf_index"),
            batch_size=100
        )
        
        # Create mock client with many files
        class MockGitLabClient:
            async def get_all_projects(self):
                return [{'id': i, 'name': f'Project-{i}'} for i in range(1, 6)]
            
            async def get_project(self, project_id):
                return {
                    'id': project_id,
                    'name': f'Project-{project_id}',
                    'default_branch': 'main'
                }
            
            async def get_repository_tree(self, project_id, ref, recursive):
                # Generate 200 files per project
                files = []
                for i in range(200):
                    files.append({
                        'type': 'blob',
                        'path': f'src/file_{i}.py',
                        'id': f'{project_id}_{i}'
                    })
                return files
            
            async def get_file_content(self, project_id, file_path, branch):
                return f"# File: {file_path}\n\ndef function_{project_id}():\n    pass"
        
        indexer = SearchIndexer(config, MockGitLabClient())
        
        start_time = asyncio.get_event_loop().time()
        stats = await indexer.build_index(force_rebuild=True)
        duration = asyncio.get_event_loop().time() - start_time
        
        # Should index 1000 files (5 projects * 200 files) in less than 5 minutes
        assert stats.total_documents == 1000
        assert duration < 300  # 5 minutes
        
        print(f"Indexed {stats.total_documents} files in {duration:.2f} seconds")
        print(f"Rate: {stats.total_documents / duration:.1f} files/second")


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])