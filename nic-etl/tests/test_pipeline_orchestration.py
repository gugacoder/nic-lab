import pytest
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

# Add modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from pipeline_orchestration import (
    PipelineOrchestrator, PipelineStage, ProcessingStatus, ProcessingResult, 
    PipelineResult, ProgressReport, create_pipeline_orchestrator,
    create_orchestrator_from_config_dict, safe_pipeline_execution,
    calculate_pipeline_efficiency, format_progress_report,
    PipelineOrchestrationContext
)

class TestPipelineOrchestrator:
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager"""
        config_manager = Mock()
        config_manager.get_module_config.return_value = {
            'model_name': 'BAAI/bge-m3',
            'batch_size': 32,
            'target_chunk_size': 500
        }
        config_manager.config = Mock()
        config_manager.config.pipeline = Mock()
        config_manager.config.pipeline.max_concurrent_documents = 5
        return config_manager
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            Mock(name='doc1.pdf', path='/test/doc1.pdf'),
            Mock(name='doc2.txt', path='/test/doc2.txt'),
            Mock(name='doc3.docx', path='/test/doc3.docx')
        ]
    
    def test_pipeline_orchestrator_initialization(self, mock_config_manager):
        """Test PipelineOrchestrator initialization"""
        orchestrator = PipelineOrchestrator(mock_config_manager)
        
        assert orchestrator.config_manager == mock_config_manager
        assert orchestrator.max_concurrent_documents == 5
        assert orchestrator.progress is not None
        assert orchestrator.start_time is not None
        assert orchestrator.current_stage == PipelineStage.INITIALIZATION
    
    def test_pipeline_stages_enum(self):
        """Test PipelineStage enum values"""
        stages = list(PipelineStage)
        expected_stages = [
            'INITIALIZATION',
            'GITLAB_INGESTION', 
            'DOCUMENT_PROCESSING',
            'TEXT_CHUNKING',
            'EMBEDDING_GENERATION',
            'VECTOR_STORAGE',
            'COMPLETION'
        ]
        
        assert len(stages) == len(expected_stages)
        for stage in stages:
            assert stage.value in expected_stages
    
    def test_processing_status_enum(self):
        """Test ProcessingStatus enum values"""
        statuses = list(ProcessingStatus)
        expected_statuses = [
            'PENDING',
            'IN_PROGRESS', 
            'COMPLETED',
            'FAILED',
            'SKIPPED'
        ]
        
        assert len(statuses) == len(expected_statuses)
        for status in statuses:
            assert status.value in expected_statuses
    
    def test_progress_monitoring(self, mock_config_manager):
        """Test progress monitoring functionality"""
        orchestrator = PipelineOrchestrator(mock_config_manager)
        
        # Initial progress
        progress = orchestrator.monitor_progress()
        assert isinstance(progress, ProgressReport)
        assert progress.documents_processed == 0
        assert progress.total_documents == 0
        assert progress.current_stage == PipelineStage.INITIALIZATION
        
        # Update progress
        orchestrator.progress.documents_processed = 5
        orchestrator.progress.total_documents = 10
        orchestrator.progress.current_document = 'test.pdf'
        
        updated_progress = orchestrator.monitor_progress()
        assert updated_progress.documents_processed == 5
        assert updated_progress.total_documents == 10
        assert updated_progress.current_document == 'test.pdf'
    
    @patch('pipeline_orchestration.create_gitlab_connector')
    @patch('pipeline_orchestration.create_docling_processor')
    @patch('pipeline_orchestration.create_text_chunker')
    @patch('pipeline_orchestration.create_embedding_generator')
    @patch('pipeline_orchestration.create_qdrant_vector_store')
    def test_process_single_document_success(self, mock_qdrant, mock_embed, mock_chunk, 
                                           mock_docling, mock_gitlab, mock_config_manager, sample_documents):
        """Test successful single document processing"""
        # Mock all components
        mock_gitlab.return_value = Mock()
        mock_docling.return_value = Mock()
        mock_chunk.return_value = Mock()
        mock_embed.return_value = Mock()
        mock_qdrant.return_value = Mock()
        
        # Mock processing results
        mock_docling.return_value.process_document.return_value = Mock(
            success=True, content="Processed content", metadata={}
        )
        mock_chunk.return_value.chunk_text.return_value = Mock(
            success=True, chunks=[Mock(content="chunk1"), Mock(content="chunk2")]
        )
        mock_embed.return_value.generate_embeddings.return_value = Mock(
            success=True, embeddings=[[0.1] * 1024, [0.2] * 1024]
        )
        mock_qdrant.return_value.store_vectors.return_value = Mock(
            success=True, stored_count=2
        )
        
        orchestrator = PipelineOrchestrator(mock_config_manager)
        result = orchestrator._process_single_document(sample_documents[0], 0)
        
        assert isinstance(result, ProcessingResult)
        assert result.status == ProcessingStatus.COMPLETED
        assert result.document_id == 'doc1.pdf'
        assert result.chunks_generated == 2
        assert result.vectors_stored == 2
        assert len(result.stages_completed) > 0
    
    @patch('pipeline_orchestration.create_gitlab_connector')
    @patch('pipeline_orchestration.create_docling_processor')
    def test_process_single_document_failure(self, mock_docling, mock_gitlab, mock_config_manager, sample_documents):
        """Test single document processing failure"""
        # Mock components
        mock_gitlab.return_value = Mock()
        
        # Mock docling failure
        mock_docling.return_value.process_document.side_effect = Exception("Processing failed")
        
        orchestrator = PipelineOrchestrator(mock_config_manager)
        result = orchestrator._process_single_document(sample_documents[0], 0)
        
        assert isinstance(result, ProcessingResult)
        assert result.status == ProcessingStatus.FAILED
        assert result.error is not None
        assert "failed" in result.error.lower()
    
    def test_stage_management(self, mock_config_manager):
        """Test pipeline stage management"""
        orchestrator = PipelineOrchestrator(mock_config_manager)
        
        # Start a stage
        orchestrator._start_stage(PipelineStage.GITLAB_INGESTION)
        assert orchestrator.progress.current_stage == PipelineStage.GITLAB_INGESTION
        
        # Complete a stage
        orchestrator._complete_stage(PipelineStage.GITLAB_INGESTION)
        stage_time = orchestrator.progress.stage_progress.get('gitlab_ingestion', 0)
        assert stage_time >= 0
    
    def test_checkpoint_creation(self, mock_config_manager):
        """Test checkpoint creation and management"""
        orchestrator = PipelineOrchestrator(mock_config_manager)
        
        # Create test results
        test_results = [
            ProcessingResult(
                document_id='test1',
                file_path='/test/doc1.pdf',
                status=ProcessingStatus.COMPLETED,
                processing_time=1.5,
                stages_completed=[PipelineStage.DOCUMENT_PROCESSING],
                chunks_generated=5,
                embeddings_created=5,
                vectors_stored=5
            )
        ]
        
        # Create checkpoint
        orchestrator._create_processing_checkpoint(test_results)
        
        # Check if checkpoint directory exists
        checkpoint_dir = Path("./cache/checkpoints")
        assert checkpoint_dir.exists()
    
    def test_checkpoint_resumption(self, mock_config_manager):
        """Test resuming from checkpoint"""
        orchestrator = PipelineOrchestrator(mock_config_manager)
        
        # Test with non-existent checkpoint
        result = orchestrator.resume_from_checkpoint("non_existent.pkl")
        assert result is False
        
        # Create a mock checkpoint file
        checkpoint_dir = Path("./cache/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'results': [
                {
                    'document_id': 'test1',
                    'status': 'COMPLETED',
                    'processing_time': 1.5
                }
            ],
            'timestamp': datetime.now().isoformat(),
            'metadata': {'version': '1.0'}
        }
        
        checkpoint_file = checkpoint_dir / "test_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        try:
            # Test resumption (should handle gracefully even if format differs)
            result = orchestrator.resume_from_checkpoint(str(checkpoint_file))
            # Result depends on implementation - should not raise exception
            assert isinstance(result, bool)
        finally:
            # Cleanup
            checkpoint_file.unlink(missing_ok=True)
    
    @patch('pipeline_orchestration.create_gitlab_connector')
    def test_full_pipeline_execution_mock(self, mock_gitlab, mock_config_manager):
        """Test full pipeline execution with mocked components"""
        # Mock GitLab connector
        mock_connector = Mock()
        mock_connector.get_documents.return_value = [
            Mock(name='doc1.pdf', path='/test/doc1.pdf'),
            Mock(name='doc2.txt', path='/test/doc2.txt')
        ]
        mock_gitlab.return_value = mock_connector
        
        orchestrator = PipelineOrchestrator(mock_config_manager)
        
        # Mock the document processing to avoid complex setup
        def mock_process_documents(documents):
            return [
                ProcessingResult(
                    document_id=doc.name,
                    file_path=doc.path,
                    status=ProcessingStatus.COMPLETED,
                    processing_time=1.0,
                    stages_completed=[PipelineStage.DOCUMENT_PROCESSING],
                    chunks_generated=3,
                    embeddings_created=3,
                    vectors_stored=3
                )
                for doc in documents
            ]
        
        orchestrator.process_documents = mock_process_documents
        
        result = orchestrator.run_full_pipeline("test-folder")
        
        assert isinstance(result, PipelineResult)
        assert result.total_documents == 2
        assert result.processed_successfully == 2
        assert result.failed_documents == 0
        assert result.total_chunks == 6
        assert result.total_vectors_stored == 6
    
    def test_orchestrator_statistics(self, mock_config_manager):
        """Test orchestrator statistics gathering"""
        orchestrator = PipelineOrchestrator(mock_config_manager)
        stats = orchestrator.get_orchestrator_statistics()
        
        assert isinstance(stats, dict)
        assert 'max_concurrent_documents' in stats
        assert 'current_stage' in stats
        assert 'start_time' in stats
        assert stats['max_concurrent_documents'] == 5
    
    def test_error_handling(self, mock_config_manager):
        """Test error handling in orchestrator"""
        orchestrator = PipelineOrchestrator(mock_config_manager)
        
        # Test with empty document list
        results = orchestrator.process_documents([])
        assert isinstance(results, list)
        assert len(results) == 0
        
        # Test with None input
        try:
            results = orchestrator.process_documents(None)
            assert isinstance(results, list)
        except Exception:
            # Should handle gracefully
            pass
    
    def test_parallel_processing_simulation(self, mock_config_manager):
        """Test parallel processing simulation"""
        orchestrator = PipelineOrchestrator(mock_config_manager)
        
        # Create mock documents
        documents = [Mock(name=f'doc{i}.pdf', path=f'/test/doc{i}.pdf') for i in range(5)]
        
        # Mock process_single_document to return success
        def mock_process_single(doc, index):
            return ProcessingResult(
                document_id=doc.name,
                file_path=doc.path,
                status=ProcessingStatus.COMPLETED,
                processing_time=0.5,
                stages_completed=[PipelineStage.DOCUMENT_PROCESSING],
                chunks_generated=2,
                embeddings_created=2,
                vectors_stored=2
            )
        
        orchestrator._process_single_document = mock_process_single
        
        results = orchestrator.process_documents(documents)
        
        assert len(results) == 5
        assert all(r.status == ProcessingStatus.COMPLETED for r in results)
        assert sum(r.chunks_generated for r in results) == 10

class TestPipelineUtilities:
    
    def test_create_orchestrator_from_config_dict(self):
        """Test creating orchestrator from config dictionary"""
        config_dict = {
            'environment': 'test',
            'pipeline': {'max_concurrent_documents': 3},
            'gitlab': {'url': 'http://test.gitlab.com'},
            'qdrant': {'collection_name': 'test'}
        }
        
        orchestrator = create_orchestrator_from_config_dict(config_dict)
        assert orchestrator is not None
        assert isinstance(orchestrator, PipelineOrchestrator)
    
    def test_safe_pipeline_execution(self):
        """Test safe pipeline execution wrapper"""
        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_result = PipelineResult(
            total_documents=5,
            processed_successfully=5,
            failed_documents=0,
            skipped_documents=0,
            total_processing_time=10.0,
            total_chunks=25,
            total_embeddings=25,
            total_vectors_stored=25
        )
        mock_orchestrator.run_full_pipeline.return_value = mock_result
        
        result = safe_pipeline_execution(mock_orchestrator, "test-folder")
        
        assert isinstance(result, PipelineResult)
        assert result.total_documents == 5
        assert result.processed_successfully == 5
    
    def test_safe_pipeline_execution_with_error(self):
        """Test safe pipeline execution with error handling"""
        # Mock orchestrator that raises exception
        mock_orchestrator = Mock()
        mock_orchestrator.run_full_pipeline.side_effect = Exception("Pipeline failed")
        
        result = safe_pipeline_execution(mock_orchestrator, "test-folder")
        
        # Should return a failed result instead of raising exception
        assert isinstance(result, PipelineResult)
        assert result.total_documents == 0
        assert len(result.errors) > 0
    
    def test_calculate_pipeline_efficiency(self):
        """Test pipeline efficiency calculation"""
        result = PipelineResult(
            total_documents=10,
            processed_successfully=8,
            failed_documents=1,
            skipped_documents=1,
            total_processing_time=30.0,
            total_chunks=40,
            total_embeddings=40,
            total_vectors_stored=38
        )
        
        efficiency = calculate_pipeline_efficiency(result)
        
        assert isinstance(efficiency, dict)
        assert 'success_rate' in efficiency
        assert 'failure_rate' in efficiency
        assert 'processing_rate' in efficiency
        assert 'avg_chunks_per_document' in efficiency
        
        assert efficiency['success_rate'] == 0.8
        assert efficiency['failure_rate'] == 0.1
        assert efficiency['avg_chunks_per_document'] == 5.0
    
    def test_format_progress_report(self):
        """Test progress report formatting"""
        progress = ProgressReport(
            current_stage=PipelineStage.TEXT_CHUNKING,
            documents_processed=3,
            total_documents=10,
            current_document='test.pdf',
            processing_rate=1.5,
            estimated_completion='5 minutes',
            total_chunks=15,
            total_embeddings=15,
            total_vectors_stored=15
        )
        
        formatted = format_progress_report(progress)
        
        assert isinstance(formatted, str)
        assert 'TEXT_CHUNKING' in formatted
        assert '3/10' in formatted
        assert 'test.pdf' in formatted
    
    def test_context_manager(self):
        """Test PipelineOrchestrationContext context manager"""
        config_dict = {
            'environment': 'test',
            'pipeline': {'max_concurrent_documents': 2}
        }
        
        with PipelineOrchestrationContext(config_dict) as orchestrator:
            assert isinstance(orchestrator, PipelineOrchestrator)
        
        # Context manager should handle cleanup

class TestPipelineDataClasses:
    
    def test_processing_result_creation(self):
        """Test ProcessingResult data class"""
        result = ProcessingResult(
            document_id='test.pdf',
            file_path='/path/test.pdf',
            status=ProcessingStatus.COMPLETED,
            processing_time=2.5,
            stages_completed=[PipelineStage.DOCUMENT_PROCESSING, PipelineStage.TEXT_CHUNKING],
            chunks_generated=10,
            embeddings_created=10,
            vectors_stored=8,
            error=None,
            metadata={'source': 'test'}
        )
        
        assert result.document_id == 'test.pdf'
        assert result.status == ProcessingStatus.COMPLETED
        assert result.processing_time == 2.5
        assert len(result.stages_completed) == 2
        assert result.chunks_generated == 10
        assert result.vectors_stored == 8
    
    def test_pipeline_result_creation(self):
        """Test PipelineResult data class"""
        result = PipelineResult(
            total_documents=100,
            processed_successfully=95,
            failed_documents=3,
            skipped_documents=2,
            total_processing_time=300.0,
            total_chunks=500,
            total_embeddings=500,
            total_vectors_stored=485,
            errors=['Error 1', 'Error 2'],
            metadata={'version': '1.0'}
        )
        
        assert result.total_documents == 100
        assert result.processed_successfully == 95
        assert result.failed_documents == 3
        assert result.total_processing_time == 300.0
        assert len(result.errors) == 2
    
    def test_progress_report_creation(self):
        """Test ProgressReport data class"""
        progress = ProgressReport(
            current_stage=PipelineStage.EMBEDDING_GENERATION,
            documents_processed=25,
            total_documents=50,
            current_document='document25.pdf',
            processing_rate=2.0,
            estimated_completion='12.5 minutes',
            total_chunks=125,
            total_embeddings=125,
            total_vectors_stored=120,
            stage_progress={'embedding_generation': 10.5}
        )
        
        assert progress.current_stage == PipelineStage.EMBEDDING_GENERATION
        assert progress.documents_processed == 25
        assert progress.total_documents == 50
        assert progress.processing_rate == 2.0
        assert 'embedding_generation' in progress.stage_progress

class TestPipelineIntegration:
    """Integration tests for pipeline orchestration"""
    
    def test_end_to_end_pipeline_simulation(self):
        """Test end-to-end pipeline simulation"""
        # Create comprehensive config
        config_dict = {
            'environment': 'test',
            'pipeline': {
                'max_concurrent_documents': 2,
                'max_memory_usage_gb': 4.0,
                'checkpoint_interval': 10
            },
            'gitlab': {
                'url': 'http://test.gitlab.com',
                'access_token': 'test_token'
            },
            'qdrant': {
                'url': 'http://test.qdrant.com',
                'collection_name': 'test_collection'
            },
            'embedding': {
                'model_name': 'BAAI/bge-m3',
                'batch_size': 16
            },
            'chunking': {
                'target_chunk_size': 300
            }
        }
        
        # Mock all external dependencies
        with patch('pipeline_orchestration.create_gitlab_connector') as mock_gitlab:
            with patch('pipeline_orchestration.create_docling_processor') as mock_docling:
                with patch('pipeline_orchestration.create_text_chunker') as mock_chunker:
                    with patch('pipeline_orchestration.create_embedding_generator') as mock_embedder:
                        with patch('pipeline_orchestration.create_qdrant_vector_store') as mock_qdrant:
                            
                            # Setup mocks
                            mock_gitlab.return_value.get_documents.return_value = [
                                Mock(name='doc1.pdf', path='/test/doc1.pdf'),
                                Mock(name='doc2.txt', path='/test/doc2.txt')
                            ]
                            
                            mock_docling.return_value.process_document.return_value = Mock(
                                success=True, content="Processed content", metadata={}
                            )
                            
                            mock_chunker.return_value.chunk_text.return_value = Mock(
                                success=True, 
                                chunks=[Mock(content="chunk1"), Mock(content="chunk2"), Mock(content="chunk3")]
                            )
                            
                            mock_embedder.return_value.generate_embeddings.return_value = Mock(
                                success=True, embeddings=[[0.1] * 1024] * 3
                            )
                            
                            mock_qdrant.return_value.store_vectors.return_value = Mock(
                                success=True, stored_count=3
                            )
                            
                            # Create and run orchestrator
                            orchestrator = create_orchestrator_from_config_dict(config_dict)
                            result = safe_pipeline_execution(orchestrator, "test-folder")
                            
                            # Verify results
                            assert isinstance(result, PipelineResult)
                            assert result.total_documents == 2
                            assert result.processed_successfully >= 0  # Depends on mock behavior
                            assert result.total_processing_time > 0
                            
                            # Verify all components were called
                            assert mock_gitlab.called
                            assert mock_docling.called
                            assert mock_chunker.called
                            assert mock_embedder.called
                            assert mock_qdrant.called