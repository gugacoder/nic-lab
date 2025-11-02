# Pipeline Orchestration - PRP

## ROLE
**Pipeline Architecture Engineer with Workflow Orchestration expertise**

Specialized in ETL pipeline design, workflow orchestration, and system integration. Responsible for implementing the main coordination module that orchestrates all pipeline stages, manages dependencies, handles failures gracefully, and provides comprehensive monitoring and progress tracking for the entire NIC ETL system.

## OBJECTIVE
**Production-Ready Pipeline Orchestration Framework**

Deliver a production-ready Python module that:
- Orchestrates the complete ETL pipeline from GitLab ingestion to Qdrant storage
- Coordinates all pipeline modules with proper dependency management
- Implements robust error handling, retry mechanisms, and failure recovery
- Provides real-time progress tracking and performance monitoring
- Supports parallel processing and resource management
- Enables pipeline configuration and customization through the Jupyter notebook interface
- Implements checkpointing and resume capabilities for large document sets

## MOTIVATION
**Unified Pipeline Control and Reliability**

Effective pipeline orchestration ensures reliable, efficient, and observable ETL processing. By implementing comprehensive workflow coordination with error handling, monitoring, and recovery capabilities, this module enables the Jupyter notebook to provide a seamless user experience while maintaining system reliability and performance for production workloads.

## CONTEXT
**Jupyter Notebook-Centric Orchestration Architecture**

- **Primary Interface**: Jupyter notebook serves as the main user interface and orchestrator
- **Module Coordination**: Coordinates GitLab, Docling, Chunking, Embedding, and Qdrant modules
- **Configuration Management**: Integrates with centralized configuration system
- **Error Handling**: Comprehensive error management with graceful degradation
- **Monitoring**: Real-time progress tracking and performance metrics
- **Scalability**: Support for large document sets and concurrent processing

## IMPLEMENTATION BLUEPRINT
**Comprehensive Pipeline Orchestration Module**

### Architecture Overview
```python
# Module Structure: modules/pipeline_orchestration.py
class PipelineOrchestrator:
    """Main pipeline orchestration with comprehensive workflow management"""
    
    def __init__(self, config_manager: ConfigurationManager)
    def run_full_pipeline(self, target_folder: str = "30-Aprovados") -> PipelineResult
    def process_documents(self, documents: List[FileMetadata]) -> List[ProcessingResult]
    def monitor_progress(self) -> ProgressReport
    def handle_pipeline_errors(self, error: Exception, context: Dict[str, Any]) -> ErrorResponse
    def create_checkpoint(self, stage: str, data: Dict[str, Any]) -> bool
    def resume_from_checkpoint(self, checkpoint_file: str) -> bool
```

### Code Structure
**File Organization**: `modules/pipeline_orchestration.py`
```python
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import pickle
from pathlib import Path
import time
from enum import Enum

# Import all pipeline modules
from modules.configuration_management import ConfigurationManager
from modules.gitlab_integration import GitLabConnector
from modules.docling_processing import DoclingProcessor
from modules.text_chunking import TextChunker
from modules.embedding_generation import EmbeddingGenerator
from modules.qdrant_integration import QdrantVectorStore
from modules.metadata_management import NICSchemaManager

class PipelineStage(Enum):
    """Pipeline execution stages"""
    INITIALIZATION = "initialization"
    GITLAB_INGESTION = "gitlab_ingestion"
    DOCUMENT_PROCESSING = "document_processing"
    TEXT_CHUNKING = "text_chunking"
    EMBEDDING_GENERATION = "embedding_generation"
    VECTOR_STORAGE = "vector_storage"
    FINALIZATION = "finalization"

class ProcessingStatus(Enum):
    """Document processing status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ProcessingResult:
    """Individual document processing result"""
    document_id: str
    file_path: str
    status: ProcessingStatus
    processing_time: float
    stages_completed: List[PipelineStage]
    chunks_generated: int = 0
    embeddings_created: int = 0
    vectors_stored: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineResult:
    """Complete pipeline execution result"""
    total_documents: int
    processed_successfully: int
    failed_documents: int
    skipped_documents: int
    total_processing_time: float
    total_chunks: int
    total_embeddings: int
    total_vectors_stored: int
    stage_timings: Dict[str, float] = field(default_factory=dict)
    document_results: List[ProcessingResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProgressReport:
    """Real-time progress tracking"""
    current_stage: PipelineStage
    documents_processed: int
    total_documents: int
    current_document: Optional[str] = None
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    processing_rate: float = 0.0
    errors_count: int = 0
    warnings_count: int = 0
    stage_progress: Dict[str, float] = field(default_factory=dict)

class PipelineOrchestrator:
    """Production-ready pipeline orchestration with comprehensive workflow management"""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize all pipeline modules
        self._initialize_modules()
        
        # Progress tracking
        self.progress = ProgressReport(
            current_stage=PipelineStage.INITIALIZATION,
            documents_processed=0,
            total_documents=0
        )
        
        # Performance tracking
        self.start_time = None
        self.stage_start_times = {}
        
        # Error handling
        self.max_retries = 3
        self.retry_delay = 5.0
        
        # Checkpointing
        self.checkpoint_interval = 10  # Save checkpoint every N documents
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def _initialize_modules(self):
        """Initialize all pipeline modules with configuration"""
        
        try:
            module_configs = self.config_manager.create_module_configs()
            
            # Initialize GitLab connector
            self.gitlab_connector = GitLabConnector(
                gitlab_url=module_configs['gitlab']['url'],
                access_token=module_configs['gitlab']['access_token'],
                project_path=module_configs['gitlab']['project_path']
            )
            
            # Initialize Docling processor
            self.docling_processor = DoclingProcessor(module_configs['docling'])
            
            # Initialize text chunker
            self.text_chunker = TextChunker(module_configs['chunking'])
            
            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator(module_configs['embedding'])
            
            # Initialize Qdrant vector store
            self.qdrant_store = QdrantVectorStore(module_configs['qdrant'])
            
            # Initialize metadata manager
            self.metadata_manager = NICSchemaManager()
            
            self.logger.info("All pipeline modules initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Module initialization failed: {e}")
            raise
    
    def run_full_pipeline(self, target_folder: str = "30-Aprovados", 
                         checkpoint_file: Optional[str] = None) -> PipelineResult:
        """Execute complete ETL pipeline with comprehensive orchestration"""
        
        self.start_time = datetime.utcnow()
        pipeline_result = PipelineResult(
            total_documents=0,
            processed_successfully=0,
            failed_documents=0,
            skipped_documents=0,
            total_processing_time=0.0,
            total_chunks=0,
            total_embeddings=0,
            total_vectors_stored=0
        )
        
        try:
            # Resume from checkpoint if provided
            if checkpoint_file and self.resume_from_checkpoint(checkpoint_file):
                self.logger.info(f"Resumed pipeline from checkpoint: {checkpoint_file}")
            
            # Stage 1: GitLab Ingestion
            self._start_stage(PipelineStage.GITLAB_INGESTION)
            documents = self._execute_gitlab_ingestion(target_folder)
            pipeline_result.total_documents = len(documents)
            self.progress.total_documents = len(documents)
            self._complete_stage(PipelineStage.GITLAB_INGESTION)
            
            if not documents:
                self.logger.warning("No documents found for processing")
                return pipeline_result
            
            # Stage 2-6: Process all documents
            processing_results = self.process_documents(documents)
            pipeline_result.document_results = processing_results
            
            # Aggregate results
            for result in processing_results:
                if result.status == ProcessingStatus.COMPLETED:
                    pipeline_result.processed_successfully += 1
                    pipeline_result.total_chunks += result.chunks_generated
                    pipeline_result.total_embeddings += result.embeddings_created
                    pipeline_result.total_vectors_stored += result.vectors_stored
                elif result.status == ProcessingStatus.FAILED:
                    pipeline_result.failed_documents += 1
                    pipeline_result.errors.extend(result.errors)
                elif result.status == ProcessingStatus.SKIPPED:
                    pipeline_result.skipped_documents += 1
            
            # Stage 7: Finalization
            self._start_stage(PipelineStage.FINALIZATION)
            self._execute_finalization(pipeline_result)
            self._complete_stage(PipelineStage.FINALIZATION)
            
            # Calculate total processing time
            pipeline_result.total_processing_time = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Generate performance metrics
            pipeline_result.performance_metrics = self._generate_performance_metrics(pipeline_result)
            
            self.logger.info(
                f"Pipeline completed: {pipeline_result.processed_successfully}/{pipeline_result.total_documents} "
                f"documents processed successfully in {pipeline_result.total_processing_time:.2f}s"
            )
            
            return pipeline_result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            pipeline_result.errors.append(str(e))
            pipeline_result.total_processing_time = (datetime.utcnow() - self.start_time).total_seconds()
            return pipeline_result
    
    def process_documents(self, documents: List[Any]) -> List[ProcessingResult]:
        """Process documents through all pipeline stages with parallel execution"""
        
        processing_results = []
        max_workers = self.config_manager.config.pipeline.max_concurrent_documents
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all document processing tasks
            future_to_doc = {
                executor.submit(self._process_single_document, doc, idx): doc 
                for idx, doc in enumerate(documents)
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    result = future.result()
                    processing_results.append(result)
                    
                    # Update progress
                    self.progress.documents_processed += 1
                    self._update_progress_metrics()
                    
                    # Create checkpoint periodically
                    if len(processing_results) % self.checkpoint_interval == 0:
                        self._create_processing_checkpoint(processing_results)
                    
                except Exception as e:
                    self.logger.error(f"Document processing failed for {doc.path}: {e}")
                    failed_result = ProcessingResult(
                        document_id=getattr(doc, 'name', 'unknown'),
                        file_path=getattr(doc, 'path', 'unknown'),
                        status=ProcessingStatus.FAILED,
                        processing_time=0.0,
                        stages_completed=[],
                        errors=[str(e)]
                    )
                    processing_results.append(failed_result)
        
        return processing_results
    
    def _process_single_document(self, document: Any, doc_index: int) -> ProcessingResult:
        """Process single document through all pipeline stages"""
        
        doc_start_time = time.time()
        result = ProcessingResult(
            document_id=document.name,
            file_path=document.path,
            status=ProcessingStatus.IN_PROGRESS,
            processing_time=0.0,
            stages_completed=[]
        )
        
        try:
            # Stage 1: Download document content
            self.logger.info(f"Processing document {doc_index + 1}: {document.path}")
            content = self.gitlab_connector.download_file(document.path)
            result.stages_completed.append(PipelineStage.GITLAB_INGESTION)
            
            # Stage 2: Document processing with Docling
            self._start_stage(PipelineStage.DOCUMENT_PROCESSING)
            processed_doc = self.docling_processor.process_document(document.path, content, document.__dict__)
            result.stages_completed.append(PipelineStage.DOCUMENT_PROCESSING)
            
            # Stage 3: Text chunking
            self._start_stage(PipelineStage.TEXT_CHUNKING)
            chunks = self.text_chunker.chunk_document(
                processed_doc.structured_content, 
                processed_doc.processing_metadata
            )
            result.chunks_generated = len(chunks)
            result.stages_completed.append(PipelineStage.TEXT_CHUNKING)
            
            # Stage 4: Embedding generation
            self._start_stage(PipelineStage.EMBEDDING_GENERATION)
            embeddings = self.embedding_generator.generate_embeddings(chunks)
            result.embeddings_created = len(embeddings)
            result.stages_completed.append(PipelineStage.EMBEDDING_GENERATION)
            
            # Stage 5: Vector storage
            self._start_stage(PipelineStage.VECTOR_STORAGE)
            storage_result = self.qdrant_store.insert_vectors(embeddings)
            result.vectors_stored = storage_result.successful_insertions
            result.stages_completed.append(PipelineStage.VECTOR_STORAGE)
            
            # Collect any warnings or errors
            if storage_result.errors:
                result.warnings.extend(storage_result.errors)
            
            # Update final status
            result.status = ProcessingStatus.COMPLETED
            result.processing_time = time.time() - doc_start_time
            
            self.logger.info(
                f"Document processed successfully: {document.path} "
                f"({result.chunks_generated} chunks, {result.vectors_stored} vectors stored)"
            )
            
        except Exception as e:
            result.status = ProcessingStatus.FAILED
            result.processing_time = time.time() - doc_start_time
            result.errors.append(str(e))
            self.logger.error(f"Failed to process document {document.path}: {e}")
        
        return result
    
    def _execute_gitlab_ingestion(self, target_folder: str) -> List[Any]:
        """Execute GitLab ingestion stage"""
        
        try:
            # Authenticate with GitLab
            if not self.gitlab_connector.authenticate():
                raise RuntimeError("GitLab authentication failed")
            
            # List files in target folder
            documents = self.gitlab_connector.list_files(
                branch=self.config_manager.config.gitlab.branch,
                folder_path=target_folder,
                extensions=self.config_manager.config.gitlab.supported_extensions
            )
            
            self.logger.info(f"Found {len(documents)} documents in GitLab folder: {target_folder}")
            return documents
            
        except Exception as e:
            self.logger.error(f"GitLab ingestion failed: {e}")
            raise
    
    def _execute_finalization(self, pipeline_result: PipelineResult):
        """Execute pipeline finalization stage"""
        
        try:
            # Perform Qdrant collection optimization
            if self.config_manager.config.qdrant.optimize_collection:
                collection_name = self.config_manager.config.qdrant.collection_name
                self.qdrant_store.optimize_collection(collection_name)
                self.logger.info(f"Optimized Qdrant collection: {collection_name}")
            
            # Generate final summary report
            self._generate_pipeline_summary(pipeline_result)
            
        except Exception as e:
            self.logger.warning(f"Finalization stage warning: {e}")
    
    def _start_stage(self, stage: PipelineStage):
        """Start pipeline stage with timing"""
        self.progress.current_stage = stage
        self.stage_start_times[stage.value] = time.time()
        self.logger.debug(f"Started stage: {stage.value}")
    
    def _complete_stage(self, stage: PipelineStage):
        """Complete pipeline stage with timing"""
        if stage.value in self.stage_start_times:
            stage_time = time.time() - self.stage_start_times[stage.value]
            self.progress.stage_progress[stage.value] = stage_time
            self.logger.debug(f"Completed stage: {stage.value} in {stage_time:.2f}s")
    
    def _update_progress_metrics(self):
        """Update progress tracking metrics"""
        if self.start_time:
            self.progress.elapsed_time = (datetime.utcnow() - self.start_time).total_seconds()
            
            if self.progress.documents_processed > 0:
                self.progress.processing_rate = self.progress.documents_processed / self.progress.elapsed_time
                
                if self.progress.processing_rate > 0:
                    remaining_docs = self.progress.total_documents - self.progress.documents_processed
                    self.progress.estimated_remaining = remaining_docs / self.progress.processing_rate
    
    def monitor_progress(self) -> ProgressReport:
        """Get current pipeline progress"""
        self._update_progress_metrics()
        return self.progress
    
    def _create_processing_checkpoint(self, results: List[ProcessingResult]):
        """Create checkpoint for pipeline resume capability"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.checkpoint_dir / f"pipeline_checkpoint_{timestamp}.pkl"
            
            checkpoint_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'progress': self.progress,
                'results': results,
                'config_hash': self.config_manager.config_hash
            }
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"Created checkpoint: {checkpoint_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create checkpoint: {e}")
    
    def resume_from_checkpoint(self, checkpoint_file: str) -> bool:
        """Resume pipeline from checkpoint"""
        try:
            checkpoint_path = Path(checkpoint_file)
            if not checkpoint_path.exists():
                self.logger.error(f"Checkpoint file not found: {checkpoint_file}")
                return False
            
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint compatibility
            if checkpoint_data['config_hash'] != self.config_manager.config_hash:
                self.logger.warning("Checkpoint configuration mismatch, may cause issues")
            
            # Restore progress
            self.progress = checkpoint_data['progress']
            
            self.logger.info(f"Resumed from checkpoint: {checkpoint_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint: {e}")
            return False
    
    def _generate_performance_metrics(self, result: PipelineResult) -> Dict[str, Any]:
        """Generate comprehensive performance metrics"""
        
        metrics = {
            'documents_per_second': result.processed_successfully / result.total_processing_time if result.total_processing_time > 0 else 0,
            'chunks_per_document': result.total_chunks / result.processed_successfully if result.processed_successfully > 0 else 0,
            'embeddings_per_second': result.total_embeddings / result.total_processing_time if result.total_processing_time > 0 else 0,
            'success_rate': result.processed_successfully / result.total_documents if result.total_documents > 0 else 0,
            'average_processing_time': sum(r.processing_time for r in result.document_results) / len(result.document_results) if result.document_results else 0,
            'stage_timings': self.progress.stage_progress.copy()
        }
        
        return metrics
    
    def _generate_pipeline_summary(self, result: PipelineResult):
        """Generate comprehensive pipeline execution summary"""
        
        summary = {
            'execution_timestamp': datetime.utcnow().isoformat(),
            'pipeline_version': self.config_manager.config.pipeline.processing_pipeline_version,
            'configuration': {
                'environment': self.config_manager.environment,
                'gitlab_branch': self.config_manager.config.gitlab.branch,
                'chunk_size': self.config_manager.config.chunking.target_chunk_size,
                'embedding_model': self.config_manager.config.embedding.model_name,
                'qdrant_collection': self.config_manager.config.qdrant.collection_name
            },
            'results': {
                'total_documents': result.total_documents,
                'processed_successfully': result.processed_successfully,
                'failed_documents': result.failed_documents,
                'total_chunks': result.total_chunks,
                'total_vectors_stored': result.total_vectors_stored,
                'processing_time': result.total_processing_time
            },
            'performance_metrics': result.performance_metrics
        }
        
        # Save summary to file
        summary_file = Path(f"pipeline_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline summary saved: {summary_file}")

def create_pipeline_orchestrator(config_manager: ConfigurationManager) -> PipelineOrchestrator:
    """Factory function for pipeline orchestrator creation"""
    return PipelineOrchestrator(config_manager)
```

## VALIDATION LOOP
**Comprehensive Testing Strategy**

### Unit Testing
```python
# tests/test_pipeline_orchestration.py
import pytest
from unittest.mock import Mock, patch
from modules.pipeline_orchestration import PipelineOrchestrator, PipelineStage, ProcessingStatus
from modules.configuration_management import ConfigurationManager

class TestPipelineOrchestrator:
    
    @pytest.fixture
    def mock_config_manager(self):
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.create_module_configs.return_value = {
            'gitlab': {'url': 'test', 'access_token': 'test', 'project_path': 'test'},
            'docling': {},
            'chunking': {},
            'embedding': {},
            'qdrant': {},
            'metadata': {}
        }
        return config_manager
    
    @patch('modules.pipeline_orchestration.GitLabConnector')
    @patch('modules.pipeline_orchestration.DoclingProcessor')
    @patch('modules.pipeline_orchestration.TextChunker')
    @patch('modules.pipeline_orchestration.EmbeddingGenerator')
    @patch('modules.pipeline_orchestration.QdrantVectorStore')
    def test_module_initialization(self, mock_qdrant, mock_embedding, mock_chunker, 
                                 mock_docling, mock_gitlab, mock_config_manager):
        """Test successful module initialization"""
        
        orchestrator = PipelineOrchestrator(mock_config_manager)
        
        assert orchestrator.gitlab_connector is not None
        assert orchestrator.docling_processor is not None
        assert orchestrator.text_chunker is not None
        assert orchestrator.embedding_generator is not None
        assert orchestrator.qdrant_store is not None
    
    def test_progress_tracking(self, mock_config_manager):
        """Test progress tracking functionality"""
        
        with patch.multiple('modules.pipeline_orchestration',
                          GitLabConnector=Mock(), DoclingProcessor=Mock(),
                          TextChunker=Mock(), EmbeddingGenerator=Mock(),
                          QdrantVectorStore=Mock(), NICSchemaManager=Mock()):
            
            orchestrator = PipelineOrchestrator(mock_config_manager)
            
            # Initial progress
            progress = orchestrator.monitor_progress()
            assert progress.documents_processed == 0
            assert progress.current_stage == PipelineStage.INITIALIZATION
            
            # Update progress
            orchestrator.progress.documents_processed = 5
            orchestrator.progress.total_documents = 10
            
            updated_progress = orchestrator.monitor_progress()
            assert updated_progress.documents_processed == 5
            assert updated_progress.total_documents == 10
```

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
- **Resource Limits**: Implement memory and processing time limits
- **Error Information**: Sanitize error messages to prevent information leakage
- **Checkpoint Security**: Secure checkpoint files against unauthorized access
- **Module Isolation**: Ensure module failures don't compromise system security

### Performance Optimization
- **Parallel Processing**: Optimal concurrency based on system resources
- **Memory Management**: Efficient memory usage across pipeline stages
- **Caching Strategy**: Cache expensive operations where appropriate
- **Resource Monitoring**: Track and optimize system resource usage

### Maintenance Requirements
- **Pipeline Monitoring**: Comprehensive logging and monitoring
- **Performance Analytics**: Track pipeline performance over time
- **Error Analysis**: Analyze and categorize pipeline errors
- **Capacity Planning**: Monitor and plan for scaling requirements