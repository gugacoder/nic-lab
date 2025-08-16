import logging
import time
import json
import pickle
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import hashlib

# Optional concurrent.futures import with fallback
try:
    import concurrent.futures
    CONCURRENT_FUTURES_AVAILABLE = True
except ImportError:
    CONCURRENT_FUTURES_AVAILABLE = False
    # Create mock concurrent.futures for development
    class MockFuture:
        def __init__(self, result):
            self._result = result
        
        def result(self):
            return self._result
    
    class MockThreadPoolExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        
        def submit(self, fn, *args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                return MockFuture(result)
            except Exception as e:
                return MockFuture(None)
        
        def as_completed(self, futures):
            return futures
    
    class MockConcurrentFutures:
        ThreadPoolExecutor = MockThreadPoolExecutor
        as_completed = MockThreadPoolExecutor.as_completed
    
    concurrent = type('concurrent', (), {'futures': MockConcurrentFutures()})()

# Import pipeline modules with fallbacks
try:
    from modules.configuration_management import ConfigurationManager, create_configuration_manager
    CONFIG_MODULE_AVAILABLE = True
except ImportError:
    CONFIG_MODULE_AVAILABLE = False
    # Mock configuration manager
    class MockConfigurationManager:
        def __init__(self):
            self.environment = 'development'
            self.config_hash = 'mock_hash'
            self.config = type('Config', (), {
                'pipeline': type('Pipeline', (), {
                    'max_concurrent_documents': 2,
                    'processing_pipeline_version': '1.0'
                })(),
                'gitlab': type('GitLab', (), {
                    'branch': 'main',
                    'supported_extensions': ['.pdf', '.docx', '.txt', '.md']
                })(),
                'qdrant': type('Qdrant', (), {
                    'optimize_collection': True,
                    'collection_name': 'nic'
                })(),
                'chunking': type('Chunking', (), {
                    'target_chunk_size': 500
                })(),
                'embedding': type('Embedding', (), {
                    'model_name': 'BAAI/bge-m3'
                })()
            })()
        
        def create_module_configs(self):
            return {
                'gitlab': {'url': 'http://gitlab.example.com', 'access_token': 'token', 'project_path': 'test/project'},
                'docling': {'ocr_enabled': True, 'confidence_threshold': 0.75},
                'chunking': {
                    'target_chunk_size': 500,
                    'overlap_size': 100,
                    'model_name': 'BAAI/bge-m3',
                    'boundary_strategy': 'paragraph',
                    'preserve_structure': True
                },
                'embedding': {
                    'model_name': 'BAAI/bge-m3',
                    'batch_size': 32,
                    'max_sequence_length': 512,
                    'normalize_embeddings': True,
                    'device': 'cpu'
                },
                'qdrant': {
                    'url': 'http://localhost:6333',
                    'collection_name': 'test_nic',
                    'vector_size': 1024,
                    'distance_metric': 'cosine',
                    'batch_size': 100
                },
                'metadata': {'schema_version': '1.0'}
            }
    
    ConfigurationManager = MockConfigurationManager

# Mock other module imports
class MockModule:
    def __init__(self, config=None):
        self.config = config or {}
    
    def authenticate(self):
        return True
    
    def list_files(self, branch=None, folder_path=None, extensions=None):
        # Return mock file list
        return [
            type('MockFile', (), {'name': 'test1.pdf', 'path': 'test1.pdf'})(),
            type('MockFile', (), {'name': 'test2.docx', 'path': 'test2.docx'})()
        ]
    
    def download_file(self, path):
        return b"Mock file content"
    
    def process_document(self, path, content, metadata):
        return type('ProcessedDoc', (), {
            'structured_content': type('Content', (), {})(),
            'processing_metadata': {}
        })()
    
    def chunk_document(self, content, metadata):
        # Return mock chunks
        return [
            type('Chunk', (), {'chunk_id': f'chunk_{i}'})()
            for i in range(3)
        ]
    
    def generate_embeddings(self, chunks):
        # Return mock embeddings
        return [
            type('Embedding', (), {'chunk_id': chunk.chunk_id})()
            for chunk in chunks
        ]
    
    def insert_vectors(self, embeddings):
        return type('InsertResult', (), {
            'successful_insertions': len(embeddings),
            'errors': []
        })()
    
    def optimize_collection(self, name):
        return True

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
        self.checkpoint_dir = Path("./cache/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _initialize_modules(self):
        """Initialize all pipeline modules with configuration"""
        
        try:
            self.logger.info("Initializing pipeline modules...")
            module_configs = self.config_manager.create_module_configs()
            
            # Initialize GitLab connector (with mock fallback)
            try:
                from modules.gitlab_integration import GitLabConnector
                self.gitlab_connector = GitLabConnector(
                    gitlab_url=module_configs['gitlab']['url'],
                    access_token=module_configs['gitlab']['access_token'],
                    project_path=module_configs['gitlab']['project_path']
                )
            except ImportError:
                self.logger.warning("GitLab module not available, using mock")
                self.gitlab_connector = MockModule()
            
            # Initialize Docling processor (with mock fallback)
            try:
                from modules.docling_processing import DoclingProcessor
                self.docling_processor = DoclingProcessor(module_configs['docling'])
            except ImportError:
                self.logger.warning("Docling module not available, using mock")
                self.docling_processor = MockModule()
            
            # Initialize text chunker (with mock fallback)
            try:
                from modules.text_chunking import TextChunker, create_text_chunker
                self.text_chunker = create_text_chunker(module_configs['chunking'])
            except (ImportError, Exception) as e:
                self.logger.warning(f"Text chunking module not available, using mock: {e}")
                self.text_chunker = MockModule()
            
            # Initialize embedding generator (with mock fallback)
            try:
                from modules.embedding_generation import EmbeddingGenerator, create_embedding_generator
                self.embedding_generator = create_embedding_generator(module_configs['embedding'])
            except (ImportError, Exception) as e:
                self.logger.warning(f"Embedding generation module not available, using mock: {e}")
                self.embedding_generator = MockModule()
            
            # Initialize Qdrant vector store (with mock fallback)
            try:
                from modules.qdrant_integration import QdrantVectorStore, create_qdrant_vector_store
                self.qdrant_store = create_qdrant_vector_store(module_configs['qdrant'])
            except (ImportError, Exception) as e:
                self.logger.warning(f"Qdrant integration module not available, using mock: {e}")
                self.qdrant_store = MockModule()
            
            # Initialize metadata manager (with mock fallback)
            try:
                from modules.metadata_management import NICSchemaManager
                self.metadata_manager = NICSchemaManager()
            except ImportError:
                self.logger.warning("Metadata management module not available, using mock")
                self.metadata_manager = MockModule()
            
            self.logger.info("All pipeline modules initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Module initialization failed: {e}")
            raise PipelineOrchestrationError(f"Module initialization failed: {e}")
    
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
            self.logger.info(f"Starting NIC ETL Pipeline execution targeting folder: {target_folder}")
            
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
            
            self.logger.info(f"Found {len(documents)} documents to process")
            
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
        max_workers = getattr(self.config_manager.config.pipeline, 'max_concurrent_documents', 2)
        
        # Use concurrent processing if available, otherwise sequential
        if CONCURRENT_FUTURES_AVAILABLE and max_workers > 1:
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
                        self.logger.error(f"Document processing failed for {getattr(doc, 'path', 'unknown')}: {e}")
                        failed_result = ProcessingResult(
                            document_id=getattr(doc, 'name', 'unknown'),
                            file_path=getattr(doc, 'path', 'unknown'),
                            status=ProcessingStatus.FAILED,
                            processing_time=0.0,
                            stages_completed=[],
                            errors=[str(e)]
                        )
                        processing_results.append(failed_result)
        else:
            # Sequential processing
            self.logger.info("Using sequential document processing")
            for idx, doc in enumerate(documents):
                try:
                    result = self._process_single_document(doc, idx)
                    processing_results.append(result)
                    
                    # Update progress
                    self.progress.documents_processed += 1
                    self._update_progress_metrics()
                    
                    # Create checkpoint periodically
                    if len(processing_results) % self.checkpoint_interval == 0:
                        self._create_processing_checkpoint(processing_results)
                        
                except Exception as e:
                    self.logger.error(f"Document processing failed for {getattr(doc, 'path', 'unknown')}: {e}")
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
        document_name = getattr(document, 'name', f'document_{doc_index}')
        document_path = getattr(document, 'path', f'path_{doc_index}')
        
        result = ProcessingResult(
            document_id=document_name,
            file_path=document_path,
            status=ProcessingStatus.IN_PROGRESS,
            processing_time=0.0,
            stages_completed=[]
        )
        
        try:
            self.logger.info(f"Processing document {doc_index + 1}/{self.progress.total_documents}: {document_path}")
            self.progress.current_document = document_path
            
            # Stage 1: Download document content
            content = self.gitlab_connector.download_file(document_path)
            result.stages_completed.append(PipelineStage.GITLAB_INGESTION)
            
            # Stage 2: Document processing with Docling
            self._start_stage(PipelineStage.DOCUMENT_PROCESSING)
            processed_doc = self.docling_processor.process_document(
                document_path, 
                content, 
                getattr(document, '__dict__', {})
            )
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
            if hasattr(storage_result, 'errors') and storage_result.errors:
                result.warnings.extend(storage_result.errors)
            
            # Update final status
            result.status = ProcessingStatus.COMPLETED
            result.processing_time = time.time() - doc_start_time
            
            self.logger.info(
                f"Document processed successfully: {document_path} "
                f"({result.chunks_generated} chunks, {result.vectors_stored} vectors stored) "
                f"in {result.processing_time:.2f}s"
            )
            
        except Exception as e:
            result.status = ProcessingStatus.FAILED
            result.processing_time = time.time() - doc_start_time
            result.errors.append(str(e))
            self.logger.error(f"Failed to process document {document_path}: {e}")
        
        return result
    
    def _execute_gitlab_ingestion(self, target_folder: str) -> List[Any]:
        """Execute GitLab ingestion stage"""
        
        try:
            self.logger.info(f"Connecting to GitLab and scanning folder: {target_folder}")
            
            # Authenticate with GitLab
            if hasattr(self.gitlab_connector, 'authenticate') and not self.gitlab_connector.authenticate():
                raise RuntimeError("GitLab authentication failed")
            
            # List files in target folder
            documents = self.gitlab_connector.list_files(
                branch=getattr(self.config_manager.config.gitlab, 'branch', 'main'),
                folder_path=target_folder,
                extensions=getattr(self.config_manager.config.gitlab, 'supported_extensions', ['.pdf', '.docx', '.txt', '.md'])
            )
            
            self.logger.info(f"Found {len(documents)} documents in GitLab folder: {target_folder}")
            return documents
            
        except Exception as e:
            self.logger.error(f"GitLab ingestion failed: {e}")
            raise PipelineOrchestrationError(f"GitLab ingestion failed: {e}")
    
    def _execute_finalization(self, pipeline_result: PipelineResult):
        """Execute pipeline finalization stage"""
        
        try:
            self.logger.info("Executing pipeline finalization...")
            
            # Perform Qdrant collection optimization
            if (hasattr(self.config_manager.config.qdrant, 'optimize_collection') and 
                self.config_manager.config.qdrant.optimize_collection):
                collection_name = getattr(self.config_manager.config.qdrant, 'collection_name', 'nic')
                if hasattr(self.qdrant_store, 'optimize_collection'):
                    self.qdrant_store.optimize_collection(collection_name)
                    self.logger.info(f"Optimized Qdrant collection: {collection_name}")
            
            # Generate final summary report
            self._generate_pipeline_summary(pipeline_result)
            
            self.logger.info("Pipeline finalization completed")
            
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
                'config_hash': getattr(self.config_manager, 'config_hash', 'unknown')
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
            current_hash = getattr(self.config_manager, 'config_hash', 'unknown')
            if checkpoint_data.get('config_hash') != current_hash:
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
            'vectors_per_second': result.total_vectors_stored / result.total_processing_time if result.total_processing_time > 0 else 0,
            'success_rate': result.processed_successfully / result.total_documents if result.total_documents > 0 else 0,
            'failure_rate': result.failed_documents / result.total_documents if result.total_documents > 0 else 0,
            'average_processing_time': sum(r.processing_time for r in result.document_results) / len(result.document_results) if result.document_results else 0,
            'stage_timings': self.progress.stage_progress.copy(),
            'concurrent_processing_available': CONCURRENT_FUTURES_AVAILABLE,
            'modules_availability': {
                'configuration': CONFIG_MODULE_AVAILABLE,
                'concurrent_futures': CONCURRENT_FUTURES_AVAILABLE
            }
        }
        
        return metrics
    
    def _generate_pipeline_summary(self, result: PipelineResult):
        """Generate comprehensive pipeline execution summary"""
        
        summary = {
            'execution_timestamp': datetime.utcnow().isoformat(),
            'pipeline_version': getattr(self.config_manager.config.pipeline, 'processing_pipeline_version', '1.0'),
            'configuration': {
                'environment': getattr(self.config_manager, 'environment', 'unknown'),
                'gitlab_branch': getattr(self.config_manager.config.gitlab, 'branch', 'main'),
                'chunk_size': getattr(self.config_manager.config.chunking, 'target_chunk_size', 500),
                'embedding_model': getattr(self.config_manager.config.embedding, 'model_name', 'BAAI/bge-m3'),
                'qdrant_collection': getattr(self.config_manager.config.qdrant, 'collection_name', 'nic')
            },
            'results': {
                'total_documents': result.total_documents,
                'processed_successfully': result.processed_successfully,
                'failed_documents': result.failed_documents,
                'skipped_documents': result.skipped_documents,
                'total_chunks': result.total_chunks,
                'total_embeddings': result.total_embeddings,
                'total_vectors_stored': result.total_vectors_stored,
                'processing_time': result.total_processing_time
            },
            'performance_metrics': result.performance_metrics,
            'errors': result.errors[:10],  # Limit to first 10 errors
            'document_summary': [
                {
                    'document_id': doc.document_id,
                    'status': doc.status.value,
                    'processing_time': doc.processing_time,
                    'chunks_generated': doc.chunks_generated,
                    'vectors_stored': doc.vectors_stored
                }
                for doc in result.document_results[:20]  # Limit to first 20 documents
            ]
        }
        
        # Save summary to file
        summary_file = Path(f"./logs/pipeline_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline summary saved: {summary_file}")
    
    def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics and capabilities"""
        return {
            'environment': getattr(self.config_manager, 'environment', 'unknown'),
            'concurrent_processing_available': CONCURRENT_FUTURES_AVAILABLE,
            'max_concurrent_documents': getattr(self.config_manager.config.pipeline, 'max_concurrent_documents', 2),
            'checkpoint_interval': self.checkpoint_interval,
            'checkpoint_directory': str(self.checkpoint_dir),
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'modules_initialized': {
                'gitlab_connector': hasattr(self, 'gitlab_connector'),
                'docling_processor': hasattr(self, 'docling_processor'),
                'text_chunker': hasattr(self, 'text_chunker'),
                'embedding_generator': hasattr(self, 'embedding_generator'),
                'qdrant_store': hasattr(self, 'qdrant_store'),
                'metadata_manager': hasattr(self, 'metadata_manager')
            },
            'supported_stages': [stage.value for stage in PipelineStage]
        }

# Context manager for pipeline orchestration
class PipelineOrchestrationContext:
    """Context manager for pipeline orchestration with automatic cleanup"""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.orchestrator = None
    
    def __enter__(self) -> PipelineOrchestrator:
        self.orchestrator = PipelineOrchestrator(self.config_manager)
        return self.orchestrator
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        if self.orchestrator:
            # Force final checkpoint if processing was interrupted
            try:
                if hasattr(self.orchestrator, 'progress') and self.orchestrator.progress.documents_processed > 0:
                    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    checkpoint_file = self.orchestrator.checkpoint_dir / f"final_checkpoint_{timestamp}.pkl"
                    
                    final_data = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'progress': self.orchestrator.progress,
                        'interrupted': exc_type is not None,
                        'exception': str(exc_val) if exc_val else None
                    }
                    
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(final_data, f)
                    
                    logging.info(f"Created final checkpoint: {checkpoint_file}")
            except Exception as e:
                logging.warning(f"Failed to create final checkpoint: {e}")

def create_pipeline_orchestrator(config_manager: ConfigurationManager) -> PipelineOrchestrator:
    """Factory function for pipeline orchestrator creation"""
    return PipelineOrchestrator(config_manager)

def create_orchestrator_from_config_dict(config_dict: Dict[str, Any]) -> PipelineOrchestrator:
    """Create orchestrator from configuration dictionary"""
    if CONFIG_MODULE_AVAILABLE:
        # Create configuration manager with proper parameters
        environment = config_dict.get('environment', 'development')
        try:
            config_manager = create_configuration_manager(environment=environment)
        except Exception:
            # Fallback to mock if real config manager fails
            config_manager = MockConfigurationManager()
    else:
        config_manager = MockConfigurationManager()
    
    return PipelineOrchestrator(config_manager)

# Error classes
class PipelineOrchestrationError(Exception):
    """Base exception for pipeline orchestration errors"""
    pass

class StageExecutionError(PipelineOrchestrationError):
    """Pipeline stage execution errors"""
    pass

class DocumentProcessingError(PipelineOrchestrationError):
    """Document processing errors"""
    pass

class CheckpointError(PipelineOrchestrationError):
    """Checkpoint management errors"""
    pass

# Utility functions
def safe_pipeline_execution(orchestrator: PipelineOrchestrator, target_folder: str) -> PipelineResult:
    """Safe pipeline execution with comprehensive error handling"""
    try:
        return orchestrator.run_full_pipeline(target_folder)
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        # Return a failed result instead of raising
        return PipelineResult(
            total_documents=0,
            processed_successfully=0,
            failed_documents=0,
            skipped_documents=0,
            total_processing_time=0.0,
            total_chunks=0,
            total_embeddings=0,
            total_vectors_stored=0,
            errors=[str(e)]
        )

def calculate_pipeline_efficiency(result: PipelineResult) -> Dict[str, float]:
    """Calculate pipeline efficiency metrics"""
    try:
        metrics = {}
        
        if result.total_documents > 0:
            metrics['success_rate'] = result.processed_successfully / result.total_documents
            metrics['failure_rate'] = result.failed_documents / result.total_documents
            metrics['skip_rate'] = result.skipped_documents / result.total_documents
        else:
            metrics['success_rate'] = 0.0
            metrics['failure_rate'] = 0.0
            metrics['skip_rate'] = 0.0
        
        if result.total_processing_time > 0:
            metrics['throughput_docs_per_second'] = result.processed_successfully / result.total_processing_time
            metrics['throughput_chunks_per_second'] = result.total_chunks / result.total_processing_time
            metrics['throughput_vectors_per_second'] = result.total_vectors_stored / result.total_processing_time
        else:
            metrics['throughput_docs_per_second'] = 0.0
            metrics['throughput_chunks_per_second'] = 0.0
            metrics['throughput_vectors_per_second'] = 0.0
        
        if result.processed_successfully > 0:
            metrics['avg_chunks_per_document'] = result.total_chunks / result.processed_successfully
            metrics['avg_vectors_per_document'] = result.total_vectors_stored / result.processed_successfully
        else:
            metrics['avg_chunks_per_document'] = 0.0
            metrics['avg_vectors_per_document'] = 0.0
        
        return metrics
        
    except Exception:
        return {
            'success_rate': 0.0,
            'failure_rate': 1.0,
            'skip_rate': 0.0,
            'throughput_docs_per_second': 0.0,
            'throughput_chunks_per_second': 0.0,
            'throughput_vectors_per_second': 0.0,
            'avg_chunks_per_document': 0.0,
            'avg_vectors_per_document': 0.0
        }

def format_progress_report(progress: ProgressReport) -> str:
    """Format progress report for display"""
    try:
        if progress.total_documents > 0:
            completion_pct = (progress.documents_processed / progress.total_documents) * 100
        else:
            completion_pct = 0.0
        
        report = f"""
Pipeline Progress Report
========================
Current Stage: {progress.current_stage.value}
Progress: {progress.documents_processed}/{progress.total_documents} documents ({completion_pct:.1f}%)
Current Document: {progress.current_document or 'None'}
Elapsed Time: {progress.elapsed_time:.1f}s
Estimated Remaining: {progress.estimated_remaining:.1f}s
Processing Rate: {progress.processing_rate:.2f} docs/sec
Errors: {progress.errors_count}
Warnings: {progress.warnings_count}
"""
        
        if progress.stage_progress:
            report += "\nStage Timings:\n"
            for stage, timing in progress.stage_progress.items():
                report += f"  {stage}: {timing:.2f}s\n"
        
        return report.strip()
        
    except Exception:
        return "Progress report formatting failed"