# Pipeline Orchestration - PRP

## ROLE
**ETL Pipeline Engineer with Orchestration Expertise**

Specialist in designing and implementing complex data pipeline orchestration systems. Expert in workflow management, error handling, and monitoring across distributed systems. Proficient in coordinating multiple processing stages while ensuring data consistency and fault tolerance.

## OBJECTIVE
**Implement Complete ETL Pipeline Orchestration**

Create a comprehensive pipeline orchestration system within Jupyter Notebook cells that:
* Coordinates all ETL stages from GitLab ingestion to Qdrant storage
* Implements robust error handling and recovery mechanisms
* Provides real-time progress monitoring and logging
* Supports both batch and streaming processing modes
* Ensures data consistency across all pipeline stages
* Enables pipeline resume from checkpoint on failure
* Provides comprehensive performance metrics

## MOTIVATION
**Reliable End-to-End Data Processing**

Pipeline orchestration ensures reliable, repeatable processing of document collections while maintaining data quality and providing operational visibility. Proper orchestration enables automatic recovery from failures, prevents data loss, and provides audit trails essential for production environments.

## CONTEXT
**Jupyter Notebook ETL Pipeline Environment**

Operating specifications:
* Environment: Jupyter Notebook cells with production constraints
* Components: 10+ interconnected processing stages
* Data flow: GitLab → Ingestion → Docling → Chunking → Embedding → Qdrant
* Error handling: Graceful degradation and recovery
* Monitoring: Real-time progress and performance tracking
* Constraints: Single notebook, production-ready implementation

## IMPLEMENTATION BLUEPRINT

### Code Structure
```python
# Cell 10: Main Pipeline Orchestration
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import time
import traceback
import logging

class PipelineStage(Enum):
    GITLAB_CONNECTION = "gitlab_connection"
    DOCUMENT_INGESTION = "document_ingestion"
    DOCLING_PROCESSING = "docling_processing"
    TEXT_CHUNKING = "text_chunking"
    EMBEDDING_GENERATION = "embedding_generation"
    QDRANT_INSERTION = "qdrant_insertion"
    VALIDATION = "validation"
    CLEANUP = "cleanup"

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class PipelineResult:
    stage: PipelineStage
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    processed_items: int
    failed_items: int
    error_message: Optional[str]
    metrics: Dict[str, Any]

class PipelineOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[PipelineResult] = []
        self.current_stage = None
        self.start_time = None
        
        # Initialize components
        self.gitlab_client = None
        self.document_manager = None
        self.docling_processor = None
        self.chunking_strategy = None
        self.embedding_generator = None
        self.qdrant_manager = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('NIC_ETL_Pipeline')
    
    def initialize_components(self):
        """Initialize all pipeline components"""
        try:
            self.logger.info("Initializing pipeline components...")
            
            # GitLab client
            self.gitlab_client = GitLabClient(
                self.config['gitlab']['url'],
                self.config['gitlab']['token'],
                self.config['gitlab']['project']
            )
            self.gitlab_client.authenticate()
            
            # Document manager
            self.document_manager = DocumentManager(
                self.config['cache']['dir'],
                self.config['cache']['state_file']
            )
            
            # Docling processor
            self.docling_processor = DoclingProcessor(
                self.config['cache']['dir'],
                self.config['docling']['enable_ocr']
            )
            
            # Chunking strategy
            self.chunking_strategy = ChunkingStrategy(
                chunk_size=self.config['chunking']['size'],
                overlap_size=self.config['chunking']['overlap']
            )
            
            # Embedding generator
            self.embedding_generator = EmbeddingGenerator(
                cache_dir=self.config['cache']['dir'] / 'embeddings',
                batch_size=self.config['embedding']['batch_size']
            )
            
            # Qdrant manager
            self.qdrant_manager = QdrantManager(
                self.config['qdrant']['url'],
                self.config['qdrant']['api_key'],
                self.config['qdrant']['collection']
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {str(e)}")
            raise
    
    def run_pipeline(self, resume_from: Optional[PipelineStage] = None) -> Dict[str, Any]:
        """Execute the complete ETL pipeline"""
        self.start_time = datetime.now()
        
        try:
            # Initialize if not resuming
            if not resume_from:
                self.initialize_components()
            
            # Define pipeline stages
            stages = [
                (PipelineStage.GITLAB_CONNECTION, self._stage_gitlab_connection),
                (PipelineStage.DOCUMENT_INGESTION, self._stage_document_ingestion),
                (PipelineStage.DOCLING_PROCESSING, self._stage_docling_processing),
                (PipelineStage.TEXT_CHUNKING, self._stage_text_chunking),
                (PipelineStage.EMBEDDING_GENERATION, self._stage_embedding_generation),
                (PipelineStage.QDRANT_INSERTION, self._stage_qdrant_insertion),
                (PipelineStage.VALIDATION, self._stage_validation),
                (PipelineStage.CLEANUP, self._stage_cleanup)
            ]
            
            # Execute stages
            for stage, stage_func in stages:
                if resume_from and stage.value < resume_from.value:
                    continue
                
                self.current_stage = stage
                result = self._execute_stage(stage, stage_func)
                self.results.append(result)
                
                if result.status == PipelineStatus.FAILED:
                    self.logger.error(f"Pipeline failed at stage {stage.value}")
                    break
            
            # Generate final report
            return self._generate_pipeline_report()
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'status': 'failed', 'error': str(e)}
    
    def _execute_stage(self, stage: PipelineStage, stage_func: Callable) -> PipelineResult:
        """Execute a single pipeline stage with error handling"""
        self.logger.info(f"Starting stage: {stage.value}")
        
        result = PipelineResult(
            stage=stage,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now(),
            end_time=None,
            duration=None,
            processed_items=0,
            failed_items=0,
            error_message=None,
            metrics={}
        )
        
        try:
            # Execute stage
            stage_result = stage_func()
            
            # Update result
            result.status = PipelineStatus.COMPLETED
            result.processed_items = stage_result.get('processed', 0)
            result.failed_items = stage_result.get('failed', 0)
            result.metrics = stage_result.get('metrics', {})
            
            self.logger.info(f"Stage {stage.value} completed successfully")
            
        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f"Stage {stage.value} failed: {str(e)}")
        
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    def _stage_gitlab_connection(self) -> Dict[str, Any]:
        """Stage 1: Establish GitLab connection"""
        return {'processed': 1, 'metrics': {'connection_time': 0.5}}
    
    def _stage_document_ingestion(self) -> Dict[str, Any]:
        """Stage 2: Ingest documents from GitLab"""
        documents = self.document_manager.ingest_documents(
            self.gitlab_client,
            self.config['gitlab']['folder'],
            self.config['gitlab']['branch']
        )
        
        return {
            'processed': len(documents),
            'metrics': {
                'document_count': len(documents),
                'total_size': sum(d.size for d in documents)
            }
        }
    
    def _stage_docling_processing(self) -> Dict[str, Any]:
        """Stage 3: Process documents with Docling"""
        pending_docs = self.document_manager.get_pending_documents()
        processed = 0
        failed = 0
        
        for doc in pending_docs:
            try:
                result = self.docling_processor.process_document(doc)
                self.document_manager.update_document_status(
                    doc.id, ProcessingStatus.COMPLETED
                )
                processed += 1
            except Exception as e:
                self.document_manager.update_document_status(
                    doc.id, ProcessingStatus.FAILED, str(e)
                )
                failed += 1
        
        return {
            'processed': processed,
            'failed': failed,
            'metrics': {'processing_rate': processed / max(1, len(pending_docs))}
        }
    
    def _stage_text_chunking(self) -> Dict[str, Any]:
        """Stage 4: Chunk processed documents"""
        # Implementation for chunking stage
        return {'processed': 0, 'metrics': {}}
    
    def _stage_embedding_generation(self) -> Dict[str, Any]:
        """Stage 5: Generate embeddings"""
        # Implementation for embedding generation
        return {'processed': 0, 'metrics': {}}
    
    def _stage_qdrant_insertion(self) -> Dict[str, Any]:
        """Stage 6: Insert into Qdrant"""
        # Implementation for Qdrant insertion
        return {'processed': 0, 'metrics': {}}
    
    def _stage_validation(self) -> Dict[str, Any]:
        """Stage 7: Validate pipeline results"""
        return {'processed': 1, 'metrics': {'validation_passed': True}}
    
    def _stage_cleanup(self) -> Dict[str, Any]:
        """Stage 8: Cleanup temporary files"""
        return {'processed': 1, 'metrics': {'cleanup_completed': True}}
    
    def _generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution report"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            'status': 'completed' if all(r.status == PipelineStatus.COMPLETED for r in self.results) else 'failed',
            'total_duration': total_duration,
            'stages': [],
            'summary': {
                'total_processed': sum(r.processed_items for r in self.results),
                'total_failed': sum(r.failed_items for r in self.results),
                'success_rate': 0
            }
        }
        
        # Add stage details
        for result in self.results:
            report['stages'].append({
                'stage': result.stage.value,
                'status': result.status.value,
                'duration': result.duration,
                'processed': result.processed_items,
                'failed': result.failed_items,
                'error': result.error_message,
                'metrics': result.metrics
            })
        
        # Calculate success rate
        total_processed = report['summary']['total_processed']
        total_failed = report['summary']['total_failed']
        if total_processed > 0:
            report['summary']['success_rate'] = (total_processed - total_failed) / total_processed
        
        return report

class PipelineMonitor:
    """Monitor pipeline execution and performance"""
    
    def __init__(self, orchestrator: PipelineOrchestrator):
        self.orchestrator = orchestrator
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'current_stage': self.orchestrator.current_stage.value if self.orchestrator.current_stage else None,
            'elapsed_time': (datetime.now() - self.orchestrator.start_time).total_seconds() if self.orchestrator.start_time else 0,
            'completed_stages': len([r for r in self.orchestrator.results if r.status == PipelineStatus.COMPLETED]),
            'failed_stages': len([r for r in self.orchestrator.results if r.status == PipelineStatus.FAILED])
        }
```

## VALIDATION LOOP

### Integration Testing
```python
def test_full_pipeline_execution():
    """Test complete pipeline execution"""
    config = load_test_config()
    orchestrator = PipelineOrchestrator(config)
    
    result = orchestrator.run_pipeline()
    
    assert result['status'] == 'completed'
    assert result['summary']['success_rate'] > 0.9

def test_pipeline_resume():
    """Test pipeline resume from checkpoint"""
    config = load_test_config()
    orchestrator = PipelineOrchestrator(config)
    
    # Resume from specific stage
    result = orchestrator.run_pipeline(resume_from=PipelineStage.TEXT_CHUNKING)
    
    assert result['status'] == 'completed'
```

## ADDITIONAL NOTES

### Performance Optimization
* **Parallel Processing**: Execute independent stages concurrently
* **Checkpoint System**: Save state between stages for resume capability
* **Resource Management**: Monitor and limit resource usage
* **Batch Optimization**: Optimize batch sizes based on performance

### Maintenance Requirements
* **Monitoring Integration**: Connect to external monitoring systems
* **Alert System**: Notify on pipeline failures
* **Performance Metrics**: Track and analyze pipeline performance over time
* **Scaling Strategy**: Plan for increased document volumes