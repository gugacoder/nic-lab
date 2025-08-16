"""
NIC ETL Pipeline - Main Package

Núcleo de Inteligência e Conhecimento - Extract, Transform, Load Pipeline
for processing documents from GitLab repositories, generating embeddings,
and storing vectors in Qdrant for semantic search.
"""

__version__ = "1.0.0"
__author__ = "NIC ETL Team"
__description__ = "Document processing pipeline with semantic search capabilities"

# Import main components for easy access
from .core.configuration import create_configuration_manager
from .core.orchestration import PipelineOrchestrator, create_pipeline_orchestrator
from .core.errors import ErrorManager, ErrorCategory

# Data processing components
from .data.ingestion import DocumentIngestionManager, create_document_ingestion_manager
from .data.processing import DoclingProcessor, create_docling_processor
from .data.chunking import TextChunker, create_text_chunker
from .data.metadata import NICSchemaManager, create_nic_schema_manager

# ML components
from .ml.embeddings import EmbeddingGenerator, create_embedding_generator

# Storage components
from .storage.qdrant import QdrantVectorStore, create_qdrant_vector_store
from .storage.gitlab import GitLabConnector, create_gitlab_connector

__all__ = [
    # Configuration
    "create_configuration_manager",
    
    # Core orchestration
    "PipelineOrchestrator",
    "create_pipeline_orchestrator",
    
    # Error handling
    "ErrorManager",
    "ErrorCategory",
    
    # Data processing
    "DocumentIngestionManager",
    "create_document_ingestion_manager",
    "DoclingProcessor", 
    "create_docling_processor",
    "TextChunker",
    "create_text_chunker",
    "NICSchemaManager",
    "create_nic_schema_manager",
    
    # ML
    "EmbeddingGenerator",
    "create_embedding_generator",
    
    # Storage
    "QdrantVectorStore",
    "create_qdrant_vector_store",
    "GitLabConnector",
    "create_gitlab_connector",
]