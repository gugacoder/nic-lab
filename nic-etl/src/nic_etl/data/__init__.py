"""
Data processing components for NIC ETL Pipeline.

This module contains components responsible for document ingestion,
processing, chunking, and metadata management.
"""

from .ingestion import DocumentIngestionManager, create_document_ingestion_manager
from .processing import DoclingProcessor, create_docling_processor
from .chunking import TextChunker, create_text_chunker
from .metadata import NICSchemaManager, create_nic_schema_manager

__all__ = [
    "DocumentIngestionManager",
    "create_document_ingestion_manager",
    "DoclingProcessor", 
    "create_docling_processor",
    "TextChunker",
    "create_text_chunker",
    "NICSchemaManager",
    "create_nic_schema_manager",
]