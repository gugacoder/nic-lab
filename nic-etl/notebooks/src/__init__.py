# -*- coding: utf-8 -*-
"""
Biblioteca simplificada para pipeline NIC ETL
Funções puras que confiam 100% nos notebooks
"""

__version__ = "1.0.0"
__author__ = "NIC ETL Team"

# Imports principais
from .pipeline_utils import (
    PipelineState,
    check_prerequisites,
    show_pipeline_progress,
    create_test_data,
    format_file_size,
    get_timestamp,
    ensure_directory
)

from .gitlab_utils import (
    download_documents,
    validate_gitlab_connection,
    list_repository_files,
    clean_download_directory
)

from .docling_utils import (
    extract_content_from_file,
    batch_process_documents,
    validate_processed_content
)

from .chunking_utils import (
    split_text_into_chunks,
    process_document_chunks,
    batch_process_documents_chunking,
    validate_chunks
)

from .embedding_utils import (
    generate_embeddings,
    process_chunks_to_embeddings,
    batch_process_chunks_to_embeddings,
    validate_embeddings,
    extract_embeddings_for_qdrant
)

from .qdrant_utils import (
    store_embeddings_in_qdrant,
    create_qdrant_collection,
    validate_qdrant_connection,
    search_similar_vectors,
    get_collection_info,
    batch_store_from_files,
    cleanup_collection
)

__all__ = [
    # Pipeline utilities
    "PipelineState",
    "check_prerequisites", 
    "show_pipeline_progress",
    "create_test_data",
    "format_file_size",
    "get_timestamp",
    "ensure_directory",
    
    # GitLab utilities
    "download_documents",
    "validate_gitlab_connection",
    "list_repository_files", 
    "clean_download_directory",
    
    # Docling utilities
    "extract_content_from_file",
    "batch_process_documents",
    "validate_processed_content",
    
    # Chunking utilities
    "split_text_into_chunks",
    "process_document_chunks",
    "batch_process_documents_chunking",
    "validate_chunks",
    
    # Embedding utilities
    "generate_embeddings",
    "process_chunks_to_embeddings", 
    "batch_process_chunks_to_embeddings",
    "validate_embeddings",
    "extract_embeddings_for_qdrant",
    
    # Qdrant utilities
    "store_embeddings_in_qdrant",
    "create_qdrant_collection",
    "validate_qdrant_connection",
    "search_similar_vectors",
    "get_collection_info",
    "batch_store_from_files",
    "cleanup_collection"
]