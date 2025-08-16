"""
Machine Learning components for NIC ETL Pipeline.

This module contains ML-related components for embedding generation
and other machine learning operations.
"""

from .embeddings import EmbeddingGenerator, create_embedding_generator

__all__ = [
    "EmbeddingGenerator",
    "create_embedding_generator",
]