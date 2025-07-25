"""
Semantic Search Strategy for GitLab Content

This module implements semantic search functionality using embeddings and 
similarity matching to find conceptually related content beyond keyword matching.
"""

import logging
import asyncio
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

import numpy as np

from ..gitlab_client import GitLabClient, GitLabSearchResult, get_gitlab_client

logger = logging.getLogger(__name__)


@dataclass
class SemanticConfig:
    """Configuration for semantic search"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.3
    max_content_length: int = 512  # Max tokens for embedding
    cache_embeddings: bool = True
    cache_duration_hours: int = 24
    top_k_results: int = 50
    rerank_results: bool = True
    chunk_size: int = 256
    chunk_overlap: int = 64


@dataclass 
class EmbeddingResult:
    """Represents an embedding result with metadata"""
    content: str
    embedding: np.ndarray
    source: GitLabSearchResult
    timestamp: datetime
    chunk_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching"""
        return {
            'content': self.content,
            'embedding': self.embedding.tolist(),
            'timestamp': self.timestamp.isoformat(),
            'chunk_index': self.chunk_index,
            'source': {
                'project_id': self.source.project_id,
                'project_name': self.source.project_name,
                'file_path': self.source.file_path,
                'ref': self.source.ref,
                'startline': self.source.startline,
                'content': self.source.content,
                'wiki': self.source.wiki
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingResult':
        """Create from dictionary (for cache loading)"""
        source_data = data['source']
        source = GitLabSearchResult(
            project_id=source_data['project_id'],
            project_name=source_data['project_name'],
            file_path=source_data['file_path'],
            ref=source_data['ref'],
            startline=source_data['startline'],
            content=source_data['content'],
            wiki=source_data.get('wiki', False)
        )
        
        return cls(
            content=data['content'],
            embedding=np.array(data['embedding']),
            source=source,
            timestamp=datetime.fromisoformat(data['timestamp']),
            chunk_index=data.get('chunk_index', 0)
        )


class SemanticSearchStrategy:
    """Semantic search strategy using embeddings for similarity matching"""
    
    def __init__(
        self,
        gitlab_client: Optional[GitLabClient] = None,
        config: Optional[SemanticConfig] = None
    ):
        """Initialize semantic search strategy
        
        Args:
            gitlab_client: GitLab client instance
            config: Semantic search configuration
        """
        self.client = gitlab_client or get_gitlab_client()
        self.config = config or SemanticConfig()
        
        # Initialize embedding model
        self._embedding_model = None
        self._embedding_cache: Dict[str, EmbeddingResult] = {}
        
        # Try to import sentence transformers
        try:
            import sentence_transformers
            self._has_sentence_transformers = True
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback similarity")
            self._has_sentence_transformers = False
    
    @property
    def embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None and self._has_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.config.embedding_model)
                logger.info(f"Loaded embedding model: {self.config.embedding_model}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self._has_sentence_transformers = False
        
        return self._embedding_model
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if not available
        """
        if not self._has_sentence_transformers or not text.strip():
            return None
        
        try:
            # Truncate text if too long
            if len(text) > self.config.max_content_length:
                text = text[:self.config.max_content_length]
            
            # Check cache first
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._embedding_cache:
                cached = self._embedding_cache[cache_key]
                # Check if cache is still valid
                if datetime.now() - cached.timestamp < timedelta(hours=self.config.cache_duration_hours):
                    return cached.embedding
            
            # Generate new embedding
            embedding = self.embedding_model.encode([text], show_progress_bar=False)[0]
            
            # Cache if enabled
            if self.config.cache_embeddings:
                # Note: We can't create a full EmbeddingResult without a source
                # This is a simplified cache for just embeddings
                pass
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            # Ensure result is in [0, 1] range
            return max(0.0, float(similarity))
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _fallback_similarity(self, query: str, content: str) -> float:
        """Fallback similarity calculation when embeddings not available
        
        Args:
            query: Search query
            content: Content to compare
            
        Returns:
            Similarity score (0-1)
        """
        # Simple word overlap similarity
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _chunk_content(self, content: str) -> List[str]:
        """Split content into chunks for embedding
        
        Args:
            content: Full content to chunk
            
        Returns:
            List of content chunks
        """
        if len(content) <= self.config.chunk_size:
            return [content]
        
        chunks = []
        words = content.split()
        
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > self.config.chunk_size and current_chunk:
                # Create chunk with some overlap
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_words = max(1, self.config.chunk_overlap // 10)  # Rough estimate
                current_chunk = current_chunk[-overlap_words:]
                current_length = sum(len(w) + 1 for w in current_chunk)
            
            current_chunk.append(word)
            current_length += word_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def search_projects(
        self,
        query: str,
        project_ids: Optional[List[int]] = None,
        file_extensions: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[GitLabSearchResult]:
        """Search projects using semantic similarity
        
        Args:
            query: Search query
            project_ids: Specific projects to search
            file_extensions: File extensions to filter by
            limit: Maximum results to return
            
        Returns:
            List of semantically similar results
        """
        logger.info(f"Starting semantic search for: '{query}'")
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            logger.warning("Could not generate query embedding, falling back to basic search")
            return self._fallback_search(query, project_ids, file_extensions, limit)
        
        # Get initial candidate results (broader search)
        candidates = self.client.search_files(
            query=query,
            project_ids=project_ids,
            file_extensions=file_extensions,
            limit=limit * 3  # Get more candidates for semantic filtering
        )
        
        if not candidates:
            logger.info("No candidate results found")
            return []
        
        # Calculate semantic similarities
        scored_results = []
        
        for result in candidates:
            try:
                # Process content in chunks if needed
                chunks = self._chunk_content(result.content)
                best_score = 0.0
                best_chunk_idx = 0
                
                for idx, chunk in enumerate(chunks):
                    chunk_embedding = self._get_embedding(chunk)
                    
                    if chunk_embedding is not None:
                        similarity = self._calculate_cosine_similarity(query_embedding, chunk_embedding)
                    else:
                        similarity = self._fallback_similarity(query, chunk)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_chunk_idx = idx
                
                # Only include results above threshold
                if best_score >= self.config.similarity_threshold:
                    # Create enhanced result with chunk info
                    enhanced_result = result
                    if len(chunks) > 1:
                        # Update content to best matching chunk
                        enhanced_result.content = chunks[best_chunk_idx]
                    
                    scored_results.append((enhanced_result, best_score))
                    
            except Exception as e:
                logger.warning(f"Error processing result {result.file_path}: {e}")
                continue
        
        # Sort by similarity score and limit results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        final_results = [result for result, score in scored_results[:limit]]
        
        logger.info(f"Semantic search returned {len(final_results)} results")
        return final_results
    
    def _fallback_search(
        self,
        query: str,
        project_ids: Optional[List[int]] = None,
        file_extensions: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[GitLabSearchResult]:
        """Fallback search when embeddings not available
        
        Args:
            query: Search query
            project_ids: Projects to search
            file_extensions: File extensions to filter
            limit: Maximum results
            
        Returns:
            Basic search results with fallback similarity scoring
        """
        logger.info("Using fallback semantic search")
        
        # Get basic results
        results = self.client.search_files(
            query=query,
            project_ids=project_ids,
            file_extensions=file_extensions,
            limit=limit * 2
        )
        
        # Score using fallback similarity
        scored_results = []
        for result in results:
            score = self._fallback_similarity(query, result.content)
            if score > 0.1:  # Lower threshold for fallback
                scored_results.append((result, score))
        
        # Sort and limit
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [result for result, score in scored_results[:limit]]
    
    def find_similar_content(
        self,
        reference_content: str,
        project_ids: Optional[List[int]] = None,
        exclude_file: Optional[str] = None,
        limit: int = 10
    ) -> List[Tuple[GitLabSearchResult, float]]:
        """Find content similar to a reference text
        
        Args:
            reference_content: Content to find similar items for
            project_ids: Projects to search in
            exclude_file: File path to exclude from results
            limit: Maximum similar items to return
            
        Returns:
            List of (result, similarity_score) tuples
        """
        logger.info("Finding similar content")
        
        # Generate embedding for reference content
        ref_embedding = self._get_embedding(reference_content)
        
        # Get broader search results (using key terms from reference)
        words = reference_content.split()
        key_words = [w for w in words if len(w) > 3][:5]  # Extract key terms
        search_query = ' '.join(key_words)
        
        candidates = self.client.search_files(
            query=search_query,
            project_ids=project_ids,
            limit=50
        )
        
        similar_results = []
        
        for result in candidates:
            # Skip excluded file
            if exclude_file and result.file_path == exclude_file:
                continue
            
            if ref_embedding is not None:
                content_embedding = self._get_embedding(result.content)
                if content_embedding is not None:
                    similarity = self._calculate_cosine_similarity(ref_embedding, content_embedding)
                else:
                    similarity = self._fallback_similarity(reference_content, result.content)
            else:
                similarity = self._fallback_similarity(reference_content, result.content)
            
            if similarity > self.config.similarity_threshold:
                similar_results.append((result, similarity))
        
        # Sort by similarity
        similar_results.sort(key=lambda x: x[1], reverse=True)
        
        return similar_results[:limit]
    
    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on partial query
        
        Args:
            partial_query: Partial search query
            limit: Maximum suggestions to return
            
        Returns:
            List of suggested search terms
        """
        # This is a simplified implementation
        # In a full system, this could use a more sophisticated approach
        
        suggestions = []
        
        # Add common technical terms that might be related
        common_terms = [
            'authentication', 'configuration', 'installation', 'setup',
            'documentation', 'api', 'database', 'deployment', 'testing',
            'security', 'performance', 'troubleshooting', 'integration'
        ]
        
        query_lower = partial_query.lower()
        
        for term in common_terms:
            if query_lower in term or term.startswith(query_lower):
                suggestions.append(term)
        
        return suggestions[:limit]
    
    def test_search(self, query: str, project_id: Optional[int] = None) -> Dict[str, Any]:
        """Test semantic search functionality
        
        Args:
            query: Test query
            project_id: Optional specific project
            
        Returns:
            Test results with performance metrics
        """
        import time
        
        start_time = time.time()
        
        try:
            project_ids = [project_id] if project_id else None
            results = self.search_projects(query, project_ids=project_ids, limit=10)
            
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            # Test embedding generation
            query_embedding = self._get_embedding(query)
            has_embeddings = query_embedding is not None
            
            return {
                'success': True,
                'query': query,
                'results_count': len(results),
                'duration_ms': duration,
                'has_embeddings': has_embeddings,
                'embedding_model': self.config.embedding_model if has_embeddings else 'fallback',
                'similarity_threshold': self.config.similarity_threshold,
                'sample_results': [
                    {
                        'project': r.project_name,
                        'file': r.file_path,
                        'content_preview': r.content[:150] + '...'
                    }
                    for r in results[:3]
                ],
                'config': asdict(self.config)
            }
            
        except Exception as e:
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'duration_ms': duration
            }


def create_semantic_search(
    gitlab_client: Optional[GitLabClient] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.3,
    cache_embeddings: bool = True
) -> SemanticSearchStrategy:
    """Factory function to create semantic search strategy
    
    Args:
        gitlab_client: GitLab client instance
        embedding_model: Model to use for embeddings
        similarity_threshold: Minimum similarity for results
        cache_embeddings: Whether to cache embeddings
        
    Returns:
        Configured semantic search strategy
    """
    config = SemanticConfig(
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        cache_embeddings=cache_embeddings
    )
    
    return SemanticSearchStrategy(gitlab_client, config)


if __name__ == "__main__":
    # Test semantic search functionality
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.integrations.search.semantic_search <query> [project_id]")
        sys.exit(1)
    
    query = sys.argv[1]
    project_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print(f"Testing semantic search with query: '{query}'")
    
    search_strategy = create_semantic_search()
    results = search_strategy.test_search(query, project_id)
    
    print("Test Results:")
    print(f"  Success: {results['success']}")
    print(f"  Duration: {results.get('duration_ms', 0):.1f}ms")
    
    if results['success']:
        print(f"  Has embeddings: {results['has_embeddings']}")
        print(f"  Model: {results['embedding_model']}")
        print(f"  Results found: {results['results_count']}")
        print(f"  Similarity threshold: {results['similarity_threshold']}")
        if results['sample_results']:
            print("  Sample results:")
            for result in results['sample_results']:
                print(f"    - {result['project']}: {result['file']}")
    else:
        print(f"  Error: {results['error']}")