"""
Integration Module for New Chunking System

This module provides integration utilities to connect the new preprocessing
system with existing components like the RAG pipeline and context assembly.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from .chunker import ContentChunker, ChunkingConfig, ChunkingStrategy, ContentChunk

logger = logging.getLogger(__name__)


class LangChainDocumentAdapter:
    """
    Adapter to convert between our ContentChunk objects and
    LangChain Document objects for seamless integration.
    """
    
    def __init__(self):
        self.logger = logger
    
    def chunks_to_documents(self, chunks: List[ContentChunk]) -> List[Document]:
        """Convert ContentChunk objects to LangChain Documents
        
        Args:
            chunks: List of ContentChunk objects
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for chunk in chunks:
            # Create LangChain document with chunk content
            doc = Document(
                page_content=chunk.content,
                metadata={
                    # Preserve original metadata if it exists
                    **(chunk.metadata or {}),
                    
                    # Add chunk-specific metadata
                    'chunk_id': chunk.chunk_id,
                    'chunk_index': chunk.chunk_index,
                    'chunk_type': chunk.chunk_type,
                    'chunk_size': chunk.size,
                    'estimated_tokens': chunk.estimated_tokens,
                    'quality_score': chunk.quality_score,
                    
                    # Position information
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'document_position': chunk.document_position,
                    
                    # Section context
                    'section_title': chunk.section_title,
                    'section_level': chunk.section_level,
                    
                    # Quality metrics
                    'semantic_coherence_score': chunk.semantic_coherence_score,
                    'structural_completeness': chunk.structural_completeness,
                    'information_density': chunk.information_density
                }
            )
            documents.append(doc)
        
        return documents
    
    def documents_to_chunks(self, documents: List[Document], document_id: str = None) -> List[ContentChunk]:
        """Convert LangChain Documents to ContentChunk objects
        
        Args:
            documents: List of LangChain Document objects
            document_id: Optional document ID for chunk IDs
            
        Returns:
            List of ContentChunk objects
        """
        chunks = []
        
        for i, doc in enumerate(documents):
            metadata = doc.metadata or {}
            
            chunk = ContentChunk(
                content=doc.page_content,
                chunk_id=metadata.get('chunk_id', f'{document_id or "doc"}_{i:04d}'),
                chunk_index=metadata.get('chunk_index', i),
                start_char=metadata.get('start_char', 0),
                end_char=metadata.get('end_char', len(doc.page_content)),
                chunk_type=metadata.get('chunk_type', 'text'),
                content_hash=metadata.get('content_hash'),
                language=metadata.get('language'),
                section_title=metadata.get('section_title'),
                section_level=metadata.get('section_level', 0),
                document_position=metadata.get('document_position', 0.0),
                semantic_coherence_score=metadata.get('semantic_coherence_score', 0.0),
                structural_completeness=metadata.get('structural_completeness', 0.0),
                information_density=metadata.get('information_density', 0.0),
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks


class ContextAssemblyIntegration:
    """
    Integration layer for the Context Assembly Engine to use
    the new intelligent chunking system.
    """
    
    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        """Initialize context assembly integration
        
        Args:
            chunking_config: Configuration for chunking
        """
        self.chunking_config = chunking_config or ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            target_chunk_size=1500,
            max_chunk_size=2000,
            optimize_for_tokens=True
        )
        
        self.chunker = ContentChunker(self.chunking_config)
        self.document_adapter = LangChainDocumentAdapter()
        
        logger.info("ContextAssemblyIntegration initialized")
    
    async def process_documents_for_context(
        self,
        documents: List[Document],
        max_context_tokens: int = 8000,
        query: Optional[str] = None
    ) -> List[Document]:
        """
        Process documents using intelligent chunking for optimal context assembly
        
        Args:
            documents: List of LangChain documents to process
            max_context_tokens: Maximum tokens for final context
            query: Optional query for relevance-based selection
            
        Returns:
            List of optimally chunked and selected documents
        """
        if not documents:
            return documents
        
        logger.debug(f"Processing {len(documents)} documents for context assembly")
        
        # Convert documents to chunks for processing
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            # Extract source metadata
            source_metadata = doc.metadata.copy() if doc.metadata else {}
            source_metadata['original_document_index'] = doc_idx
            
            # Determine file path for type detection
            file_path = source_metadata.get('file_path') or source_metadata.get('source', 'unknown.txt')
            
            # Chunk the document content
            document_chunks = await self.chunker.chunk_document(
                doc.page_content,
                document_id=f"doc_{doc_idx}",
                source_metadata=source_metadata,
                file_path=file_path
            )
            
            all_chunks.extend(document_chunks)
        
        # Select optimal chunks for context
        selected_chunks = await self._select_optimal_chunks(
            all_chunks, max_context_tokens, query
        )
        
        # Convert back to LangChain documents
        optimized_documents = self.document_adapter.chunks_to_documents(selected_chunks)
        
        logger.debug(f"Selected {len(optimized_documents)} optimized chunks from {len(all_chunks)} total chunks")
        
        return optimized_documents
    
    async def _select_optimal_chunks(
        self,
        chunks: List[ContentChunk],
        max_tokens: int,
        query: Optional[str] = None
    ) -> List[ContentChunk]:
        """Select optimal chunks within token limit
        
        Args:
            chunks: List of available chunks
            max_tokens: Maximum token limit
            query: Optional query for relevance scoring
            
        Returns:
            List of selected chunks
        """
        if not chunks:
            return chunks
        
        # Score chunks for relevance and quality
        scored_chunks = []
        for chunk in chunks:
            score = self._calculate_chunk_score(chunk, query)
            scored_chunks.append((chunk, score))
        
        # Sort by score (highest first)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Select chunks within token limit
        selected_chunks = []
        total_tokens = 0
        
        for chunk, score in scored_chunks:
            chunk_tokens = chunk.estimated_tokens
            
            if total_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                # Try to fit a truncated version if there's significant space
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # At least 100 tokens worth
                    truncated_chunk = self._truncate_chunk(chunk, remaining_tokens)
                    if truncated_chunk:
                        selected_chunks.append(truncated_chunk)
                        total_tokens += truncated_chunk.estimated_tokens
                break
        
        # Sort selected chunks by document position to maintain order
        selected_chunks.sort(key=lambda c: (
            c.metadata.get('original_document_index', 0),
            c.start_char
        ))
        
        return selected_chunks
    
    def _calculate_chunk_score(self, chunk: ContentChunk, query: Optional[str] = None) -> float:
        """Calculate relevance score for chunk selection
        
        Args:
            chunk: ContentChunk to score
            query: Optional query for relevance
            
        Returns:
            Score for chunk selection (higher is better)
        """
        # Base score from chunk quality
        score = chunk.quality_score * 0.4
        
        # Add position bonus (earlier chunks often more important)
        position_bonus = (1.0 - chunk.document_position) * 0.1
        score += position_bonus
        
        # Add section level bonus (higher level sections more important)
        if chunk.section_level > 0:
            section_bonus = min(chunk.section_level / 6.0, 0.2)
            score += section_bonus
        
        # Add content type bonus
        type_bonuses = {
            'header': 0.3,
            'section': 0.2,
            'code': 0.1,
            'table': 0.05,
            'list': 0.05
        }
        score += type_bonuses.get(chunk.chunk_type, 0.0)
        
        # Query relevance (simplified)
        if query and query.strip():
            query_words = set(query.lower().split())
            chunk_words = set(chunk.content.lower().split())
            
            if query_words and chunk_words:
                overlap = len(query_words.intersection(chunk_words))
                relevance_bonus = (overlap / len(query_words)) * 0.3
                score += relevance_bonus
        
        return min(score, 1.0)
    
    def _truncate_chunk(self, chunk: ContentChunk, max_tokens: int) -> Optional[ContentChunk]:
        """Truncate chunk to fit within token limit
        
        Args:
            chunk: Chunk to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated chunk or None if too small
        """
        if max_tokens < 50:  # Too small to be useful
            return None
        
        # Estimate character limit (rough: 1 token ≈ 4 characters)
        max_chars = max_tokens * 4
        
        if len(chunk.content) <= max_chars:
            return chunk
        
        # Truncate at sentence boundary if possible
        truncated_content = chunk.content[:max_chars]
        
        # Find last sentence boundary
        sentence_endings = ['.', '!', '?']
        last_sentence_end = -1
        
        for ending in sentence_endings:
            pos = truncated_content.rfind(ending)
            if pos > max_chars * 0.7:  # Only if we keep most of the content
                last_sentence_end = max(last_sentence_end, pos + 1)
        
        if last_sentence_end > 0:
            truncated_content = truncated_content[:last_sentence_end]
        else:
            # Truncate at word boundary
            last_space = truncated_content.rfind(' ')
            if last_space > max_chars * 0.8:
                truncated_content = truncated_content[:last_space]
            
            truncated_content += "..."
        
        # Create truncated chunk
        truncated_chunk = ContentChunk(
            content=truncated_content,
            chunk_id=f"{chunk.chunk_id}_trunc",
            chunk_index=chunk.chunk_index,
            start_char=chunk.start_char,
            end_char=chunk.start_char + len(truncated_content),
            chunk_type=f"{chunk.chunk_type}_truncated",
            section_title=chunk.section_title,
            section_level=chunk.section_level,
            document_position=chunk.document_position,
            semantic_coherence_score=chunk.semantic_coherence_score * 0.8,  # Slightly lower
            structural_completeness=chunk.structural_completeness * 0.7,    # Lower due to truncation
            information_density=chunk.information_density,
            metadata={**(chunk.metadata or {}), 'truncated': True, 'original_size': chunk.size}
        )
        
        return truncated_chunk
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get statistics about chunking operations
        
        Returns:
            Dictionary of chunking statistics
        """
        return self.chunker.get_stats()


def patch_qa_chain_for_intelligent_chunking():
    """
    Monkey patch the existing QA chain to use intelligent chunking
    instead of simple truncation.
    """
    from ..ai.chains.qa_chain import RAGQAChain
    
    # Create context assembly integration
    context_integration = ContextAssemblyIntegration()
    
    async def intelligent_truncate_documents(self, documents: List[Document]) -> List[Document]:
        """Intelligent document processing using the new chunking system"""
        if not documents:
            return documents
        
        # Use the new context assembly integration
        processed_docs = await context_integration.process_documents_for_context(
            documents=documents,
            max_context_tokens=self.max_context_length // 4,  # Convert chars to tokens estimate
            query=getattr(self, '_current_query', None)
        )
        
        return processed_docs
    
    # Store original method
    RAGQAChain._original_truncate_documents = RAGQAChain._truncate_documents
    
    # Replace with intelligent version
    RAGQAChain._truncate_documents = intelligent_truncate_documents
    
    # Store current query for relevance scoring
    original_retrieve_documents = RAGQAChain._retrieve_documents
    
    async def enhanced_retrieve_documents(self, question: str, run_manager=None):
        """Enhanced document retrieval that stores query for context"""
        self._current_query = question
        try:
            return await original_retrieve_documents(self, question, run_manager)
        finally:
            delattr(self, '_current_query')
    
    RAGQAChain._retrieve_documents = enhanced_retrieve_documents
    
    logger.info("Successfully patched QA chain for intelligent chunking")


# Factory function for easy integration
def create_context_assembly_integration(
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
    target_chunk_size: int = 1500,
    optimize_for_tokens: bool = True
) -> ContextAssemblyIntegration:
    """Create a context assembly integration instance
    
    Args:
        strategy: Chunking strategy to use
        target_chunk_size: Target size for chunks
        optimize_for_tokens: Whether to optimize for token efficiency
        
    Returns:
        Configured context assembly integration
    """
    config = ChunkingConfig(
        strategy=strategy,
        target_chunk_size=target_chunk_size,
        optimize_for_tokens=optimize_for_tokens
    )
    
    return ContextAssemblyIntegration(config)


if __name__ == "__main__":
    # Test integration functionality
    import asyncio
    
    async def test_integration():
        """Test the integration functionality"""
        
        # Create test documents
        test_docs = [
            Document(
                page_content="""
# API Documentation

This document describes the REST API endpoints for our service.
It provides comprehensive information about authentication, request formats,
and response structures.

## Authentication

All API requests must include an authentication token in the header.
The token can be obtained through the login endpoint.

### Token Format

The authentication token is a JWT with the following structure:
- Header: Contains algorithm and token type
- Payload: Contains user information and expiration
- Signature: Ensures token integrity

## Endpoints

### GET /api/users

Retrieves a list of all users in the system.
""",
                metadata={
                    'source': 'api_docs.md',
                    'file_path': 'docs/api_docs.md',
                    'project': 'main-app'
                }
            ),
            Document(
                page_content="""
```python
def authenticate_user(email: str, password: str) -> Optional[User]:
    '''Authenticate user with email and password'''
    user = get_user_by_email(email)
    if user and user.check_password(password):
        return user
    return None

def generate_token(user: User) -> str:
    '''Generate JWT token for user'''
    payload = {
        'user_id': user.id,
        'email': user.email,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
```
""",
                metadata={
                    'source': 'auth.py',
                    'file_path': 'src/auth.py',
                    'project': 'main-app'
                }
            )
        ]
        
        # Test context assembly integration
        integration = create_context_assembly_integration()
        
        print("Testing context assembly integration...")
        
        # Process documents
        processed_docs = await integration.process_documents_for_context(
            test_docs,
            max_context_tokens=2000,
            query="authentication token generation"
        )
        
        print(f"Original documents: {len(test_docs)}")
        print(f"Processed chunks: {len(processed_docs)}")
        
        for i, doc in enumerate(processed_docs):
            print(f"\nChunk {i+1}:")
            print(f"  Type: {doc.metadata.get('chunk_type', 'unknown')}")
            print(f"  Size: {len(doc.page_content)} chars")
            print(f"  Quality: {doc.metadata.get('quality_score', 0):.3f}")
            print(f"  Content: {doc.page_content[:100]}...")
        
        # Test stats
        stats = integration.get_chunking_stats()
        print(f"\nChunking statistics: {stats}")
        
        print("\nIntegration test completed successfully!")
    
    asyncio.run(test_integration())