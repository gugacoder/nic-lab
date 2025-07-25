"""
GitLab Content Retriever for LangChain

This module implements a custom LangChain retriever that searches GitLab repositories
and wikis to provide relevant content for RAG (Retrieval-Augmented Generation) pipelines.
It supports multiple search strategies, intelligent caching, and optimized content chunking.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from ...integrations.gitlab_client import get_gitlab_client, GitLabSearchResult
from ...integrations.search.keyword_search import KeywordSearchStrategy, create_keyword_search
from ...integrations.search.semantic_search import SemanticSearchStrategy, create_semantic_search
from ...integrations.search.aggregator import SearchResultAggregator, create_result_aggregator
from ...integrations.cache.search_cache import get_search_cache, CacheConfig
from ...ai.preprocessing.content_chunker import ContentChunker, create_content_chunker, ChunkingStrategy
from ...config.settings import get_settings

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search modes for GitLab retriever"""
    KEYWORD_ONLY = "keyword"
    SEMANTIC_ONLY = "semantic"
    HYBRID = "hybrid"
    AUTO = "auto"


@dataclass
class RetrievalConfig:
    """Configuration for GitLab retrieval operations"""
    
    # Basic search configuration
    max_results: int = 50
    search_timeout: int = 10
    file_extensions: List[str] = None
    project_ids: List[int] = None
    include_wikis: bool = True
    
    # Content processing
    content_max_length: int = 8000
    chunk_size: int = 2000
    chunk_overlap: int = 200
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_AWARE
    
    # Search strategy configuration
    search_mode: SearchMode = SearchMode.HYBRID
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Multi-strategy weights (used in hybrid mode)  
    keyword_weight: float = 1.0
    semantic_weight: float = 1.2
    
    # Performance optimization
    enable_parallel_search: bool = True
    enable_result_deduplication: bool = True
    similarity_threshold: float = 0.3
    
    def __post_init__(self):
        if self.file_extensions is None:
            # Default to documentation and code file types
            self.file_extensions = ['md', 'txt', 'py', 'js', 'ts', 'json', 'yaml', 'yml', 'rst', 'adoc']


class GitLabRetriever(BaseRetriever):
    """Custom LangChain retriever for GitLab content
    
    This retriever searches across GitLab repositories and wikis to find relevant
    content for user queries. It supports multiple search strategies (keyword, semantic, hybrid),
    intelligent caching, and optimized content chunking for LLM processing.
    """
    
    def __init__(
        self,
        gitlab_instance: Optional[str] = None,
        config: Optional[RetrievalConfig] = None,
        **kwargs
    ):
        """Initialize the GitLab retriever
        
        Args:
            gitlab_instance: GitLab instance name (uses default if None)
            config: Retrieval configuration
            **kwargs: Additional arguments for BaseRetriever
        """
        super().__init__(**kwargs)
        self.gitlab_client = get_gitlab_client(gitlab_instance)
        self.config = config or RetrievalConfig()
        self.settings = get_settings()
        
        # Apply settings overrides
        if self.settings.gitlab.max_search_results:
            self.config.max_results = min(self.config.max_results, self.settings.gitlab.max_search_results)
        if self.settings.gitlab.search_timeout:
            self.config.search_timeout = self.settings.gitlab.search_timeout
        if self.settings.gitlab.search_file_extensions:
            self.config.file_extensions = self.settings.gitlab.search_file_extensions
        if self.settings.gitlab.accessible_projects:
            self.config.project_ids = self.settings.gitlab.accessible_projects
        
        # Initialize search strategies
        self.keyword_search = create_keyword_search(self.gitlab_client)
        self.semantic_search = create_semantic_search(self.gitlab_client)
        
        # Initialize result aggregator
        self.result_aggregator = create_result_aggregator(
            strategy_weights={
                'keyword': self.config.keyword_weight,
                'semantic': self.config.semantic_weight
            },
            max_results=self.config.max_results
        )
        self.result_aggregator.set_search_strategies(
            keyword_search=self.keyword_search,
            semantic_search=self.semantic_search
        )
        
        # Initialize content chunker
        self.content_chunker = create_content_chunker(
            max_chunk_size=self.config.chunk_size,
            overlap_size=self.config.chunk_overlap,
            strategy=self.config.chunking_strategy
        )
        
        # Initialize cache if enabled
        self.search_cache = None
        if self.config.enable_caching:
            cache_config = CacheConfig(
                default_ttl_seconds=self.config.cache_ttl_seconds,
                max_size_mb=50,  # 50MB cache
                enable_persistence=True
            )
            self.search_cache = get_search_cache(cache_config)
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents for the given query using multi-strategy search
        
        Args:
            query: Search query string
            run_manager: Callback manager for retrieval run
            
        Returns:
            List of relevant documents
        """
        try:
            start_time = datetime.now()
            logger.info(f"Retrieving documents for query: '{query}' using {self.config.search_mode.value} mode")
            
            # Check cache first if enabled
            if self.search_cache:
                cached_results = self.search_cache.get_search_results(
                    query=query,
                    project_ids=self.config.project_ids,
                    file_extensions=self.config.file_extensions,
                    strategy=self.config.search_mode.value
                )
                
                if cached_results:
                    logger.info(f"Using cached results ({len(cached_results)} results)")
                    run_manager.on_text(f"Retrieved {len(cached_results)} cached documents from GitLab")
                    documents = self._process_search_results(cached_results, query)
                    return documents[:self.config.max_results]
            
            # Determine search strategies to use
            strategies = self._determine_search_strategies(query)
            
            # Perform multi-strategy search
            search_results = self.result_aggregator.aggregate_results(
                query=query,
                project_ids=self.config.project_ids,
                file_extensions=self.config.file_extensions,
                strategies=strategies,
                limit=self.config.max_results
            )
            
            # Cache results if caching is enabled
            if self.search_cache and search_results:
                self.search_cache.cache_search_results(
                    query=query,
                    results=search_results,
                    project_ids=self.config.project_ids,
                    file_extensions=self.config.file_extensions,
                    strategy=self.config.search_mode.value,
                    ttl_seconds=self.config.cache_ttl_seconds
                )
            
            # Process results into LangChain documents
            documents = self._process_search_results(search_results, query)
            
            # Apply final ranking and deduplication
            if self.config.enable_result_deduplication:
                documents = self._deduplicate_documents(documents)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Retrieved {len(documents)} documents in {duration:.2f}s using strategies: {strategies}")
            run_manager.on_text(f"Retrieved {len(documents)} documents from GitLab using {', '.join(strategies)} search")
            
            return documents[:self.config.max_results]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            run_manager.on_text(f"Error retrieving documents: {e}")
            return []
    
    def _determine_search_strategies(self, query: str) -> List[str]:
        """Determine which search strategies to use based on query and configuration
        
        Args:
            query: Search query
            
        Returns:
            List of strategy names to use
        """
        if self.config.search_mode == SearchMode.KEYWORD_ONLY:
            return ['keyword']
        elif self.config.search_mode == SearchMode.SEMANTIC_ONLY:
            return ['semantic']
        elif self.config.search_mode == SearchMode.HYBRID:
            return ['keyword', 'semantic']
        elif self.config.search_mode == SearchMode.AUTO:
            # Auto-determine based on query characteristics
            query_lower = query.lower()
            
            # Use semantic search for conceptual queries
            conceptual_indicators = [
                'how to', 'what is', 'explain', 'tutorial', 'guide', 'best practice',
                'concept', 'overview', 'introduction', 'documentation'
            ]
            
            if any(indicator in query_lower for indicator in conceptual_indicators):
                return ['semantic', 'keyword']  # Semantic first for conceptual queries
            
            # Use keyword search for specific technical terms
            if any(char in query for char in ['()', '{', '}', '<', '>', '=']):
                return ['keyword', 'semantic']  # Keyword first for technical queries
            
            # Default to hybrid for balanced results
            return ['keyword', 'semantic']
        
        return ['keyword']  # Fallback
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate or very similar documents
        
        Args:
            documents: List of documents to deduplicate
            
        Returns:
            Deduplicated list of documents
        """
        if not documents:
            return documents
        
        deduplicated = []
        seen_sources = set()
        
        for doc in documents:
            source_key = f"{doc.metadata.get('project_id', 'unknown')}:{doc.metadata.get('file_path', 'unknown')}"
            
            # Skip exact duplicates by source
            if source_key in seen_sources:
                continue
            
            # Check content similarity with existing documents
            is_duplicate = False
            for existing_doc in deduplicated[-5:]:  # Check last 5 docs for efficiency
                similarity = self._calculate_content_similarity(
                    doc.page_content, 
                    existing_doc.page_content
                )
                
                if similarity > 0.9:  # Very similar content
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(doc)
                seen_sources.add(source_key)
        
        logger.debug(f"Deduplicated {len(documents)} -> {len(deduplicated)} documents")
        return deduplicated
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings
        
        Args:
            content1: First content string
            content2: Second content string
            
        Returns:
            Similarity score (0-1)
        """
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based similarity for deduplication
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _search_files(self, query: str) -> List[GitLabSearchResult]:
        """Search GitLab repository files
        
        Args:
            query: Search query
            
        Returns:
            List of search results from files
        """
        try:
            results = self.gitlab_client.search_files(
                query=query,
                project_ids=self.config.project_ids,
                file_extensions=self.config.file_extensions,
                limit=self.config.max_results // 2  # Reserve half for wikis
            )
            logger.debug(f"Found {len(results)} file search results")
            return results
            
        except Exception as e:
            logger.warning(f"Error searching files: {e}")
            return []
    
    def _search_wikis(self, query: str) -> List[GitLabSearchResult]:
        """Search GitLab wiki content
        
        Args:
            query: Search query
            
        Returns:
            List of search results from wikis
        """
        try:
            results = self.gitlab_client.search_wikis(
                query=query,
                project_ids=self.config.project_ids,
                limit=self.config.max_results // 2  # Reserve half for files
            )
            logger.debug(f"Found {len(results)} wiki search results")
            return results
            
        except Exception as e:
            logger.warning(f"Error searching wikis: {e}")
            return []
    
    def _process_search_results(
        self,
        search_results: List[GitLabSearchResult],
        query: str
    ) -> List[Document]:
        """Convert GitLab search results to LangChain documents with intelligent chunking
        
        Args:
            search_results: Raw search results from GitLab
            query: Original search query
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for result in search_results:
            try:
                # Get full file content for better context
                full_content = self._get_full_content(result)
                
                if not full_content:
                    # Fallback to search result content
                    full_content = result.content
                
                # Use intelligent content chunking
                source_metadata = {
                    'project_id': result.project_id,
                    'project_name': result.project_name,
                    'file_path': result.file_path,
                    'ref': result.ref,
                    'is_wiki': result.wiki
                }
                
                # Detect content type for optimal chunking
                content_type = self._detect_content_type(result.file_path, full_content)
                
                # Chunk content using appropriate strategy
                chunks = self.content_chunker.chunk_content(
                    content=full_content,
                    content_type=content_type,
                    source_metadata=source_metadata
                )
                
                for chunk in chunks:
                    if not chunk.content.strip():
                        continue
                    
                    # Create enhanced document metadata
                    metadata = {
                        'source': f"{result.project_name}/{result.file_path}",
                        'project_id': result.project_id,
                        'project_name': result.project_name,
                        'file_path': result.file_path,
                        'ref': result.ref,
                        'start_line': result.startline,
                        'is_wiki': result.wiki,
                        'chunk_index': chunk.chunk_index,
                        'total_chunks': chunk.metadata.get('total_chunks', 1),
                        'chunk_type': chunk.chunk_type,
                        'chunk_size': chunk.size,
                        'word_count': chunk.word_count,
                        'query': query,
                        'retrieved_at': datetime.now().isoformat(),
                        'content_type': content_type,
                        'chunking_strategy': self.config.chunking_strategy.value
                    }
                    
                    # Add chunk-specific metadata
                    metadata.update(chunk.metadata)
                    
                    # Add web URL if possible
                    if hasattr(result, 'web_url'):
                        metadata['web_url'] = result.web_url
                    
                    document = Document(
                        page_content=chunk.content,
                        metadata=metadata
                    )
                    documents.append(document)
                    
            except Exception as e:
                logger.warning(f"Error processing search result {result.file_path}: {e}")
                continue
        
        logger.debug(f"Processed {len(search_results)} search results into {len(documents)} document chunks")
        return documents
    
    def _detect_content_type(self, file_path: str, content: str) -> str:
        """Detect content type for optimal processing
        
        Args:
            file_path: File path
            content: File content
            
        Returns:
            Detected content type
        """
        # Extract file extension
        file_ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
        
        # Check file extension first
        if file_ext in ['md', 'markdown']:
            return 'markdown'
        elif file_ext in ['py', 'js', 'ts', 'java', 'cpp', 'c', 'go', 'rs', 'rb']:
            return 'code'
        elif file_ext in ['txt', 'rst', 'adoc']:
            return 'text'
        elif file_ext in ['json', 'yaml', 'yml', 'xml']:
            return 'structured_data'
        
        # Fallback to content analysis
        return self.content_chunker._detect_content_type(content)
    
    def _get_full_content(self, result: GitLabSearchResult) -> Optional[str]:
        """Get full content of a file for better context
        
        Args:
            result: GitLab search result
            
        Returns:
            Full file content if available, None otherwise
        """
        try:
            # For wiki results, use the search content as-is
            if result.wiki:
                return result.content
            
            # For file results, try to get the full file content
            file_obj = self.gitlab_client.get_file(
                project_id=result.project_id,
                file_path=result.file_path,
                ref=result.ref
            )
            
            if file_obj and file_obj.is_text:
                # Limit content length to prevent token overflow
                content = file_obj.content
                if len(content) > self.config.content_max_length:
                    content = content[:self.config.content_max_length] + "\n\n[Content truncated...]"
                return content
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not retrieve full content for {result.file_path}: {e}")
            return None
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if caching is enabled
        
        Returns:
            Cache statistics or None if caching disabled
        """
        if self.search_cache:
            return self.search_cache.get_stats()
        return None
    
    def clear_cache(self, project_id: Optional[int] = None):
        """Clear cache entries
        
        Args:
            project_id: Optional project ID to clear cache for specific project
        """
        if self.search_cache:
            if project_id:
                self.search_cache.invalidate_project(project_id)
            else:
                self.search_cache.clear()
    
    def update_search_mode(self, mode: SearchMode):
        """Update search mode dynamically
        
        Args:
            mode: New search mode to use
        """
        self.config.search_mode = mode
        logger.info(f"Updated search mode to: {mode.value}")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze retriever performance
        
        Returns:
            Performance analysis results
        """
        analysis = {
            'config': {
                'search_mode': self.config.search_mode.value,
                'max_results': self.config.max_results,
                'caching_enabled': self.config.enable_caching,
                'chunking_strategy': self.config.chunking_strategy.value
            }
        }
        
        # Add cache statistics if available
        if self.search_cache:
            cache_stats = self.search_cache.get_stats()
            analysis['cache_performance'] = {
                'hit_rate': cache_stats['hit_rate'],
                'total_entries': cache_stats['total_entries'],
                'size_mb': cache_stats['size_mb']
            }
            
            # Get cache analysis
            cache_analysis = self.search_cache.analyze_cache()
            analysis['cache_analysis'] = cache_analysis
        
        return analysis
    
    def test_retrieval(self, query: str) -> Dict[str, Any]:
        """Test the enhanced retrieval system with a sample query
        
        Args:
            query: Test query
            
        Returns:
            Test results and statistics
        """
        start_time = datetime.now()
        
        try:
            # Test connection first
            success, message = self.gitlab_client.test_connection()
            if not success:
                return {
                    'success': False,
                    'error': f"GitLab connection failed: {message}",
                    'query': query,
                    'duration_ms': 0
                }
            
            # Create a dummy callback manager for testing
            class DummyCallbackManager:
                def on_text(self, text: str):
                    pass
            
            # Perform retrieval
            documents = self._get_relevant_documents(query, run_manager=DummyCallbackManager())
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() * 1000
            
            # Analyze search strategies used
            strategies_used = self._determine_search_strategies(query)
            
            # Get performance analysis
            performance_analysis = self.analyze_performance()
            
            return {
                'success': True,
                'query': query,
                'documents_found': len(documents),
                'duration_ms': duration,
                'strategies_used': strategies_used,
                'search_mode': self.config.search_mode.value,
                'sources': [doc.metadata.get('source') for doc in documents],
                'sample_content': documents[0].page_content[:200] + "..." if documents else None,
                'chunk_info': {
                    'total_chunks': len(documents),
                    'chunking_strategy': self.config.chunking_strategy.value,
                    'avg_chunk_size': sum(doc.metadata.get('chunk_size', 0) for doc in documents) / len(documents) if documents else 0
                },
                'performance_analysis': performance_analysis,
                'config': {
                    'max_results': self.config.max_results,
                    'file_extensions': self.config.file_extensions,
                    'include_wikis': self.config.include_wikis,
                    'caching_enabled': self.config.enable_caching,
                    'search_timeout': self.config.search_timeout
                }
            }
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() * 1000
            
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'duration_ms': duration
            }


def create_gitlab_retriever(
    gitlab_instance: Optional[str] = None,
    max_results: int = 20,
    include_wikis: bool = True,
    file_extensions: Optional[List[str]] = None,
    project_ids: Optional[List[int]] = None,
    search_mode: SearchMode = SearchMode.HYBRID,
    enable_caching: bool = True,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_AWARE,
    chunk_size: int = 2000,
    chunk_overlap: int = 200
) -> GitLabRetriever:
    """Factory function to create a configured GitLab retriever with enhanced features
    
    Args:
        gitlab_instance: GitLab instance name
        max_results: Maximum number of documents to retrieve
        include_wikis: Whether to search wiki content
        file_extensions: File extensions to search
        project_ids: Specific project IDs to search
        search_mode: Search strategy mode (keyword, semantic, hybrid, auto)
        enable_caching: Whether to enable search result caching
        chunking_strategy: Strategy for content chunking
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        Configured GitLab retriever with enhanced capabilities
    """
    config = RetrievalConfig(
        max_results=max_results,
        file_extensions=file_extensions,
        project_ids=project_ids,
        include_wikis=include_wikis,
        search_mode=search_mode,
        enable_caching=enable_caching,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return GitLabRetriever(
        gitlab_instance=gitlab_instance,
        config=config
    )


if __name__ == "__main__":
    # Test the enhanced GitLab retriever
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.ai.retrievers.gitlab_retriever <test_query> [search_mode]")
        print("Search modes: keyword, semantic, hybrid, auto")
        sys.exit(1)
    
    query = sys.argv[1]
    search_mode_str = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    
    # Parse search mode
    try:
        search_mode = SearchMode(search_mode_str)
    except ValueError:
        print(f"Invalid search mode: {search_mode_str}")
        print("Valid modes: keyword, semantic, hybrid, auto")
        sys.exit(1)
    
    print(f"Testing enhanced GitLab retriever with query: '{query}'")
    print(f"Search mode: {search_mode.value}")
    
    # Create retriever with specified configuration
    retriever = create_gitlab_retriever(
        max_results=10,
        search_mode=search_mode,
        enable_caching=True,
        chunking_strategy=ChunkingStrategy.SENTENCE_AWARE
    )
    
    # Run test
    results = retriever.test_retrieval(query)
    
    print(f"\nTest Results:")
    print(f"  Success: {results['success']}")
    print(f"  Duration: {results['duration_ms']:.1f}ms")
    
    if results['success']:
        print(f"  Search mode used: {results['search_mode']}")
        print(f"  Strategies used: {results['strategies_used']}")
        print(f"  Documents found: {results['documents_found']}")
        print(f"  Chunking strategy: {results['chunk_info']['chunking_strategy']}")
        print(f"  Total chunks: {results['chunk_info']['total_chunks']}")
        print(f"  Average chunk size: {results['chunk_info']['avg_chunk_size']:.0f} chars")
        
        # Show cache performance if available
        if 'cache_performance' in results['performance_analysis']:
            cache_perf = results['performance_analysis']['cache_performance']
            print(f"  Cache hit rate: {cache_perf['hit_rate']:.2%}")
            print(f"  Cache entries: {cache_perf['total_entries']}")
        
        print(f"  Sources: {results['sources'][:3]}")  # Show first 3
        if results['sample_content']:
            print(f"  Sample content: {results['sample_content']}")
    else:
        print(f"  Error: {results['error']}")
    
    print("\nEnhanced GitLab retriever testing complete")