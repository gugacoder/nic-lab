"""
GitLab Content Retriever for LangChain

This module implements a custom LangChain retriever that searches GitLab repositories
and wikis to provide relevant content for RAG (Retrieval-Augmented Generation) pipelines.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from ...integrations.gitlab_client import get_gitlab_client, GitLabSearchResult
from ...config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for GitLab retrieval operations"""
    
    max_results: int = 50
    search_timeout: int = 10
    file_extensions: List[str] = None
    project_ids: List[int] = None
    include_wikis: bool = True
    content_max_length: int = 8000
    chunk_size: int = 2000
    chunk_overlap: int = 200
    
    def __post_init__(self):
        if self.file_extensions is None:
            # Default to documentation and code file types
            self.file_extensions = ['md', 'txt', 'py', 'js', 'ts', 'json', 'yaml', 'yml', 'rst', 'adoc']


class GitLabRetriever(BaseRetriever):
    """Custom LangChain retriever for GitLab content
    
    This retriever searches across GitLab repositories and wikis to find relevant
    content for user queries. It supports multiple search strategies and formats
    results as LangChain Documents for use in RAG pipelines.
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
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents for the given query
        
        Args:
            query: Search query string
            run_manager: Callback manager for retrieval run
            
        Returns:
            List of relevant documents
        """
        try:
            logger.info(f"Retrieving documents for query: '{query}'")
            
            # Perform parallel searches
            file_results = self._search_files(query)
            wiki_results = []
            
            if self.config.include_wikis:
                wiki_results = self._search_wikis(query)
            
            # Combine and process results
            all_results = file_results + wiki_results
            documents = self._process_search_results(all_results, query)
            
            # Sort by relevance and limit results
            documents = self._rank_documents(documents, query)[:self.config.max_results]
            
            logger.info(f"Retrieved {len(documents)} documents")
            run_manager.on_text(f"Retrieved {len(documents)} documents from GitLab")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            run_manager.on_text(f"Error retrieving documents: {e}")
            return []
    
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
        """Convert GitLab search results to LangChain documents
        
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
                
                # Chunk content if it's too long
                chunks = self._chunk_content(full_content)
                
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    
                    # Create document metadata
                    metadata = {
                        'source': f"{result.project_name}/{result.file_path}",
                        'project_id': result.project_id,
                        'project_name': result.project_name,
                        'file_path': result.file_path,
                        'ref': result.ref,
                        'start_line': result.startline,
                        'is_wiki': result.wiki,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'query': query,
                        'retrieved_at': datetime.now().isoformat()
                    }
                    
                    # Add web URL if possible
                    if hasattr(result, 'web_url'):
                        metadata['web_url'] = result.web_url
                    
                    document = Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                    documents.append(document)
                    
            except Exception as e:
                logger.warning(f"Error processing search result {result.file_path}: {e}")
                continue
        
        return documents
    
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
    
    def _chunk_content(self, content: str) -> List[str]:
        """Split content into manageable chunks for LLM processing
        
        Args:
            content: Full content to chunk
            
        Returns:
            List of content chunks
        """
        if len(content) <= self.config.chunk_size:
            return [content]
        
        chunks = []
        
        # Simple chunking by lines with overlap
        lines = content.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length > self.config.chunk_size and current_chunk:
                # Create chunk and start new one with overlap
                chunk_text = '\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap (last few lines)
                overlap_lines = max(1, self.config.chunk_overlap // 50)  # Rough estimate
                current_chunk = current_chunk[-overlap_lines:]
                current_length = sum(len(l) + 1 for l in current_chunk)
            
            current_chunk.append(line)
            current_length += line_length
        
        # Add final chunk if there's content
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _rank_documents(self, documents: List[Document], query: str) -> List[Document]:
        """Rank documents by relevance to query
        
        Args:
            documents: List of documents to rank
            query: Original search query
            
        Returns:
            Documents sorted by relevance (best first)
        """
        # Simple relevance scoring based on query term frequency and document properties
        query_terms = set(query.lower().split())
        
        def calculate_relevance(doc: Document) -> float:
            content = doc.page_content.lower()
            score = 0.0
            
            # Term frequency scoring
            for term in query_terms:
                score += content.count(term) * 1.0
            
            # Boost for certain file types
            file_path = doc.metadata.get('file_path', '').lower()
            if file_path.endswith(('.md', '.txt', '.rst')):
                score *= 1.2  # Boost documentation files
            elif file_path.endswith('.py'):
                score *= 1.1  # Slight boost for Python files
            
            # Boost for wiki content
            if doc.metadata.get('is_wiki', False):
                score *= 1.15
            
            # Penalize very short content
            if len(doc.page_content) < 100:
                score *= 0.8
            
            # Boost for files with query terms in filename
            for term in query_terms:
                if term in file_path:
                    score *= 1.3
                    break
            
            return score
        
        # Sort by relevance score
        scored_docs = [(doc, calculate_relevance(doc)) for doc in documents]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs]
    
    def test_retrieval(self, query: str) -> Dict[str, Any]:
        """Test the retrieval system with a sample query
        
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
            
            # Perform retrieval
            documents = self._get_relevant_documents(query, run_manager=None)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() * 1000
            
            return {
                'success': True,
                'query': query,
                'documents_found': len(documents),
                'duration_ms': duration,
                'sources': [doc.metadata.get('source') for doc in documents],
                'sample_content': documents[0].page_content[:200] + "..." if documents else None,
                'config': {
                    'max_results': self.config.max_results,
                    'file_extensions': self.config.file_extensions,
                    'include_wikis': self.config.include_wikis
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
    project_ids: Optional[List[int]] = None
) -> GitLabRetriever:
    """Factory function to create a configured GitLab retriever
    
    Args:
        gitlab_instance: GitLab instance name
        max_results: Maximum number of documents to retrieve
        include_wikis: Whether to search wiki content
        file_extensions: File extensions to search
        project_ids: Specific project IDs to search
        
    Returns:
        Configured GitLab retriever
    """
    config = RetrievalConfig(
        max_results=max_results,
        file_extensions=file_extensions,
        project_ids=project_ids,
        include_wikis=include_wikis
    )
    
    return GitLabRetriever(
        gitlab_instance=gitlab_instance,
        config=config
    )


if __name__ == "__main__":
    # Test the GitLab retriever
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.ai.retrievers.gitlab_retriever <test_query>")
        sys.exit(1)
    
    query = ' '.join(sys.argv[1:])
    
    print(f"Testing GitLab retriever with query: '{query}'")
    
    retriever = create_gitlab_retriever(max_results=10)
    results = retriever.test_retrieval(query)
    
    print(f"Test Results:")
    print(f"  Success: {results['success']}")
    print(f"  Duration: {results['duration_ms']:.1f}ms")
    
    if results['success']:
        print(f"  Documents found: {results['documents_found']}")
        print(f"  Sources: {results['sources'][:5]}")  # Show first 5
        if results['sample_content']:
            print(f"  Sample content: {results['sample_content']}")
    else:
        print(f"  Error: {results['error']}")