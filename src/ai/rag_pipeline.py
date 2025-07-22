"""
RAG Pipeline Orchestrator for NIC Chat System

This module provides the main entry point for the Retrieval-Augmented Generation
pipeline, orchestrating GitLab content retrieval, conversation memory, and LLM inference.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import asynccontextmanager

from .retrievers.gitlab_retriever import GitLabRetriever, create_gitlab_retriever
from .memory.conversation_memory import ConversationMemory, create_conversation_memory, get_global_memory_store
from .prompts.templates import PromptManager, PromptContext, get_prompt_manager
from .chains.qa_chain import RAGQAChain, create_qa_chain, QAChainResult
from .groq_client import GroqClient
from .postprocessing.response_formatter import ResponseFormatter
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    
    # Retrieval settings
    max_sources: int = 10
    include_wikis: bool = True
    file_extensions: Optional[List[str]] = None
    project_ids: Optional[List[int]] = None
    
    # LLM settings
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # Memory settings
    max_conversation_turns: int = 5
    session_ttl_hours: int = 24
    
    # Performance settings
    max_context_length: int = 8000
    timeout_seconds: int = 30
    enable_caching: bool = True
    
    # Response settings
    enable_streaming: bool = False
    format_response: bool = True
    include_sources: bool = True


@dataclass
class RAGRequest:
    """RAG pipeline request"""
    
    query: str
    session_id: Optional[str] = None
    config_override: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RAGResponse:
    """RAG pipeline response"""
    
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: str
    execution_time_ms: float
    token_usage: Dict[str, int]
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'answer': self.answer,
            'sources': self.sources,
            'conversation_id': self.conversation_id,
            'execution_time_ms': self.execution_time_ms,
            'token_usage': self.token_usage,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class RAGPipeline:
    """
    Main RAG Pipeline Orchestrator
    
    Coordinates all components of the RAG system:
    - Document retrieval from GitLab
    - Conversation memory management  
    - Prompt generation and LLM inference
    - Response post-processing and formatting
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        gitlab_instance: Optional[str] = None
    ):
        """Initialize RAG pipeline
        
        Args:
            config: Pipeline configuration
            gitlab_instance: GitLab instance name
        """
        self.config = config or RAGConfig()
        self.gitlab_instance = gitlab_instance
        self.settings = get_settings()
        
        # Initialize components
        self._retriever: Optional[GitLabRetriever] = None
        self._llm_client: Optional[GroqClient] = None
        self._prompt_manager: Optional[PromptManager] = None
        self._response_formatter: Optional[ResponseFormatter] = None
        self._memory_store = get_global_memory_store()
        
        # Performance tracking
        self._stats = {
            'requests_processed': 0,
            'total_tokens_used': 0,
            'total_execution_time': 0.0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'errors': 0
        }
        
        logger.info("RAG pipeline initialized")
    
    async def initialize(self):
        """Initialize pipeline components asynchronously"""
        logger.info("Initializing RAG pipeline components...")
        
        # Initialize retriever
        self._retriever = create_gitlab_retriever(
            gitlab_instance=self.gitlab_instance,
            max_results=self.config.max_sources,
            include_wikis=self.config.include_wikis,
            file_extensions=self.config.file_extensions,
            project_ids=self.config.project_ids
        )
        
        # Initialize LLM client
        self._llm_client = GroqClient()
        
        # Test connection
        connection_ok = await self._llm_client.test_connection()
        if not connection_ok:
            raise RuntimeError("Failed to connect to Groq API")
        
        # Initialize other components
        self._prompt_manager = get_prompt_manager()
        self._response_formatter = ResponseFormatter()
        
        logger.info("RAG pipeline initialization completed")
    
    async def process_query(
        self,
        request: Union[RAGRequest, str],
        stream: bool = False
    ) -> Union[RAGResponse, AsyncGenerator[str, None]]:
        """Process a query through the RAG pipeline
        
        Args:
            request: RAG request or query string
            stream: Whether to stream the response
            
        Returns:
            RAG response or async generator for streaming
        """
        # Normalize request
        if isinstance(request, str):
            request = RAGRequest(query=request)
        
        start_time = time.time()
        
        try:
            # Ensure pipeline is initialized
            if not self._retriever or not self._llm_client:
                await self.initialize()
            
            # Generate session ID if not provided
            session_id = request.session_id or self._generate_session_id()
            
            # Apply config overrides
            effective_config = self._merge_config(request.config_override)
            
            if stream:
                return self._process_query_streaming(request, session_id, effective_config)
            else:
                return await self._process_query_sync(request, session_id, effective_config, start_time)
                
        except Exception as e:
            self._stats['errors'] += 1
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"RAG pipeline error: {e}", exc_info=True)
            
            # Return error response
            return RAGResponse(
                answer=f"I apologize, but I encountered an error processing your request: {str(e)}",
                sources=[],
                conversation_id=request.session_id or 'error',
                execution_time_ms=execution_time,
                token_usage={},
                metadata={'error': str(e), 'error_type': type(e).__name__},
                timestamp=datetime.now()
            )
    
    async def _process_query_sync(
        self,
        request: RAGRequest,
        session_id: str,
        config: RAGConfig,
        start_time: float
    ) -> RAGResponse:
        """Process query synchronously (non-streaming)"""
        logger.info(f"Processing query: '{request.query}' (session: {session_id})")
        
        # Create QA chain for this request
        qa_chain = create_qa_chain(
            gitlab_instance=self.gitlab_instance,
            session_id=session_id,
            max_sources=config.max_sources,
            include_wikis=config.include_wikis,
            max_context_length=config.max_context_length
        )
        
        # Execute the chain
        chain_input = {
            'question': request.query,
            'session_id': session_id
        }
        
        # Add timeout
        try:
            result = await asyncio.wait_for(
                qa_chain._acall(chain_input),
                timeout=config.timeout_seconds
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Query processing timed out after {config.timeout_seconds}s")
        
        # Format response if enabled
        answer = result['answer']
        if config.format_response and self._response_formatter:
            answer = await self._response_formatter.format_response(
                answer,
                result.get('source_documents', []),
                include_sources=config.include_sources
            )
        
        # Prepare source information
        sources = []
        if config.include_sources and 'source_documents' in result:
            sources = [
                {
                    'source': doc.metadata.get('source', 'Unknown'),
                    'project': doc.metadata.get('project_name', 'Unknown'),
                    'file_path': doc.metadata.get('file_path', 'Unknown'),
                    'relevance': doc.metadata.get('relevance_score', 0.0)
                }
                for doc in result['source_documents']
            ]
        
        # Calculate execution time and update stats
        execution_time = (time.time() - start_time) * 1000
        self._update_stats(result, execution_time)
        
        # Create response
        response = RAGResponse(
            answer=answer,
            sources=sources,
            conversation_id=session_id,
            execution_time_ms=execution_time,
            token_usage=result.get('token_usage', {}),
            metadata={
                'query': request.query,
                'sources_count': len(sources),
                'chain_metadata': result.get('metadata', {}),
                'config': asdict(config)
            },
            timestamp=datetime.now()
        )
        
        logger.info(f"Query processed successfully in {execution_time:.1f}ms")
        return response
    
    async def _process_query_streaming(
        self,
        request: RAGRequest,
        session_id: str,
        config: RAGConfig
    ) -> AsyncGenerator[str, None]:
        """Process query with streaming response"""
        
        async def _stream():
            try:
                # Note: This is a simplified streaming implementation
                # In a production system, you'd want to stream tokens as they're generated
                response = await self._process_query_sync(
                    request, session_id, config, time.time()
                )
                
                # Simulate streaming by yielding chunks
                answer = response.answer
                chunk_size = 50
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i:i + chunk_size]
                    yield chunk
                    await asyncio.sleep(0.1)  # Simulate streaming delay
                    
            except Exception as e:
                yield f"Error: {str(e)}"
        
        return _stream()
    
    def _merge_config(self, overrides: Optional[Dict[str, Any]]) -> RAGConfig:
        """Merge configuration overrides with default config"""
        if not overrides:
            return self.config
        
        config_dict = asdict(self.config)
        config_dict.update(overrides)
        return RAGConfig(**config_dict)
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        import uuid
        return f"session_{uuid.uuid4().hex[:12]}"
    
    def _update_stats(self, result: Dict[str, Any], execution_time: float):
        """Update pipeline performance statistics"""
        self._stats['requests_processed'] += 1
        self._stats['total_execution_time'] += execution_time
        self._stats['average_response_time'] = (
            self._stats['total_execution_time'] / self._stats['requests_processed']
        )
        
        token_usage = result.get('token_usage', {})
        self._stats['total_tokens_used'] += token_usage.get('total_tokens', 0)
        
        if result.get('cached', False):
            self._stats['cache_hits'] += 1
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_info = {
            'status': 'unknown',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            # Check retriever
            if self._retriever:
                test_results = self._retriever.test_retrieval("test query")
                health_info['components']['retriever'] = {
                    'status': 'healthy' if test_results['success'] else 'unhealthy',
                    'response_time_ms': test_results['duration_ms']
                }
            else:
                health_info['components']['retriever'] = {'status': 'not_initialized'}
            
            # Check LLM client
            if self._llm_client:
                llm_health = await self._llm_client.health_check()
                health_info['components']['llm'] = llm_health
            else:
                health_info['components']['llm'] = {'status': 'not_initialized'}
            
            # Check memory store
            memory_stats = self._memory_store.get_session_stats()
            health_info['components']['memory'] = {
                'status': 'healthy',
                **memory_stats
            }
            
            # Overall status
            component_statuses = [
                comp.get('status', 'unknown') 
                for comp in health_info['components'].values()
            ]
            
            if all(status == 'healthy' for status in component_statuses):
                health_info['status'] = 'healthy'
            elif any(status == 'unhealthy' for status in component_statuses):
                health_info['status'] = 'unhealthy'
            else:
                health_info['status'] = 'degraded'
            
            # Add pipeline stats
            health_info['pipeline_stats'] = dict(self._stats)
            
        except Exception as e:
            health_info['status'] = 'error'
            health_info['error'] = str(e)
        
        return health_info
    
    async def cleanup(self):
        """Cleanup pipeline resources"""
        logger.info("Cleaning up RAG pipeline resources...")
        
        if self._llm_client:
            await self._llm_client.close()
        
        logger.info("RAG pipeline cleanup completed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        return dict(self._stats)


# Global pipeline instance
_pipeline_instance: Optional[RAGPipeline] = None


@asynccontextmanager
async def get_rag_pipeline(
    config: Optional[RAGConfig] = None,
    gitlab_instance: Optional[str] = None
):
    """Get RAG pipeline instance with context manager"""
    pipeline = RAGPipeline(config, gitlab_instance)
    await pipeline.initialize()
    try:
        yield pipeline
    finally:
        await pipeline.cleanup()


def create_rag_pipeline(
    config: Optional[RAGConfig] = None,
    gitlab_instance: Optional[str] = None
) -> RAGPipeline:
    """Create a new RAG pipeline instance"""
    return RAGPipeline(config, gitlab_instance)


async def process_query(
    query: str,
    session_id: Optional[str] = None,
    config: Optional[RAGConfig] = None,
    gitlab_instance: Optional[str] = None
) -> RAGResponse:
    """Convenience function to process a single query"""
    async with get_rag_pipeline(config, gitlab_instance) as pipeline:
        request = RAGRequest(query=query, session_id=session_id)
        return await pipeline.process_query(request)


if __name__ == "__main__":
    # Test RAG pipeline functionality
    import sys
    
    async def test_pipeline():
        if len(sys.argv) < 2:
            print("Usage: python -m src.ai.rag_pipeline <test_query> [session_id]")
            return
        
        query = ' '.join(sys.argv[1:-1]) if len(sys.argv) > 2 else ' '.join(sys.argv[1:])
        session_id = sys.argv[-1] if len(sys.argv) > 2 and len(sys.argv[-1].split()) == 1 else None
        
        print(f"Testing RAG pipeline with query: '{query}'")
        if session_id:
            print(f"Session ID: {session_id}")
        
        try:
            response = await process_query(query, session_id)
            
            print(f"\nAnswer: {response.answer}")
            print(f"Sources: {len(response.sources)}")
            for i, source in enumerate(response.sources[:3]):
                print(f"  {i+1}. {source['source']}")
            print(f"Execution time: {response.execution_time_ms:.1f}ms")
            print(f"Token usage: {response.token_usage}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Test health check
    async def test_health():
        print("Testing RAG pipeline health check...")
        
        async with get_rag_pipeline() as pipeline:
            health = await pipeline.health_check()
            print(f"Overall status: {health['status']}")
            
            for component, status in health['components'].items():
                print(f"  {component}: {status.get('status', 'unknown')}")
            
            if 'pipeline_stats' in health:
                stats = health['pipeline_stats']
                print(f"Pipeline stats: {stats['requests_processed']} requests processed")
    
    # Run appropriate test
    if len(sys.argv) > 1 and sys.argv[1] == "test-health":
        asyncio.run(test_health())
    else:
        asyncio.run(test_pipeline())