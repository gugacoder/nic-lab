"""
Question-Answering Chain for RAG Pipeline

This module implements the LangChain QA chain that orchestrates retrieval,
prompt generation, and LLM inference for the NIC Chat system.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from datetime import datetime
import time

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.chains.base import Chain

from ..groq_client import GroqClient
from ..retrievers.gitlab_retriever import GitLabRetriever, create_gitlab_retriever
from ..memory.conversation_memory import ConversationMemory, get_global_memory_store
from ..prompts.templates import PromptManager, PromptContext, get_prompt_manager
from ...config.settings import get_settings

logger = logging.getLogger(__name__)


class QAChainError(Exception):
    """Custom exception for QA chain errors"""
    pass


class QAChainResult:
    """Result container for QA chain execution"""
    
    def __init__(
        self,
        answer: str,
        source_documents: List[Document],
        execution_time_ms: float,
        token_usage: Dict[str, int],
        metadata: Dict[str, Any]
    ):
        self.answer = answer
        self.source_documents = source_documents
        self.execution_time_ms = execution_time_ms
        self.token_usage = token_usage
        self.metadata = metadata
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'answer': self.answer,
            'source_documents': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                } for doc in self.source_documents
            ],
            'execution_time_ms': self.execution_time_ms,
            'token_usage': self.token_usage,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class RAGQAChain(Chain):
    """Retrieval-Augmented Generation Question Answering Chain
    
    This chain implements the complete RAG pipeline:
    1. Query understanding and intent detection
    2. Document retrieval from GitLab
    3. Context assembly and prompt generation
    4. LLM inference with Groq API
    5. Response post-processing and source attribution
    """
    
    input_key: str = "question"
    output_key: str = "answer"
    
    def __init__(
        self,
        retriever: GitLabRetriever,
        llm_client: GroqClient,
        memory: Optional[ConversationMemory] = None,
        prompt_manager: Optional[PromptManager] = None,
        max_context_length: int = 8000,
        min_sources: int = 1,
        fallback_on_no_sources: bool = True,
        **kwargs
    ):
        """Initialize the RAG QA Chain
        
        Args:
            retriever: GitLab document retriever
            llm_client: Groq API client for LLM inference
            memory: Conversation memory (optional)
            prompt_manager: Prompt template manager
            max_context_length: Maximum context length for LLM
            min_sources: Minimum required sources for answering
            fallback_on_no_sources: Whether to answer without sources
            **kwargs: Additional Chain arguments
        """
        super().__init__(**kwargs)
        self.retriever = retriever
        self.llm_client = llm_client
        self.memory = memory
        self.prompt_manager = prompt_manager or get_prompt_manager()
        self.max_context_length = max_context_length
        self.min_sources = min_sources
        self.fallback_on_no_sources = fallback_on_no_sources
        self.settings = get_settings()
    
    @property
    def input_keys(self) -> List[str]:
        """Input keys required by this chain"""
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        """Output keys provided by this chain"""
        return [self.output_key]
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, Any]:
        """Execute the QA chain synchronously
        
        Args:
            inputs: Input dictionary containing the question
            run_manager: Callback manager for chain run
            
        Returns:
            Dictionary containing the answer and metadata
        """
        # Use asyncio to run the async method
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in an async context, create a new event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self._acall_internal(inputs, run_manager))
                )
                return future.result()
        else:
            return loop.run_until_complete(self._acall_internal(inputs, run_manager))
    
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None
    ) -> Dict[str, Any]:
        """Execute the QA chain asynchronously
        
        Args:
            inputs: Input dictionary containing the question
            run_manager: Callback manager for chain run
            
        Returns:
            Dictionary containing the answer and metadata
        """
        return await self._acall_internal(inputs, run_manager)
    
    async def _acall_internal(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Internal async call implementation"""
        start_time = time.time()
        
        try:
            question = inputs[self.input_key]
            session_id = inputs.get('session_id', 'default')
            
            if run_manager:
                run_manager.on_text(f"Processing question: {question}")
            
            # Step 1: Retrieve relevant documents
            logger.info(f"Retrieving documents for question: '{question}'")
            documents = await self._retrieve_documents(question, run_manager)
            
            # Step 2: Load conversation memory if available
            conversation_context = ""
            if self.memory:
                try:
                    memory_vars = self.memory.load_memory_variables({})
                    if 'chat_history' in memory_vars and memory_vars['chat_history']:
                        conversation_context = self._format_conversation_history(
                            memory_vars['chat_history']
                        )
                except Exception as e:
                    logger.warning(f"Error loading conversation memory: {e}")
            
            # Step 3: Check if we have sufficient sources
            if len(documents) < self.min_sources and not self.fallback_on_no_sources:
                raise QAChainError(
                    f"Insufficient sources found: {len(documents)} < {self.min_sources}"
                )
            
            # Step 4: Generate prompt with context
            prompt_context = PromptContext(
                user_query=question,
                retrieved_documents=[doc.page_content for doc in documents],
                source_names=[doc.metadata.get('source', 'Unknown') for doc in documents],
                conversation_history=conversation_context
            )
            
            # Step 5: Generate response using LLM
            logger.info("Generating response with Groq LLM")
            if run_manager:
                run_manager.on_text("Generating AI response...")
            
            response_result = await self._generate_response(prompt_context, run_manager)
            
            # Step 6: Save to conversation memory if available
            if self.memory:
                try:
                    self.memory.save_context(
                        inputs={'input': question},
                        outputs={'output': response_result['answer']}
                    )
                except Exception as e:
                    logger.warning(f"Error saving to conversation memory: {e}")
            
            # Step 7: Prepare final result
            execution_time = (time.time() - start_time) * 1000
            
            result = QAChainResult(
                answer=response_result['answer'],
                source_documents=documents,
                execution_time_ms=execution_time,
                token_usage=response_result.get('token_usage', {}),
                metadata={
                    'question': question,
                    'session_id': session_id,
                    'sources_count': len(documents),
                    'has_conversation_context': bool(conversation_context),
                    'prompt_intent': prompt_context.query_intent,
                    'retrieval_config': getattr(self.retriever, 'config', None)
                }
            )
            
            logger.info(f"QA chain completed in {execution_time:.1f}ms")
            if run_manager:
                run_manager.on_text(f"Response generated in {execution_time:.1f}ms")
            
            return {
                self.output_key: result.answer,
                'source_documents': result.source_documents,
                'execution_time_ms': result.execution_time_ms,
                'token_usage': result.token_usage,
                'metadata': result.metadata
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_message = f"QA chain error: {str(e)}"
            logger.error(error_message, exc_info=True)
            
            if run_manager:
                run_manager.on_text(f"Error: {str(e)}")
            
            # Return error response
            return {
                self.output_key: f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'source_documents': [],
                'execution_time_ms': execution_time,
                'token_usage': {},
                'metadata': {
                    'error': str(e),
                    'error_type': type(e).__name__
                }
            }
    
    async def _retrieve_documents(
        self,
        question: str,
        run_manager: Optional[Any] = None
    ) -> List[Document]:
        """Retrieve relevant documents for the question
        
        Args:
            question: User question
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        try:
            if run_manager:
                run_manager.on_text("Searching GitLab repositories...")
            
            # Use the retriever to get relevant documents
            documents = self.retriever._get_relevant_documents(question, run_manager=run_manager)
            
            # Truncate documents to fit context window
            documents = self._truncate_documents(documents)
            
            logger.debug(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            if run_manager:
                run_manager.on_text(f"Retrieval error: {str(e)}")
            return []
    
    def _truncate_documents(self, documents: List[Document]) -> List[Document]:
        """Truncate documents to fit within context limits
        
        Args:
            documents: List of documents to truncate
            
        Returns:
            Truncated documents that fit within limits
        """
        if not documents:
            return documents
        
        total_length = 0
        truncated_docs = []
        
        for doc in documents:
            doc_length = len(doc.page_content)
            
            if total_length + doc_length <= self.max_context_length:
                truncated_docs.append(doc)
                total_length += doc_length
            else:
                # Try to fit a truncated version of this document
                remaining_space = self.max_context_length - total_length
                if remaining_space > 100:  # Only if significant space remaining
                    truncated_content = doc.page_content[:remaining_space-50] + "..."
                    truncated_doc = Document(
                        page_content=truncated_content,
                        metadata={**doc.metadata, 'truncated': True}
                    )
                    truncated_docs.append(truncated_doc)
                break
        
        if len(truncated_docs) < len(documents):
            logger.debug(f"Truncated documents from {len(documents)} to {len(truncated_docs)}")
        
        return truncated_docs
    
    def _format_conversation_history(self, history: Any) -> str:
        """Format conversation history for prompt context
        
        Args:
            history: Conversation history (messages or string)
            
        Returns:
            Formatted conversation history string
        """
        try:
            if isinstance(history, str):
                return history
            elif isinstance(history, list):
                # Format LangChain messages
                formatted_parts = []
                for msg in history[-6:]:  # Last 3 turns (6 messages)
                    if hasattr(msg, 'content'):
                        msg_type = type(msg).__name__.replace('Message', '')
                        formatted_parts.append(f"{msg_type}: {msg.content}")
                    else:
                        formatted_parts.append(str(msg))
                return "\n".join(formatted_parts)
            else:
                return str(history)
        except Exception as e:
            logger.warning(f"Error formatting conversation history: {e}")
            return ""
    
    async def _generate_response(
        self,
        prompt_context: PromptContext,
        run_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Generate response using the LLM
        
        Args:
            prompt_context: Prompt context information
            run_manager: Callback manager
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Generate prompt using the prompt manager
            formatted_prompt = self.prompt_manager.generate_prompt(prompt_context)
            
            # Call Groq API using the complete method
            response = await self.llm_client.complete(
                prompt=formatted_prompt,
                temperature=0.7,
                max_tokens=self.settings.groq.max_tokens,
                cache=True
            )
            
            return {
                'answer': response.content,
                'token_usage': response.usage,
                'model_used': response.model,
                'prompt_length': len(formatted_prompt),
                'response_time': response.response_time,
                'cached': response.cached
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise QAChainError(f"Failed to generate response: {str(e)}")
    
    def stream_response(
        self,
        question: str,
        session_id: str = 'default'
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens as they're generated
        
        Args:
            question: User question
            session_id: Session identifier
            
        Yields:
            Response tokens as they're generated
        """
        async def _stream():
            try:
                inputs = {'question': question, 'session_id': session_id}
                result = await self._acall(inputs)
                
                # For now, yield the complete response
                # In a future enhancement, this could stream tokens from Groq
                yield result[self.output_key]
                
            except Exception as e:
                yield f"Error: {str(e)}"
        
        return _stream()


def create_qa_chain(
    gitlab_instance: Optional[str] = None,
    session_id: Optional[str] = None,
    max_sources: int = 10,
    include_wikis: bool = True,
    max_context_length: int = 8000
) -> RAGQAChain:
    """Factory function to create a configured QA chain
    
    Args:
        gitlab_instance: GitLab instance name
        session_id: Session identifier for conversation memory
        max_sources: Maximum number of source documents
        include_wikis: Whether to search wiki content
        max_context_length: Maximum context length
        
    Returns:
        Configured RAG QA chain
    """
    # Create components
    retriever = create_gitlab_retriever(
        gitlab_instance=gitlab_instance,
        max_results=max_sources,
        include_wikis=include_wikis
    )
    
    from ..groq_client import GroqClient
    llm_client = GroqClient()
    
    # Create memory if session_id provided
    memory = None
    if session_id:
        from ..memory.conversation_memory import ConversationMemory
        memory_store = get_global_memory_store()
        memory = ConversationMemory(
            session_id=session_id,
            memory_store=memory_store,
            max_context_turns=5
        )
    
    prompt_manager = get_prompt_manager()
    
    return RAGQAChain(
        retriever=retriever,
        llm_client=llm_client,
        memory=memory,
        prompt_manager=prompt_manager,
        max_context_length=max_context_length
    )


if __name__ == "__main__":
    # Test QA chain functionality
    import sys
    import asyncio
    
    async def test_qa_chain():
        if len(sys.argv) < 2:
            print("Usage: python -m src.ai.chains.qa_chain <question>")
            return
        
        question = ' '.join(sys.argv[1:])
        print(f"Testing QA chain with question: '{question}'")
        
        try:
            # Create QA chain
            chain = create_qa_chain(
                session_id='test_session',
                max_sources=5
            )
            
            # Run the chain
            result = await chain._acall({'question': question})
            
            print(f"\nAnswer: {result['answer']}")
            print(f"Sources: {len(result['source_documents'])}")
            print(f"Execution time: {result['execution_time_ms']:.1f}ms")
            print(f"Token usage: {result['token_usage']}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Run test
    asyncio.run(test_qa_chain())