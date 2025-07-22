"""
AI Module - RAG Pipeline and LLM Integration

This module contains the complete Retrieval-Augmented Generation (RAG) pipeline
for the NIC Chat system, including GitLab retrieval, conversation memory,
prompt engineering, and Groq LLM integration.
"""

from .rag_pipeline import RAGPipeline, RAGConfig, process_query
from .retrievers.gitlab_retriever import GitLabRetriever, create_gitlab_retriever
from .memory.conversation_memory import ConversationMemory, create_conversation_memory
from .prompts.templates import PromptManager, get_prompt_manager
from .chains.qa_chain import RAGQAChain, create_qa_chain

__all__ = [
    'RAGPipeline',
    'RAGConfig', 
    'process_query',
    'GitLabRetriever',
    'create_gitlab_retriever',
    'ConversationMemory', 
    'create_conversation_memory',
    'PromptManager',
    'get_prompt_manager',
    'RAGQAChain',
    'create_qa_chain'
]