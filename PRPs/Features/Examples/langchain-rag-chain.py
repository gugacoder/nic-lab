"""
LangChain RAG Pipeline Example
Demonstrates setting up a RAG chain with GitLab retriever and Groq LLM
"""

import os
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


class GitLabRetriever(BaseRetriever):
    """Custom retriever for GitLab content"""
    
    def __init__(self, gitlab_client, project_ids: List[int] = None):
        super().__init__()
        self.gitlab_client = gitlab_client
        self.project_ids = project_ids
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents from GitLab"""
        
        # Search GitLab for relevant content
        search_results = self.gitlab_client.search_content(
            query=query,
            project_ids=self.project_ids
        )
        
        documents = []
        for result in search_results:
            # Create document with metadata
            doc = Document(
                page_content=result['content'],
                metadata={
                    'source': f"{result['project']}/{result['path']}",
                    'project_id': result['project_id'],
                    'filename': result['filename']
                }
            )
            
            # Split large documents into chunks
            if len(doc.page_content) > 1000:
                chunks = self.text_splitter.split_documents([doc])
                documents.extend(chunks)
            else:
                documents.append(doc)
        
        return documents


class MockGroqLLM(BaseChatModel):
    """Mock Groq LLM for example purposes"""
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Mock generation - replace with actual Groq API call"""
        # This is a mock response - in production, use actual Groq API
        response = "Based on the provided context, here's a comprehensive answer to your question..."
        return response
    
    @property
    def _llm_type(self):
        return "mock_groq"


def create_rag_chain(gitlab_client, groq_api_key: str):
    """Create a complete RAG chain"""
    
    # Initialize components
    retriever = GitLabRetriever(gitlab_client)
    llm = MockGroqLLM()  # In production, use actual Groq client
    
    # Create prompt template
    template = """You are a helpful AI assistant with access to a corporate knowledge base.
Use the following context to answer the user's question. If you cannot find the answer in the context,
say so clearly and provide the best response you can based on general knowledge.

Context:
{context}

Question: {question}

Please provide a comprehensive answer, citing specific sources when possible.
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Build the chain
    chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def format_documents(docs: List[Document]) -> str:
    """Format documents for the prompt"""
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content.strip()
        formatted_docs.append(f"[Source {i+1}: {source}]\n{content}")
    
    return "\n\n---\n\n".join(formatted_docs)


class ConversationMemory:
    """Simple conversation memory management"""
    
    def __init__(self, max_messages: int = 10):
        self.messages = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str):
        """Add a message to memory"""
        self.messages.append({"role": role, "content": content})
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> str:
        """Get conversation context"""
        context_parts = []
        for msg in self.messages[-4:]:  # Last 2 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts) if context_parts else ""


class RAGPipeline:
    """Complete RAG pipeline with memory"""
    
    def __init__(self, chain, memory: ConversationMemory):
        self.chain = chain
        self.memory = memory
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the RAG pipeline"""
        
        # Add conversation context to query if available
        context = self.memory.get_context()
        if context:
            enhanced_query = f"Previous conversation:\n{context}\n\nCurrent question: {query}"
        else:
            enhanced_query = query
        
        # Run the chain
        response = await self.chain.ainvoke(enhanced_query)
        
        # Update memory
        self.memory.add_message("user", query)
        self.memory.add_message("assistant", response)
        
        return {
            "query": query,
            "response": response,
            "sources": self._extract_sources(response),
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_sources(self, response: str) -> List[str]:
        """Extract source citations from response"""
        # Simple extraction - in production, use more sophisticated parsing
        sources = []
        lines = response.split('\n')
        for line in lines:
            if line.startswith('[Source'):
                sources.append(line)
        return sources


# Example usage
if __name__ == "__main__":
    from datetime import datetime
    
    print("=== LangChain RAG Pipeline Example ===\n")
    
    # Note: This is a demonstration of the structure
    # In production, you would:
    # 1. Initialize actual GitLab client
    # 2. Use real Groq API client
    # 3. Implement proper async handling
    
    # Mock components for demonstration
    class MockGitLabClient:
        def search_content(self, query, project_ids=None):
            return [
                {
                    'project': 'documentation',
                    'project_id': 1,
                    'path': 'docs/setup.md',
                    'filename': 'setup.md',
                    'content': 'To set up authentication, configure your GitLab token...'
                },
                {
                    'project': 'wiki',
                    'project_id': 2,
                    'path': 'wiki/auth-guide.md',
                    'filename': 'auth-guide.md',
                    'content': 'Authentication requires a personal access token with read permissions...'
                }
            ]
    
    # Create pipeline
    gitlab_client = MockGitLabClient()
    chain = create_rag_chain(gitlab_client, "mock_groq_key")
    memory = ConversationMemory()
    pipeline = RAGPipeline(chain, memory)
    
    # Example queries
    queries = [
        "How do I set up authentication?",
        "What permissions are needed?",
        "Can you provide more details about tokens?"
    ]
    
    print("Processing example queries:\n")
    for query in queries:
        print(f"User: {query}")
        # In production, this would be async
        result = {
            "query": query,
            "response": f"Mock response for: {query}",
            "sources": ["[Source 1: documentation/docs/setup.md]"],
            "timestamp": datetime.now().isoformat()
        }
        print(f"Assistant: {result['response']}")
        print(f"Sources: {result['sources']}")
        print("-" * 50)