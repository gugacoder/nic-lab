"""
Response Post-processing and Formatting

This module handles post-processing of AI responses, including source attribution,
formatting, and enhancement for the NIC Chat system.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class SourceReference:
    """Structured source reference information"""
    
    name: str
    project: str
    file_path: str
    url: Optional[str] = None
    relevance_score: float = 0.0
    content_snippet: Optional[str] = None


class ResponseFormatter:
    """Formats and enhances AI responses with proper citations and structure"""
    
    def __init__(self, max_snippet_length: int = 150):
        """Initialize response formatter
        
        Args:
            max_snippet_length: Maximum length for content snippets
        """
        self.max_snippet_length = max_snippet_length
    
    async def format_response(
        self,
        response: str,
        source_documents: List[Document],
        include_sources: bool = True,
        enhance_formatting: bool = True
    ) -> str:
        """Format AI response with citations and enhancements
        
        Args:
            response: Raw AI response text
            source_documents: Source documents used for the response
            include_sources: Whether to include source citations
            enhance_formatting: Whether to enhance text formatting
            
        Returns:
            Formatted response text
        """
        try:
            # Clean up the response
            cleaned_response = self._clean_response_text(response)
            
            # Enhance formatting if requested
            if enhance_formatting:
                cleaned_response = self._enhance_formatting(cleaned_response)
            
            # Add source citations if requested
            if include_sources and source_documents:
                citations = self._generate_citations(source_documents)
                if citations:
                    cleaned_response += "\n\n" + citations
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return response  # Return original response if formatting fails
    
    def _clean_response_text(self, text: str) -> str:
        """Clean up response text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        
        # Clean up common LLM artifacts
        text = re.sub(r'^(Answer:|Response:)\s*', '', text, flags=re.IGNORECASE)
        text = text.strip()
        
        return text
    
    def _enhance_formatting(self, text: str) -> str:
        """Enhance text formatting for readability"""
        # Convert **bold** to markdown if not already
        if '**' not in text:
            # Look for common emphasis patterns
            text = re.sub(r'\b([A-Z][A-Z\s]{2,})\b', r'**\1**', text)  # ALL CAPS to bold
        
        # Ensure proper spacing around headers
        text = re.sub(r'\n(#{1,6})\s*([^\n]+)', r'\n\1 \2\n', text)
        
        # Ensure proper list formatting
        text = re.sub(r'\n(\d+\.)\s*([^\n]+)', r'\n\1 \2', text)
        text = re.sub(r'\n(-|\*)\s*([^\n]+)', r'\n\1 \2', text)
        
        return text
    
    def _generate_citations(self, source_documents: List[Document]) -> str:
        """Generate formatted citations from source documents
        
        Args:
            source_documents: List of source documents
            
        Returns:
            Formatted citations text
        """
        if not source_documents:
            return ""
        
        # Group sources by project for better organization
        sources_by_project = self._group_sources_by_project(source_documents)
        
        citation_parts = ["**Sources:**"]
        
        for project, sources in sources_by_project.items():
            if len(sources_by_project) > 1:
                citation_parts.append(f"\n*{project}:*")
            
            for i, source in enumerate(sources, 1):
                citation_line = f"{i}. {source.name}"
                
                if source.content_snippet:
                    citation_line += f" - \"{source.content_snippet}...\""
                
                if source.url:
                    citation_line += f" ([View source]({source.url}))"
                
                citation_parts.append(citation_line)
        
        return "\n".join(citation_parts)
    
    def _group_sources_by_project(self, source_documents: List[Document]) -> Dict[str, List[SourceReference]]:
        """Group source documents by project
        
        Args:
            source_documents: List of source documents
            
        Returns:
            Dictionary mapping project names to source references
        """
        sources_by_project = {}
        
        for doc in source_documents:
            metadata = doc.metadata
            project = metadata.get('project_name', 'Unknown Project')
            
            # Create source reference
            source_ref = SourceReference(
                name=metadata.get('file_path', 'Unknown File'),
                project=project,
                file_path=metadata.get('file_path', ''),
                url=metadata.get('web_url'),
                relevance_score=metadata.get('relevance_score', 0.0),
                content_snippet=self._create_snippet(doc.page_content)
            )
            
            if project not in sources_by_project:
                sources_by_project[project] = []
            
            sources_by_project[project].append(source_ref)
        
        # Sort sources within each project by relevance
        for sources in sources_by_project.values():
            sources.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return sources_by_project
    
    def _create_snippet(self, content: str) -> str:
        """Create a content snippet for citations
        
        Args:
            content: Full content text
            
        Returns:
            Short snippet for citation
        """
        if not content:
            return ""
        
        # Find a good sentence to use as snippet
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if 10 <= len(sentence) <= self.max_snippet_length:
                return sentence
        
        # Fallback to truncated content
        if len(content) > self.max_snippet_length:
            return content[:self.max_snippet_length].rsplit(' ', 1)[0]
        
        return content
    
    def format_error_response(
        self,
        error: Exception,
        query: str,
        helpful_suggestions: Optional[List[str]] = None
    ) -> str:
        """Format an error response with helpful information
        
        Args:
            error: The error that occurred
            query: Original user query
            helpful_suggestions: Optional list of suggestions
            
        Returns:
            Formatted error response
        """
        response_parts = [
            "I apologize, but I encountered an issue while processing your request."
        ]
        
        # Add specific error context if appropriate
        error_type = type(error).__name__
        if error_type in ['TimeoutError', 'asyncio.TimeoutError']:
            response_parts.append(
                "The request took longer than expected to process. "
                "This might be due to high system load or complex queries."
            )
        elif 'connection' in str(error).lower():
            response_parts.append(
                "There was a connectivity issue with our systems. "
                "Please try again in a moment."
            )
        elif 'authentication' in str(error).lower():
            response_parts.append(
                "There was an authentication issue accessing the GitLab repositories. "
                "Please contact your administrator if this persists."
            )
        
        # Add helpful suggestions
        if helpful_suggestions:
            response_parts.append("\n**Suggestions:**")
            for suggestion in helpful_suggestions:
                response_parts.append(f"- {suggestion}")
        else:
            # Default suggestions
            response_parts.extend([
                "\n**You might try:**",
                "- Rephrasing your question with different keywords",
                "- Being more specific about the topic or project you're interested in",
                "- Asking about a related topic first to establish context"
            ])
        
        return "\n".join(response_parts)
    
    def format_no_sources_response(
        self,
        query: str,
        search_attempted: bool = True
    ) -> str:
        """Format response when no sources are found
        
        Args:
            query: Original user query
            search_attempted: Whether a search was actually attempted
            
        Returns:
            Formatted no-sources response
        """
        if search_attempted:
            response = (
                "I couldn't find specific information in our GitLab repositories "
                f"to answer your question about: **{query}**"
            )
        else:
            response = (
                "I wasn't able to search our repositories for information "
                f"about: **{query}**"
            )
        
        response += "\n\n**This could mean:**"
        response += "\n- The information might be in a repository I don't have access to"
        response += "\n- The topic might not be covered in our current documentation" 
        response += "\n- The question might need to be phrased differently"
        
        response += "\n\n**Try:**"
        response += "\n- Using different keywords or phrases"
        response += "\n- Being more specific about the project or area you're asking about"
        response += "\n- Asking about related topics that might be documented"
        
        return response
    
    def enhance_code_blocks(self, text: str) -> str:
        """Enhance code blocks with proper formatting
        
        Args:
            text: Text potentially containing code blocks
            
        Returns:
            Text with enhanced code formatting
        """
        # Detect and format inline code
        text = re.sub(
            r'`([^`\n]+)`',
            r'`\1`',
            text
        )
        
        # Detect and format code blocks
        text = re.sub(
            r'```(\w+)?\n(.*?)```',
            r'```\1\n\2```',
            text,
            flags=re.DOTALL
        )
        
        return text
    
    def add_response_metadata(
        self,
        response: str,
        metadata: Dict[str, Any],
        include_debug: bool = False
    ) -> str:
        """Add metadata information to response
        
        Args:
            response: Response text
            metadata: Metadata dictionary
            include_debug: Whether to include debug information
            
        Returns:
            Response with metadata appended
        """
        if not include_debug:
            return response
        
        debug_parts = [
            "\n---",
            "**Debug Information:**"
        ]
        
        if 'execution_time_ms' in metadata:
            debug_parts.append(f"- Response time: {metadata['execution_time_ms']:.1f}ms")
        
        if 'token_usage' in metadata:
            tokens = metadata['token_usage']
            if isinstance(tokens, dict) and 'total_tokens' in tokens:
                debug_parts.append(f"- Tokens used: {tokens['total_tokens']}")
        
        if 'sources_count' in metadata:
            debug_parts.append(f"- Sources found: {metadata['sources_count']}")
        
        return response + "\n" + "\n".join(debug_parts)


# Global formatter instance
_formatter_instance: Optional[ResponseFormatter] = None


def get_response_formatter() -> ResponseFormatter:
    """Get global response formatter singleton
    
    Returns:
        Global response formatter instance
    """
    global _formatter_instance
    if _formatter_instance is None:
        _formatter_instance = ResponseFormatter()
    return _formatter_instance


if __name__ == "__main__":
    # Test response formatter
    import asyncio
    
    async def test_formatter():
        formatter = ResponseFormatter()
        
        # Test basic formatting
        raw_response = """Answer:    GitLab is a     web-based DevOps platform.
        
        
        
        It provides    Git repository management and CI/CD capabilities."""
        
        formatted = await formatter.format_response(
            raw_response,
            [],
            include_sources=False,
            enhance_formatting=True
        )
        
        print("Original response:")
        print(repr(raw_response))
        print("\nFormatted response:")
        print(formatted)
        
        # Test with mock source documents
        mock_doc = Document(
            page_content="GitLab is a complete DevOps platform that enables teams to collaborate...",
            metadata={
                'source': 'gitlab-docs/overview.md',
                'project_name': 'GitLab Documentation',
                'file_path': 'overview.md',
                'web_url': 'https://gitlab.example.com/docs/-/blob/main/overview.md'
            }
        )
        
        formatted_with_sources = await formatter.format_response(
            "GitLab is a DevOps platform used for version control and CI/CD.",
            [mock_doc],
            include_sources=True
        )
        
        print("\n" + "="*50)
        print("Response with sources:")
        print(formatted_with_sources)
    
    asyncio.run(test_formatter())