"""
Prompt Templates for NIC Chat RAG Pipeline

This module provides engineered prompt templates for different query types
and interaction patterns in the corporate AI knowledge base system.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """Context information for prompt generation"""
    
    user_query: str
    retrieved_documents: List[str]
    source_names: List[str]
    conversation_history: Optional[str] = None
    query_intent: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class PromptTemplates:
    """Collection of prompt templates for different query types"""
    
    # System message for all interactions
    SYSTEM_MESSAGE = """You are an AI assistant for the NIC Chat corporate knowledge base system. Your role is to help employees find information from internal GitLab repositories and documentation.

CORE PRINCIPLES:
- Always ground your responses in the provided context from GitLab repositories
- Be accurate, professional, and helpful
- Cite sources clearly using the format [Source: project/file.md]
- If you cannot find relevant information in the provided context, say so clearly
- Maintain conversation context but prioritize current query requirements

RESPONSE GUIDELINES:
- Use clear, professional language appropriate for corporate communication
- Structure responses logically with headings and bullet points where helpful
- Include specific examples from the documentation when available
- Provide actionable guidance when possible
- Always cite the specific sources used in your response

Current timestamp: {timestamp}
"""

    # Basic Q&A template for straightforward information requests
    QA_TEMPLATE = """Based on the following information from our GitLab repositories, please answer the user's question.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED INFORMATION:
{context}

SOURCES:
{sources}

USER QUESTION: {question}

Please provide a comprehensive answer based on the retrieved information. If the information doesn't fully address the question, explain what aspects you can answer and what information might be missing.

Remember to cite your sources using the format [Source: project/file.md].

ANSWER:"""

    # Template for explanatory queries that need detailed breakdowns
    EXPLANATION_TEMPLATE = """Based on the following information from our GitLab repositories, please provide a detailed explanation of the concept or process the user asked about.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED INFORMATION:
{context}

SOURCES:
{sources}

USER QUESTION: {question}

Please provide a thorough explanation that includes:
1. A clear definition or overview
2. Key components or steps involved
3. Practical examples from our documentation
4. Any important considerations or best practices
5. Related concepts or next steps

Structure your explanation logically and cite specific sources for each key point using [Source: project/file.md].

EXPLANATION:"""

    # Template for comparison queries
    COMPARISON_TEMPLATE = """Based on the following information from our GitLab repositories, please provide a detailed comparison addressing the user's question.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED INFORMATION:
{context}

SOURCES:
{sources}

USER QUESTION: {question}

Please provide a structured comparison that includes:
1. Clear identification of what is being compared
2. Key similarities between the options
3. Important differences and their implications
4. Use cases or scenarios where each option is preferred
5. Recommendations based on the available information

Present your comparison in a clear, organized format and cite sources using [Source: project/file.md].

COMPARISON:"""

    # Template for troubleshooting and problem-solving queries
    TROUBLESHOOTING_TEMPLATE = """Based on the following information from our GitLab repositories, please help troubleshoot the user's issue.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED INFORMATION:
{context}

SOURCES:
{sources}

USER PROBLEM: {question}

Please provide troubleshooting guidance that includes:
1. Analysis of the described problem
2. Potential root causes based on our documentation
3. Step-by-step troubleshooting steps
4. Common solutions from our knowledge base
5. When to escalate or seek additional help
6. Prevention strategies for the future

Focus on actionable solutions and cite relevant documentation using [Source: project/file.md].

TROUBLESHOOTING GUIDANCE:"""

    # Template for procedural/how-to queries
    PROCEDURE_TEMPLATE = """Based on the following information from our GitLab repositories, please provide step-by-step instructions for the user's request.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED INFORMATION:
{context}

SOURCES:
{sources}

USER REQUEST: {question}

Please provide clear, actionable instructions that include:
1. Prerequisites or requirements
2. Detailed step-by-step procedure
3. Expected outcomes for each step
4. Common issues and how to resolve them
5. Verification steps to confirm success
6. Next steps or related procedures

Format your instructions clearly with numbered steps and cite sources using [Source: project/file.md].

INSTRUCTIONS:"""

    # Template for when no relevant context is found
    NO_CONTEXT_TEMPLATE = """I apologize, but I couldn't find relevant information in our GitLab repositories to answer your question about: {question}

This could mean:
1. The information might be in a repository I don't have access to
2. The topic might not be covered in our current documentation
3. The question might need to be phrased differently to find relevant content

SUGGESTIONS:
- Try rephrasing your question with different keywords
- Be more specific about the area or project you're interested in
- Consider if this might be covered in external documentation or require contacting a subject matter expert

CONVERSATION HISTORY (for context):
{chat_history}

Is there a different way you'd like to approach this question, or would you like to try searching for related topics?"""

    # Template for follow-up questions
    FOLLOWUP_TEMPLATE = """Based on our previous conversation and the following information from our GitLab repositories, please answer the user's follow-up question.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED INFORMATION:
{context}

SOURCES:
{sources}

FOLLOW-UP QUESTION: {question}

Please provide a response that:
1. Acknowledges the conversation context
2. Directly addresses the follow-up question
3. Builds on previous information when relevant
4. Provides additional details as needed
5. Maintains consistency with earlier responses

Cite sources using [Source: project/file.md] and reference previous discussion points when helpful.

RESPONSE:"""


class PromptManager:
    """Manages prompt template selection and generation"""
    
    def __init__(self):
        self.templates = PromptTemplates()
        self.intent_mapping = {
            'search': self.templates.QA_TEMPLATE,
            'explain': self.templates.EXPLANATION_TEMPLATE,
            'compare': self.templates.COMPARISON_TEMPLATE,
            'troubleshoot': self.templates.TROUBLESHOOTING_TEMPLATE,
            'procedure': self.templates.PROCEDURE_TEMPLATE,
            'howto': self.templates.PROCEDURE_TEMPLATE,
            'followup': self.templates.FOLLOWUP_TEMPLATE,
            'default': self.templates.QA_TEMPLATE
        }
    
    def detect_query_intent(self, query: str) -> str:
        """Detect query intent from user input
        
        Args:
            query: User query string
            
        Returns:
            Detected intent category
        """
        query_lower = query.lower()
        
        # Explanation keywords
        if any(keyword in query_lower for keyword in [
            'what is', 'explain', 'describe', 'definition of', 'tell me about'
        ]):
            return 'explain'
        
        # Comparison keywords  
        elif any(keyword in query_lower for keyword in [
            'difference between', 'compare', 'vs', 'versus', 'which is better',
            'pros and cons', 'advantages', 'disadvantages'
        ]):
            return 'compare'
        
        # Troubleshooting keywords
        elif any(keyword in query_lower for keyword in [
            'error', 'problem', 'issue', 'not working', 'failed', 'troubleshoot',
            'debug', 'fix', 'broken', 'help with'
        ]):
            return 'troubleshoot'
        
        # Procedure keywords
        elif any(keyword in query_lower for keyword in [
            'how to', 'how do i', 'steps to', 'guide', 'tutorial',
            'create', 'setup', 'configure', 'install'
        ]):
            return 'procedure'
        
        # Default to search/Q&A
        else:
            return 'search'
    
    def create_chat_prompt(
        self,
        context: PromptContext,
        intent_override: Optional[str] = None
    ) -> ChatPromptTemplate:
        """Create a chat prompt template based on query intent
        
        Args:
            context: Prompt context information
            intent_override: Override detected intent
            
        Returns:
            Configured chat prompt template
        """
        # Determine intent
        intent = intent_override or context.query_intent or self.detect_query_intent(context.user_query)
        
        # Handle case where no context was retrieved
        if not context.retrieved_documents or not any(doc.strip() for doc in context.retrieved_documents):
            template = self.templates.NO_CONTEXT_TEMPLATE
            
            return ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    self.templates.SYSTEM_MESSAGE
                ),
                HumanMessagePromptTemplate.from_template(template)
            ])
        
        # Select appropriate template
        template = self.intent_mapping.get(intent, self.intent_mapping['default'])
        
        # Create chat prompt template
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                self.templates.SYSTEM_MESSAGE
            ),
            HumanMessagePromptTemplate.from_template(template)
        ])
    
    def format_context_for_prompt(self, context: PromptContext) -> Dict[str, str]:
        """Format context information for prompt variables
        
        Args:
            context: Prompt context information
            
        Returns:
            Dictionary of formatted prompt variables
        """
        # Format retrieved documents
        if context.retrieved_documents:
            formatted_context = "\n\n".join([
                f"Document {i+1}:\n{doc}" 
                for i, doc in enumerate(context.retrieved_documents)
                if doc.strip()
            ])
        else:
            formatted_context = "No relevant documents found."
        
        # Format source names
        if context.source_names:
            formatted_sources = "\n".join([
                f"- {source}" for source in context.source_names if source.strip()
            ])
        else:
            formatted_sources = "No sources available."
        
        # Format conversation history
        chat_history = context.conversation_history or "No previous conversation."
        
        return {
            'timestamp': context.timestamp,
            'question': context.user_query,
            'context': formatted_context,
            'sources': formatted_sources,
            'chat_history': chat_history
        }
    
    def generate_prompt(
        self,
        context: PromptContext,
        intent_override: Optional[str] = None
    ) -> str:
        """Generate a complete prompt string
        
        Args:
            context: Prompt context information
            intent_override: Override detected intent
            
        Returns:
            Formatted prompt string
        """
        chat_prompt = self.create_chat_prompt(context, intent_override)
        prompt_variables = self.format_context_for_prompt(context)
        
        try:
            formatted_prompt = chat_prompt.format(**prompt_variables)
            logger.debug(f"Generated prompt for intent: {context.query_intent or 'detected'}")
            return formatted_prompt
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            # Fallback to basic template
            fallback_template = PromptTemplate.from_template(
                "Please answer the following question based on available information: {question}"
            )
            return fallback_template.format(question=context.user_query)
    
    def get_available_intents(self) -> List[str]:
        """Get list of available query intents
        
        Returns:
            List of supported intent categories
        """
        return list(self.intent_mapping.keys())


# Factory function for easy prompt creation
def create_prompt_manager() -> PromptManager:
    """Create a prompt manager instance
    
    Returns:
        Configured prompt manager
    """
    return PromptManager()


# Global prompt manager instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get global prompt manager singleton
    
    Returns:
        Global prompt manager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


if __name__ == "__main__":
    # Test prompt template functionality
    import sys
    
    print("Testing prompt template system...")
    
    manager = PromptManager()
    
    # Test intent detection
    test_queries = [
        "What is GitLab?",
        "How do I create a new project?", 
        "Compare Docker vs Kubernetes",
        "I'm getting an authentication error",
        "Explain the CI/CD pipeline"
    ]
    
    print("\nIntent Detection Tests:")
    for query in test_queries:
        intent = manager.detect_query_intent(query)
        print(f"  '{query}' -> {intent}")
    
    # Test prompt generation
    if len(sys.argv) > 1:
        test_query = ' '.join(sys.argv[1:])
        print(f"\nGenerating prompt for: '{test_query}'")
        
        context = PromptContext(
            user_query=test_query,
            retrieved_documents=["Sample document content about GitLab authentication..."],
            source_names=["gitlab/docs/auth.md"],
            conversation_history="Previous question: What is GitLab?\nPrevious answer: GitLab is a DevOps platform..."
        )
        
        prompt = manager.generate_prompt(context)
        print(f"\nGenerated prompt length: {len(prompt)} characters")
        print("Sample prompt (first 500 chars):")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    print("\n✅ Prompt template tests completed")