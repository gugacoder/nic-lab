"""
LLM Chat Bridge Implementation Example

This example demonstrates how to replace hardcoded responses with real LLM integration,
addressing the critical gap identified in the NIC Chat system.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime

# Import existing components (these should already exist)
try:
    from src.ai.groq_client import GroqClient, GroqResponse
    from src.utils.session import ChatStateManager
    from src.config.settings import get_settings
except ImportError:
    # Fallback for example purposes
    class GroqClient:
        async def stream_completion(self, messages, **kwargs):
            # Mock implementation for example
            response = "This would be a real LLM response"
            for word in response.split():
                yield type('Chunk', (), {'content': word + ' '})()
                await asyncio.sleep(0.1)

logger = logging.getLogger(__name__)


class LLMChatBridge:
    """
    Bridge service that connects the chat interface to the Groq LLM client.
    
    This replaces hardcoded responses with real AI-generated content while
    maintaining the existing chat UI behavior and adding proper error handling.
    """
    
    def __init__(self):
        """Initialize the LLM chat bridge"""
        self.settings = get_settings() if 'get_settings' in globals() else None
        self.groq_client = GroqClient()
        self.conversation_context = {}
        self.is_connected = False
        self.last_health_check = None
        
    async def initialize(self) -> bool:
        """Initialize and test LLM connection"""
        try:
            # Test connection to Groq API
            test_response = await self.groq_client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            self.is_connected = True
            self.last_health_check = datetime.now()
            logger.info("LLM Chat Bridge initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM Chat Bridge: {e}")
            self.is_connected = False
            return False
    
    async def send_message(self, user_message: str, session_id: str = None) -> AsyncGenerator[str, None]:
        """
        Send message to LLM and stream response.
        
        This is the main function that replaces hardcoded responses.
        
        Args:
            user_message: User's input message
            session_id: Session ID for conversation context
            
        Yields:
            str: Streaming response tokens from LLM
        """
        try:
            # Build conversation context
            messages = self._build_conversation_context(user_message, session_id)
            
            # Stream response from Groq LLM
            response_content = ""
            async for chunk in self.groq_client.stream_completion(
                messages=messages,
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=2048
            ):
                if chunk.content:
                    response_content += chunk.content
                    yield chunk.content
            
            # Store conversation context for future turns
            self._update_conversation_context(user_message, response_content, session_id)
            
        except Exception as e:
            logger.error(f"Error in LLM response generation: {e}")
            # Yield error fallback response
            yield self._generate_error_fallback(str(e))
    
    async def get_complete_response(self, user_message: str, session_id: str = None) -> str:
        """
        Get complete LLM response (non-streaming version).
        
        Alternative to streaming for simpler integration patterns.
        """
        try:
            response_parts = []
            async for chunk in self.send_message(user_message, session_id):
                response_parts.append(chunk)
            return "".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error getting complete response: {e}")
            return self._generate_error_fallback(str(e))
    
    def _build_conversation_context(self, user_message: str, session_id: str = None) -> List[Dict[str, str]]:
        """Build conversation context for LLM from session history"""
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": (
                "You are a helpful AI assistant for the NIC Chat corporate knowledge system. "
                "Provide professional, accurate, and helpful responses. If you need information "
                "from corporate documents, clearly indicate what type of information would be helpful."
            )
        })
        
        # Add conversation history from session
        if session_id and session_id in self.conversation_context:
            history = self.conversation_context[session_id]
            for entry in history[-10:]:  # Limit context to last 10 exchanges
                messages.append({"role": "user", "content": entry["user"]})
                messages.append({"role": "assistant", "content": entry["assistant"]})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _update_conversation_context(self, user_message: str, response: str, session_id: str = None):
        """Update conversation context with new exchange"""
        if not session_id:
            session_id = "default"
        
        if session_id not in self.conversation_context:
            self.conversation_context[session_id] = []
        
        self.conversation_context[session_id].append({
            "user": user_message,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit conversation history size
        if len(self.conversation_context[session_id]) > 50:
            self.conversation_context[session_id] = self.conversation_context[session_id][-25:]
    
    def _generate_error_fallback(self, error_message: str) -> str:
        """Generate fallback response when LLM fails"""
        return (
            "I apologize, but I'm currently experiencing technical difficulties connecting to the AI service. "
            "Please try again in a moment. If the issue persists, you can:\n\n"
            "• Check your internet connection\n"
            "• Contact system support\n"
            "• Try a simpler query\n\n"
            f"Technical details: {error_message}"
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LLM API health and connectivity"""
        try:
            start_time = datetime.now()
            test_response = await self.groq_client.complete(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            response_time = (datetime.now() - start_time).total_seconds()
            
            self.is_connected = True
            self.last_health_check = datetime.now()
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "last_check": self.last_health_check.isoformat(),
                "api_connected": True
            }
            
        except Exception as e:
            self.is_connected = False
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat(),
                "api_connected": False
            }


# Example usage replacing hardcoded functions

async def handle_ai_response_new(user_message: str, session_id: str = None) -> str:
    """
    NEW: Real LLM integration function to replace hardcoded responses.
    
    This replaces the old _handle_ai_response() function in src/app.py
    """
    bridge = LLMChatBridge()
    
    # Ensure bridge is initialized
    if not bridge.is_connected:
        await bridge.initialize()
    
    # Get complete response from LLM
    return await bridge.get_complete_response(user_message, session_id)


async def stream_ai_response_new(user_message: str, session_id: str = None) -> AsyncGenerator[str, None]:
    """
    NEW: Streaming LLM integration function for real-time responses.
    
    This provides streaming capability for better user experience.
    """
    bridge = LLMChatBridge()
    
    # Ensure bridge is initialized
    if not bridge.is_connected:
        await bridge.initialize()
    
    # Stream response from LLM
    async for chunk in bridge.send_message(user_message, session_id):
        yield chunk


# Example integration with existing chat components

def integrate_with_app_py():
    """
    Example of how to modify src/app.py to use real LLM integration.
    
    Replace this code in src/app.py around line 162:
    
    OLD CODE:
    def _handle_ai_response(user_message: str) -> str:
        import time
        import random
        time.sleep(random.uniform(0.5, 2.0))
        responses = [...]  # hardcoded array
        return random.choice(responses)
    
    NEW CODE:
    async def _handle_ai_response(user_message: str) -> str:
        from integrations.llm_chat_bridge import handle_ai_response_new
        return await handle_ai_response_new(user_message)
    """
    pass


def integrate_with_chat_container():
    """
    Example of how to modify src/components/chat/chat_container.py.
    
    Replace the _generate_placeholder_response function with:
    
    @staticmethod
    async def _generate_llm_response(user_message: str) -> str:
        from integrations.llm_chat_bridge import handle_ai_response_new
        return await handle_ai_response_new(user_message)
    """
    pass


# Testing and validation functions

async def test_integration():
    """Test the LLM chat bridge integration"""
    print("Testing LLM Chat Bridge Integration...")
    
    bridge = LLMChatBridge()
    
    # Test initialization
    print("1. Testing initialization...")
    success = await bridge.initialize()
    print(f"   Initialization: {'✅ Success' if success else '❌ Failed'}")
    
    # Test health check
    print("2. Testing health check...")
    health = await bridge.health_check()
    print(f"   Health Status: {health['status']}")
    print(f"   API Connected: {health['api_connected']}")
    
    # Test complete response
    print("3. Testing complete response...")
    response = await bridge.get_complete_response("Hello, how are you?")
    print(f"   Response: {response[:100]}...")
    
    # Test streaming response
    print("4. Testing streaming response...")
    print("   Streaming: ", end="")
    async for chunk in bridge.send_message("Tell me about AI"):
        print(chunk, end="", flush=True)
    print("\n")
    
    print("✅ Integration test complete!")


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_integration())