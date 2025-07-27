"""
LLM Chat Bridge Implementation

This service connects the chat interface to the Groq LLM client,
replacing hardcoded responses with real AI-generated content.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime

from src.utils.session import ChatStateManager
from src.config.settings import get_settings

# Import GroqClient directly to avoid circular imports
try:
    from src.ai.groq_client import GroqClient, GroqResponse
except ImportError:
    # Fallback if there are circular import issues
    GroqClient = None
    GroqResponse = None

logger = logging.getLogger(__name__)


class LLMChatBridge:
    """
    Bridge service that connects the chat interface to the Groq LLM client.
    
    This replaces hardcoded responses with real AI-generated content while
    maintaining the existing chat UI behavior and adding proper error handling.
    """
    
    def __init__(self):
        """Initialize the LLM chat bridge"""
        self.settings = get_settings()
        
        # Initialize GroqClient with error handling
        if GroqClient is not None:
            self.groq_client = GroqClient()
        else:
            logger.error("GroqClient not available due to import issues")
            self.groq_client = None
            
        self.conversation_context = {}
        self.is_connected = False
        self.last_health_check = None
        
    async def initialize(self) -> bool:
        """Initialize and test LLM connection"""
        if self.groq_client is None:
            logger.error("Cannot initialize - GroqClient not available")
            self.is_connected = False
            return False
            
        try:
            # Test connection to Groq API
            test_response = await self.groq_client.complete(
                prompt="Hello",
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
        if self.groq_client is None:
            logger.error("GroqClient not available - using fallback response")
            yield self._generate_error_fallback("LLM service temporarily unavailable")
            return
            
        try:
            # Build conversation context
            messages = self._build_conversation_context(user_message, session_id)
            
            # Stream response from Groq LLM
            response_content = ""
            
            # Convert messages format for GroqClient
            if messages and len(messages) > 0:
                # Use the last user message as prompt
                prompt = messages[-1]["content"]
            else:
                prompt = user_message
                
            # Get the streaming generator from GroqClient (await the coroutine)
            stream_generator = await self.groq_client.complete(
                prompt=prompt,
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=2048,
                stream=True
            )
            
            # Iterate over the streaming response
            async for chunk in stream_generator:
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
    
    def get_sync_response(self, user_message: str, session_id: str = None) -> str:
        """
        Synchronous version for compatibility with existing sync functions.
        
        This is a wrapper around the async version for easy integration.
        """
        try:
            # Create new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            return loop.run_until_complete(
                self.get_complete_response(user_message, session_id)
            )
        except Exception as e:
            logger.error(f"Error in sync response generation: {e}")
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
                prompt="test",
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


# Global bridge instance for easy access
_bridge_instance = None

def get_llm_bridge() -> LLMChatBridge:
    """Get or create global LLM bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = LLMChatBridge()
    return _bridge_instance


# Replacement functions for hardcoded responses

def handle_ai_response_sync(user_message: str) -> str:
    """
    Synchronous LLM integration function to replace hardcoded responses.
    
    This replaces the old _handle_ai_response() function in src/app.py
    """
    bridge = get_llm_bridge()
    return bridge.get_sync_response(user_message)


async def handle_ai_response_async(user_message: str, session_id: str = None) -> str:
    """
    Asynchronous LLM integration function to replace hardcoded responses.
    """
    bridge = get_llm_bridge()
    
    # Ensure bridge is initialized
    if not bridge.is_connected:
        await bridge.initialize()
    
    # Get complete response from LLM
    return await bridge.get_complete_response(user_message, session_id)


async def stream_ai_response(user_message: str, session_id: str = None) -> AsyncGenerator[str, None]:
    """
    Streaming LLM integration function for real-time responses.
    
    This provides streaming capability for better user experience.
    """
    bridge = get_llm_bridge()
    
    # Ensure bridge is initialized
    if not bridge.is_connected:
        await bridge.initialize()
    
    # Stream response from LLM
    async for chunk in bridge.send_message(user_message, session_id):
        yield chunk


# Testing and validation functions

async def test_integration():
    """Test the LLM chat bridge integration"""
    print("Testing LLM Chat Bridge Integration...")
    
    bridge = LLMChatBridge()
    
    # Test initialization
    print("1. Testing initialization...")
    success = await bridge.initialize()
    print(f"   Initialization: {'✅ Success' if success else '❌ Failed'}")
    
    if not success:
        print("❌ Cannot continue tests without successful initialization")
        return
    
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