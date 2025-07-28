"""
Conversation Memory Management for RAG Pipeline

This module provides conversation memory and context management for the NIC Chat
system, maintaining conversation history and context for improved AI responses.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import hashlib

from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single conversation turn (question + answer)"""
    
    human_message: str
    ai_message: str
    timestamp: datetime
    context_used: List[str]  # Sources used in this turn
    metadata: Dict[str, Any]
    turn_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'human_message': self.human_message,
            'ai_message': self.ai_message,
            'timestamp': self.timestamp.isoformat(),
            'context_used': self.context_used,
            'metadata': self.metadata,
            'turn_id': self.turn_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary (deserialization)"""
        return cls(
            human_message=data['human_message'],
            ai_message=data['ai_message'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            context_used=data['context_used'],
            metadata=data['metadata'],
            turn_id=data['turn_id']
        )


@dataclass
class ConversationSession:
    """Represents a complete conversation session"""
    
    session_id: str
    created_at: datetime
    last_activity: datetime
    turns: List[ConversationTurn]
    metadata: Dict[str, Any]
    
    def add_turn(self, turn: ConversationTurn):
        """Add a new conversation turn"""
        self.turns.append(turn)
        self.last_activity = datetime.now()
    
    def get_recent_turns(self, count: int = 5) -> List[ConversationTurn]:
        """Get the most recent conversation turns"""
        return self.turns[-count:] if self.turns else []
    
    def get_context_summary(self) -> str:
        """Generate a summary of recent conversation context"""
        recent_turns = self.get_recent_turns(3)
        if not recent_turns:
            return ""
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"Human: {turn.human_message}")
            context_parts.append(f"AI: {turn.ai_message}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'turns': [turn.to_dict() for turn in self.turns],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        """Create from dictionary (deserialization)"""
        return cls(
            session_id=data['session_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_activity=datetime.fromisoformat(data['last_activity']),
            turns=[ConversationTurn.from_dict(t) for t in data['turns']],
            metadata=data['metadata']
        )


class ConversationMemoryStore:
    """In-memory conversation storage with optional persistence"""
    
    def __init__(self, max_sessions: int = 100, session_ttl_hours: int = 24):
        """Initialize conversation memory store
        
        Args:
            max_sessions: Maximum number of sessions to keep in memory
            session_ttl_hours: Session time-to-live in hours
        """
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self.access_order = deque()  # For LRU eviction
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationSession:
        """Create a new conversation session
        
        Args:
            session_id: Session ID (auto-generated if None)
            metadata: Session metadata
            
        Returns:
            New conversation session
        """
        if session_id is None:
            session_id = self._generate_session_id()
        
        metadata = metadata or {}
        session = ConversationSession(
            session_id=session_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            turns=[],
            metadata=metadata
        )
        
        self.sessions[session_id] = session
        self.access_order.append(session_id)
        
        # Clean up old sessions if needed
        self._cleanup_sessions()
        
        logger.debug(f"Created conversation session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation session if found, None otherwise
        """
        session = self.sessions.get(session_id)
        if session:
            # Update access order for LRU
            if session_id in self.access_order:
                self.access_order.remove(session_id)
            self.access_order.append(session_id)
            
            # Check if session has expired
            if datetime.now() - session.last_activity > self.session_ttl:
                logger.debug(f"Session {session_id} has expired, removing")
                self.delete_session(session_id)
                return None
        
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session
        
        Args:
            session_id: Session to delete
            
        Returns:
            True if session was deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            if session_id in self.access_order:
                self.access_order.remove(session_id)
            logger.debug(f"Deleted conversation session: {session_id}")
            return True
        return False
    
    def add_turn(
        self,
        session_id: str,
        human_message: str,
        ai_message: str,
        context_used: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """Add a conversation turn to a session
        
        Args:
            session_id: Session identifier
            human_message: User's message
            ai_message: AI's response
            context_used: Sources used in the response
            metadata: Turn metadata
            
        Returns:
            The created conversation turn
        """
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        turn = ConversationTurn(
            human_message=human_message,
            ai_message=ai_message,
            timestamp=datetime.now(),
            context_used=context_used or [],
            metadata=metadata or {},
            turn_id=self._generate_turn_id()
        )
        
        session.add_turn(turn)
        logger.debug(f"Added turn to session {session_id}: {turn.turn_id}")
        return turn
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about stored sessions
        
        Returns:
            Dictionary with session statistics
        """
        total_sessions = len(self.sessions)
        total_turns = sum(len(session.turns) for session in self.sessions.values())
        
        if self.sessions:
            oldest_session = min(self.sessions.values(), key=lambda s: s.created_at)
            newest_session = max(self.sessions.values(), key=lambda s: s.created_at)
        else:
            oldest_session = newest_session = None
        
        return {
            'total_sessions': total_sessions,
            'total_turns': total_turns,
            'max_sessions': self.max_sessions,
            'session_ttl_hours': self.session_ttl.total_seconds() / 3600,
            'oldest_session_age_hours': (
                (datetime.now() - oldest_session.created_at).total_seconds() / 3600
                if oldest_session else 0
            ),
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"session_{timestamp}".encode()).hexdigest()[:16]
    
    def _generate_turn_id(self) -> str:
        """Generate a unique turn ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"turn_{timestamp}".encode()).hexdigest()[:12]
    
    def _cleanup_sessions(self):
        """Remove expired and excess sessions"""
        now = datetime.now()
        
        # Remove expired sessions
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if now - session.last_activity > self.session_ttl
        ]
        
        for sid in expired_sessions:
            self.delete_session(sid)
        
        # Remove excess sessions using LRU
        while len(self.sessions) > self.max_sessions:
            if self.access_order:
                oldest_session_id = self.access_order.popleft()
                self.delete_session(oldest_session_id)
            else:
                break
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB (rough approximation)"""
        total_chars = 0
        
        for session in self.sessions.values():
            for turn in session.turns:
                total_chars += len(turn.human_message)
                total_chars += len(turn.ai_message)
                total_chars += sum(len(source) for source in turn.context_used)
        
        # Rough estimate: 1 char ≈ 2 bytes (Unicode), plus overhead
        estimated_bytes = total_chars * 3  # Include overhead
        return estimated_bytes / (1024 * 1024)


class ConversationMemory(BaseMemory):
    """LangChain-compatible conversation memory implementation"""
    
    memory_key: str = "chat_history"
    return_messages: bool = True
    
    def __init__(
        self,
        session_id: str,
        memory_store: Optional[ConversationMemoryStore] = None,
        max_context_turns: int = 5,
        **kwargs
    ):
        """Initialize conversation memory
        
        Args:
            session_id: Unique session identifier
            memory_store: Memory store instance (creates new if None)
            max_context_turns: Maximum conversation turns to include in context
            **kwargs: Additional arguments for BaseMemory
        """
        super().__init__(**kwargs)
        self.session_id = session_id
        self.memory_store = memory_store or ConversationMemoryStore()
        self.max_context_turns = max_context_turns
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables that this memory class provides"""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables from storage
        
        Args:
            inputs: Input variables (not used)
            
        Returns:
            Dictionary with memory variables
        """
        session = self.memory_store.get_session(self.session_id)
        
        if not session:
            return {self.memory_key: []}
        
        recent_turns = session.get_recent_turns(self.max_context_turns)
        
        if self.return_messages:
            # Return as LangChain messages
            messages = []
            for turn in recent_turns:
                messages.append(HumanMessage(content=turn.human_message))
                messages.append(AIMessage(content=turn.ai_message))
            return {self.memory_key: messages}
        else:
            # Return as formatted string
            context = session.get_context_summary()
            return {self.memory_key: context}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context to memory
        
        Args:
            inputs: Input variables (should contain human message)
            outputs: Output variables (should contain AI response)
        """
        # Extract human message
        human_message = inputs.get("input", inputs.get("question", ""))
        
        # Extract AI response
        ai_message = outputs.get("output", outputs.get("answer", ""))
        
        # Extract context sources if available
        context_used = outputs.get("source_documents", [])
        if context_used and hasattr(context_used[0], 'metadata'):
            sources = [doc.metadata.get('source', '') for doc in context_used]
        else:
            sources = []
        
        # Save the turn
        self.memory_store.add_turn(
            session_id=self.session_id,
            human_message=human_message,
            ai_message=ai_message,
            context_used=sources,
            metadata={
                'input_keys': list(inputs.keys()),
                'output_keys': list(outputs.keys())
            }
        )
    
    def clear(self) -> None:
        """Clear memory for this session"""
        self.memory_store.delete_session(self.session_id)


def create_conversation_memory(
    session_id: str,
    max_context_turns: int = 5,
    max_sessions: int = 100,
    session_ttl_hours: int = 24
) -> ConversationMemory:
    """Factory function to create conversation memory
    
    Args:
        session_id: Unique session identifier
        max_context_turns: Maximum turns to include in context
        max_sessions: Maximum sessions in memory store
        session_ttl_hours: Session time-to-live in hours
        
    Returns:
        Configured conversation memory instance
    """
    memory_store = ConversationMemoryStore(
        max_sessions=max_sessions,
        session_ttl_hours=session_ttl_hours
    )
    
    return ConversationMemory(
        session_id=session_id,
        memory_store=memory_store,
        max_context_turns=max_context_turns
    )


# Global memory store for sharing across sessions
_global_memory_store: Optional[ConversationMemoryStore] = None


def get_global_memory_store() -> ConversationMemoryStore:
    """Get global memory store singleton
    
    Returns:
        Global conversation memory store
    """
    global _global_memory_store
    if _global_memory_store is None:
        _global_memory_store = ConversationMemoryStore()
    return _global_memory_store


if __name__ == "__main__":
    # Test conversation memory functionality
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test-persistence":
        # Test memory persistence
        print("Testing conversation memory persistence...")
        
        memory_store = ConversationMemoryStore(max_sessions=5)
        
        # Create a session and add some turns
        session = memory_store.create_session("test_session_1")
        memory_store.add_turn(
            "test_session_1",
            "What is GitLab?",
            "GitLab is a web-based DevOps lifecycle tool...",
            ["gitlab_docs.md"],
            {"test": True}
        )
        memory_store.add_turn(
            "test_session_1",
            "How do I configure authentication?",
            "To configure authentication in GitLab...",
            ["auth_guide.md", "config.md"]
        )
        
        # Test retrieval
        retrieved_session = memory_store.get_session("test_session_1")
        if retrieved_session:
            print(f"Session has {len(retrieved_session.turns)} turns")
            context = retrieved_session.get_context_summary()
            print(f"Context summary length: {len(context)} characters")
            print("✅ Persistence test passed")
        else:
            print("❌ Persistence test failed")
        
        # Test stats
        stats = memory_store.get_session_stats()
        print(f"Memory stats: {stats}")
    
    else:
        # Test LangChain memory integration
        print("Testing LangChain memory integration...")
        
        memory = create_conversation_memory("test_session", max_context_turns=3)
        
        # Simulate conversation
        inputs1 = {"input": "Hello, what can you help me with?"}
        outputs1 = {"output": "I can help you with GitLab questions and documentation."}
        memory.save_context(inputs1, outputs1)
        
        inputs2 = {"input": "How do I create a new project?"}
        outputs2 = {"output": "To create a new project in GitLab, follow these steps..."}
        memory.save_context(inputs2, outputs2)
        
        # Load memory variables
        loaded = memory.load_memory_variables({})
        messages = loaded["chat_history"]
        
        print(f"Loaded {len(messages)} messages from memory")
        for i, msg in enumerate(messages):
            print(f"Message {i}: {type(msg).__name__}: {msg.content[:50]}...")
        
        print("✅ LangChain integration test passed")