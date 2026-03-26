"""
Conversation history manager for storing and retrieving user conversation history.
Provides persistence across bot restarts using JSON file storage.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history storage and retrieval for users."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the conversation manager.
        
        Args:
            data_dir: Directory to store conversation data
        """
        self.data_dir = Path(data_dir)
        self.conversations_file = self.data_dir / "conversations.json"
        self.conversations: Dict[int, List[Dict[str, Any]]] = {}
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
        
        # Load existing conversations
        self._load_conversations()
    
    def _load_conversations(self) -> None:
        """Load conversations from JSON file."""
        try:
            if self.conversations_file.exists():
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert string keys back to integers (JSON only supports string keys)
                    self.conversations = {
                        int(user_id): conversations 
                        for user_id, conversations in data.items()
                    }
                logger.info(f"Loaded conversations for {len(self.conversations)} users")
            else:
                logger.info("No existing conversations file found, starting fresh")
                self.conversations = {}
        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
            self.conversations = {}
    
    def _save_conversations(self) -> None:
        """Save conversations to JSON file."""
        try:
            # Convert to serializable format
            serializable_data = {
                str(user_id): conversations 
                for user_id, conversations in self.conversations.items()
            }
            
            with open(self.conversations_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved conversations for {len(self.conversations)} users")
        except Exception as e:
            logger.error(f"Failed to save conversations: {e}")
    
    def add_message(self, user_id: int, role: str, content: Any,
                   timestamp: Optional[datetime] = None) -> None:
        """
        Add a message to a user's conversation history.
        
        Args:
            user_id: Telegram user ID
            role: Message role ('user' or 'assistant')
            content: Message content (string or list for multimodal content)
            timestamp: Optional timestamp (defaults to current time)
        """
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": (timestamp or datetime.now()).isoformat()
        }
        
        self.conversations[user_id].append(message)
        self._save_conversations()
        
        logger.debug(f"Added {role} message to conversation for user {user_id}")
    
    def get_conversation(self, user_id: int, max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get a user's conversation history.
        
        Args:
            user_id: Telegram user ID
            max_messages: Optional limit on number of messages to return (most recent)
            
        Returns:
            List of message dictionaries
        """
        if user_id not in self.conversations:
            return []
        
        conversation = self.conversations[user_id]
        
        if max_messages and len(conversation) > max_messages:
            return conversation[-max_messages:]
        
        return conversation
    
    def get_conversation_for_api(self, user_id: int, max_messages: Optional[int] = 20) -> List[Dict[str, str]]:
        """
        Get conversation formatted for API consumption (only role and content).
        
        Args:
            user_id: Telegram user ID
            max_messages: Optional limit on number of messages to return
            
        Returns:
            List of message dictionaries with only 'role' and 'content' keys
        """
        conversation = self.get_conversation(user_id, max_messages)
        
        # Format for API (strip timestamp, keep only role and content)
        api_conversation = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation
        ]
        
        return api_conversation
    
    def clear_conversation(self, user_id: int) -> None:
        """
        Clear a user's conversation history.
        
        Args:
            user_id: Telegram user ID
        """
        if user_id in self.conversations:
            del self.conversations[user_id]
            self._save_conversations()
            logger.info(f"Cleared conversation for user {user_id}")
    
    def clear_all_conversations(self) -> None:
        """Clear all conversation history."""
        self.conversations = {}
        self._save_conversations()
        logger.info("Cleared all conversations")
    
    def get_user_ids(self) -> List[int]:
        """
        Get list of all user IDs with conversation history.
        
        Returns:
            List of user IDs
        """
        return list(self.conversations.keys())
    
    def get_conversation_length(self, user_id: int) -> int:
        """
        Get the length of a user's conversation history.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Number of messages in conversation
        """
        if user_id not in self.conversations:
            return 0
        return len(self.conversations[user_id])
    
    def trim_conversation(self, user_id: int, max_messages: int = 50) -> None:
        """
        Trim a conversation to a maximum number of messages (keep most recent).
        
        Args:
            user_id: Telegram user ID
            max_messages: Maximum number of messages to keep
        """
        if user_id in self.conversations and len(self.conversations[user_id]) > max_messages:
            self.conversations[user_id] = self.conversations[user_id][-max_messages:]
            self._save_conversations()
            logger.debug(f"Trimmed conversation for user {user_id} to {max_messages} messages")
    
    def smart_truncate_conversation(self, user_id: int, max_tokens: int = 8000, chars_per_token: int = 4) -> List[Dict[str, Any]]:
        """
        Smart truncate conversation to fit within token limit while preserving context.
        Keeps system messages and recent messages, removes middle messages if needed.
        
        Args:
            user_id: Telegram user ID
            max_tokens: Maximum token limit
            chars_per_token: Estimated characters per token
            
        Returns:
            Truncated conversation
        """
        if user_id not in self.conversations:
            return []
        
        conversation = self.conversations[user_id]
        
        # Calculate token count for each message
        message_tokens = []
        total_tokens = 0
        
        for msg in conversation:
            content = msg.get("content", "")
            tokens = max(1, len(content) // chars_per_token)
            message_tokens.append(tokens)
            total_tokens += tokens
        
        # If within limit, return full conversation
        if total_tokens <= max_tokens:
            return conversation
        
        # Strategy: Keep first message (system/context) and last messages
        # Remove from the middle until within limit
        
        truncated = []
        kept_tokens = 0
        
        # Always keep first message (usually system context)
        if conversation:
            truncated.append(conversation[0])
            kept_tokens += message_tokens[0]
        
        # Add messages from the end until we hit the limit
        for i in range(len(conversation) - 1, 0, -1):  # Start from end, skip first
            if kept_tokens + message_tokens[i] <= max_tokens:
                truncated.insert(1, conversation[i])  # Insert after first message
                kept_tokens += message_tokens[i]
            else:
                # Can't add more without exceeding limit
                break
        
        logger.debug(f"Smart truncated conversation for user {user_id}: {len(conversation)} -> {len(truncated)} messages, {total_tokens} -> {kept_tokens} tokens")
        
        return truncated
    
    def get_conversation_for_api_smart(self, user_id: int, max_tokens: int = 8000) -> List[Dict[str, str]]:
        """
        Get conversation formatted for API with smart truncation based on token limits.
        
        Args:
            user_id: Telegram user ID
            max_tokens: Maximum token limit
            
        Returns:
            List of message dictionaries with only 'role' and 'content' keys
        """
        truncated_conversation = self.smart_truncate_conversation(user_id, max_tokens)
        
        # Format for API (strip timestamp, keep only role and content)
        api_conversation = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in truncated_conversation
        ]
        
        return api_conversation
    
    def estimate_tokens(self, user_id: int, chars_per_token: int = 4) -> int:
        """
        Estimate token count for a user's conversation.
        
        Args:
            user_id: Telegram user ID
            chars_per_token: Estimated characters per token (default 4)
            
        Returns:
            Estimated token count
        """
        if user_id not in self.conversations:
            return 0
        
        total_chars = sum(len(msg["content"]) for msg in self.conversations[user_id])
        return max(1, total_chars // chars_per_token)


# Singleton instance for easy access
_conversation_manager: Optional[ConversationManager] = None

def get_conversation_manager(data_dir: str = "data") -> ConversationManager:
    """
    Get or create the conversation manager singleton.
    
    Args:
        data_dir: Directory to store conversation data
        
    Returns:
        ConversationManager instance
    """
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager(data_dir)
    return _conversation_manager