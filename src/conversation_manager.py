"""
Manage conversations with LLM personas for depression detection.
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # "user" or "assistant"
    message: str
    turn: int


class ConversationManager:
    """Manage conversations with LLM personas."""
    
    def __init__(self, model, tokenizer, system_prompt: str, config: dict):
        """
        Initialize conversation manager.
        
        Args:
            model: Loaded persona model
            tokenizer: Tokenizer for the model
            system_prompt: Official system prompt (must be used verbatim)
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.config = config
        
        self.max_new_tokens = config['model'].get('max_new_tokens', 512)
        self.temperature = config['model'].get('temperature', 0.7)
        self.top_p = config['model'].get('top_p', 0.9)
        
        self.conversation_history: List[Message] = []
        self.turn_count = 0
    
    def format_conversation(self, user_message: str) -> str:
        """
        Format conversation with system prompt and history.
        
        Args:
            user_message: New user message
            
        Returns:
            Formatted prompt string
        """
        # Build conversation context
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Add conversation history
        for msg in self.conversation_history:
            messages.append({
                "role": msg.role,
                "content": msg.message
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Format using tokenizer's chat template (with fallback for small models)
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple instruction format (e.g. SmolLM2, LLaMA-style)
            parts = []
            for m in messages:
                role, content = m["role"], m["content"]
                if role == "system":
                    parts.append(f"<|system|>\n{content}\n")
                elif role == "user":
                    parts.append(f"<|user|>\n{content}\n")
                elif role == "assistant":
                    parts.append(f"<|assistant|>\n{content}\n")
            parts.append("<|assistant|>\n")
            formatted = "".join(parts)
        
        return formatted
    
    def send_message(self, user_message: str) -> str:
        """
        Send a message to the persona and get response.
        
        Args:
            user_message: User's message
            
        Returns:
            Persona's response
        """
        if not user_message or not user_message.strip():
            raise ValueError("user message is empty")

        # Format before appending so the new turn is not sent twice.
        self.turn_count += 1
        formatted_prompt = self.format_conversation(user_message)
        user_msg = Message(role="user", message=user_message, turn=self.turn_count)
        self.conversation_history.append(user_msg)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Add assistant response to history
        assistant_msg = Message(role="assistant", message=generated_text, turn=self.turn_count)
        self.conversation_history.append(assistant_msg)
        
        return generated_text
    
    def get_conversation_log(self, persona_id: int) -> Dict:
        """
        Get conversation log in submission format.
        
        Args:
            persona_id: Persona ID
            
        Returns:
            Dictionary in submission format
        """
        conversation = []
        for msg in self.conversation_history:
            conversation.append({
                "role": msg.role,
                "message": msg.message
            })
        
        return {
            "LLM": str(persona_id),
            "conversation": conversation
        }
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.turn_count = 0
    
    def get_turn_count(self) -> int:
        """Get current number of turns."""
        return self.turn_count
    
    def get_conversation_text(self) -> str:
        """Get full conversation as text."""
        text_parts = []
        for msg in self.conversation_history:
            role_label = "User" if msg.role == "user" else "Persona"
            text_parts.append(f"{role_label}: {msg.message}")
        return "\n".join(text_parts)
