"""
Load and manage LLM personas (LoRA adapters) for eRisk 2026 Task 1.
"""

import os
import torch
from typing import Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from huggingface_hub import login, hf_hub_download
import yaml


class PersonaLoader:
    """Load and manage LoRA adapter personas."""
    
    def __init__(self, config_path: str = "config/config.yaml", hf_token: Optional[str] = None):
        """
        Initialize the persona loader.
        
        Args:
            config_path: Path to configuration YAML file
            hf_token: Hugging Face authentication token
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        if self.hf_token:
            login(token=self.hf_token)
        
        self.base_model_name = self.config['model']['base_model']
        self.hf_collection = self.config['model'].get('hf_collection', 'irlab-udc/erisk2026')
        self.use_open_slm = self.config['model'].get('use_open_slm', False)
        # Auto-detect device: use GPU if available, otherwise CPU
        device_config = self.config['model'].get('device', 'auto')
        if device_config == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_config
        # Only use 8-bit quantization if GPU is available
        self.load_in_8bit = self.config['model'].get('load_in_8bit', False) and torch.cuda.is_available()
        
        self.base_model = None
        self.base_tokenizer = None
        self.loaded_personas = {}  # Cache for loaded personas
        
    def _load_base_model(self):
        """Load the base model and tokenizer."""
        if self.base_model is not None:
            return
        
        print(f"Loading base model: {self.base_model_name}")
        
        # Configure quantization if needed
        if self.load_in_8bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        # Load base model
        print(f"Loading on device: {self.device}")
        if self.device == 'cpu':
            # CPU mode: float32, low_cpu_mem_usage for faster load
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                token=self.hf_token,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.base_model = self.base_model.to('cpu')
        else:
            # GPU mode: use quantization and device_map
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                token=self.hf_token,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        print(f"Base model loaded successfully on {self.device}")
    
    def load_persona(self, persona_id: int) -> tuple:
        """
        Load a specific persona by ID.
        When use_open_slm is True: uses open base model only (no LoRA).
        Otherwise: loads base model + eRisk LoRA adapter (requires gated LLaMA access).
        
        Args:
            persona_id: Persona ID (1-20)
            
        Returns:
            Tuple of (model, tokenizer) for the persona
        """
        if persona_id in self.loaded_personas:
            return self.loaded_personas[persona_id]
        
        # Load base model if not already loaded
        self._load_base_model()
        
        if self.use_open_slm:
            # Open SLM mode: use base model only (no LoRA - works without gated access)
            self.base_model.eval()
            persona_tuple = (self.base_model, self.base_tokenizer)
            self.loaded_personas[persona_id] = persona_tuple
            print(f"Persona {persona_id} loaded (open SLM base model)")
            return persona_tuple
        
        # Original: try to load LoRA adapter (requires meta-llama/Meta-Llama-3-8B-Instruct access)
        persona_name = f"persona{persona_id}"
        adapter_path = f"{self.hf_collection}/{persona_name}"
        print(f"Loading persona {persona_id} from {adapter_path}")
        
        try:
            model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                token=self.hf_token
            )
            model.eval()
            persona_tuple = (model, self.base_tokenizer)
            self.loaded_personas[persona_id] = persona_tuple
            print(f"Persona {persona_id} loaded successfully")
            return persona_tuple
        except Exception as e:
            print(f"Error loading persona {persona_id}: {e}")
            raise
    
    def get_system_prompt(self, persona_id: int) -> str:
        """
        Get the official system prompt for a persona.
        Must be used verbatim according to task rules.
        
        Args:
            persona_id: Persona ID
            
        Returns:
            System prompt string
        """
        if self.use_open_slm:
            # Varied prompts per persona for demo (simulate different conversation styles)
            prompts = [
                "You are a helpful person. Reply naturally and briefly to questions about how you feel and what you've been doing.",
                "You are someone having a casual chat. Answer honestly about your mood, sleep, and daily life in a natural way.",
                "You are talking to a friend. Share how you've been feeling and what's on your mind in an authentic way.",
                "You are in a conversation. Respond naturally about your feelings, energy, and interests.",
                "You are chatting openly. Answer questions about your day, mood, and thoughts in a genuine way.",
            ]
            return prompts[(persona_id - 1) % len(prompts)]
        
        # Try to load from adapter repository (when using eRisk LoRA)
        try:
            persona_name = f"persona{persona_id}"
            adapter_path = f"{self.hf_collection}/{persona_name}"
            readme_path = hf_hub_download(
                repo_id=adapter_path,
                filename="README.md",
                token=self.hf_token
            )
            with open(readme_path, 'r') as f:
                return self._extract_system_prompt_from_readme(f.read())
        except Exception:
            return (
                "You are a helpful assistant. Respond naturally and authentically "
                "to the user's questions and statements."
            )
    
    def _extract_system_prompt_from_readme(self, readme_content: str) -> str:
        """Extract system prompt from README content."""
        # This is a placeholder - actual implementation depends on README format
        # Look for system prompt markers in the README
        lines = readme_content.split('\n')
        prompt_lines = []
        in_prompt = False
        
        for line in lines:
            if 'system prompt' in line.lower() or 'system_prompt' in line.lower():
                in_prompt = True
                continue
            if in_prompt:
                if line.strip().startswith('```') or line.strip().startswith('---'):
                    break
                if line.strip():
                    prompt_lines.append(line.strip())
        
        if prompt_lines:
            return '\n'.join(prompt_lines)
        
        # Default fallback
        return (
            "You are a helpful assistant. Respond naturally and authentically "
            "to the user's questions and statements."
        )
    
    def unload_persona(self, persona_id: int):
        """Unload a persona from memory."""
        if persona_id in self.loaded_personas:
            del self.loaded_personas[persona_id]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"Persona {persona_id} unloaded")
    
    def clear_cache(self):
        """Clear all loaded personas from memory."""
        self.loaded_personas.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("All personas cleared from cache")
