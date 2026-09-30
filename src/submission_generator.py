"""
Generate submission files in the required format.
"""

import json
import os
from typing import List, Dict
from pathlib import Path


class SubmissionGenerator:
    """Generate submission JSON files."""
    
    def __init__(self, output_dir: str = "./submissions"):
        """
        Initialize submission generator.
        
        Args:
            output_dir: Directory to save submission files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_interactions_file(
        self,
        conversations: List[Dict],
        run_id: int,
        is_manual: bool = False
    ) -> str:
        """
        Generate interactions log file.
        
        Args:
            conversations: List of conversation logs (from ConversationManager)
            run_id: Run identifier (1, 2, or 3)
            is_manual: Whether this is a manual run
            
        Returns:
            Path to generated file
        """
        prefix = "manual_" if is_manual else ""
        filename = f"{prefix}interactions_run{run_id}.json"
        filepath = self.output_dir / filename
        
        # Ensure conversations are in correct format
        formatted_conversations = []
        for conv in conversations:
            if isinstance(conv, dict) and "LLM" in conv and "conversation" in conv:
                formatted_conversations.append(conv)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(formatted_conversations, f, indent=2, ensure_ascii=False)
        
        print(f"Generated interactions file: {filepath}")
        return str(filepath)
    
    def generate_results_file(
        self,
        predictions: List[Dict],
        run_id: int,
        is_manual: bool = False
    ) -> str:
        """
        Generate results file with predictions.
        
        Args:
            predictions: List of prediction dictionaries with:
                - LLM: persona ID (string)
                - bdi-score: integer (0-63)
                - key-symptoms: list of strings (max 4)
            run_id: Run identifier (1, 2, or 3)
            is_manual: Whether this is a manual run
            
        Returns:
            Path to generated file
        """
        prefix = "manual_" if is_manual else ""
        filename = f"{prefix}results_run{run_id}.json"
        filepath = self.output_dir / filename
        
        # Ensure predictions are in correct format
        formatted_predictions = []
        for pred in predictions:
            formatted_pred = {
                "LLM": str(pred.get("LLM", "")),
                "bdi-score": int(pred.get("bdi-score", 0)),
                "key-symptoms": list(pred.get("key-symptoms", []))[:4]  # Max 4 symptoms
            }
            formatted_predictions.append(formatted_pred)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(formatted_predictions, f, indent=2, ensure_ascii=False)
        
        print(f"Generated results file: {filepath}")
        return str(filepath)
    
    def generate_submission(
        self,
        conversations: List[Dict],
        predictions: List[Dict],
        run_id: int,
        is_manual: bool = False
    ) -> tuple:
        """
        Generate both interaction and results files for a run.
        
        Args:
            conversations: List of conversation logs
            predictions: List of predictions
            run_id: Run identifier
            is_manual: Whether this is a manual run
            
        Returns:
            Tuple of (interactions_filepath, results_filepath)
        """
        interactions_path = self.generate_interactions_file(
            conversations, run_id, is_manual
        )
        results_path = self.generate_results_file(
            predictions, run_id, is_manual
        )
        
        return interactions_path, results_path
    
    def validate_submission(self, run_id: int, is_manual: bool = False) -> bool:
        """
        Validate submission files.
        
        Args:
            run_id: Run identifier
            is_manual: Whether this is a manual run
            
        Returns:
            True if valid, False otherwise
        """
        prefix = "manual_" if is_manual else ""
        interactions_file = self.output_dir / f"{prefix}interactions_run{run_id}.json"
        results_file = self.output_dir / f"{prefix}results_run{run_id}.json"
        
        # Check files exist
        if not interactions_file.exists() or not results_file.exists():
            print(f"Missing submission files for run {run_id}")
            return False
        
        # Validate JSON format
        try:
            with open(interactions_file, 'r') as f:
                interactions = json.load(f)
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Validate structure
            if not isinstance(interactions, list):
                print("Interactions file must be a list")
                return False
            
            if not isinstance(results, list):
                print("Results file must be a list")
                return False
            
            # Validate each interaction
            for conv in interactions:
                if "LLM" not in conv or "conversation" not in conv:
                    print("Invalid interaction format")
                    return False
                if not isinstance(conv["conversation"], list):
                    print("Conversation must be a list")
                    return False
            
            # Validate each result
            for result in results:
                if "LLM" not in result or "bdi-score" not in result or "key-symptoms" not in result:
                    print("Invalid result format")
                    return False
                if not isinstance(result["key-symptoms"], list):
                    print("key-symptoms must be a list")
                    return False
                if len(result["key-symptoms"]) > 4:
                    print("key-symptoms must have at most 4 items")
                    return False
            
            print(f"Submission files for run {run_id} are valid")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            print(f"Validation error: {e}")
            return False
