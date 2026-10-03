"""
Main execution script for eRisk 2026 Task 1.
"""

import argparse
import sys
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

from src.persona_loader import PersonaLoader
from src.conversation_manager import ConversationManager
from src.depression_detector import DepressionDetector
from src.submission_generator import SubmissionGenerator
from src.utils import (
    get_conversation_strategies,
    validate_persona_id,
    validate_run_id,
    format_bdi_severity
)


def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_automated_conversation(
    persona_id: int,
    run_id: int,
    config: dict,
    hf_token: str,
    max_turns: int = None
):
    """
    Run automated conversation with a persona.
    
    Args:
        persona_id: Persona ID (1-20)
        run_id: Run ID (1-3)
        config: Configuration dictionary
        hf_token: Hugging Face token
        max_turns: Maximum number of conversation turns
    """
    print(f"\n{'='*60}")
    print(f"Processing Persona {persona_id} - Run {run_id}")
    print(f"{'='*60}\n")
    
    # Initialize components
    loader = PersonaLoader(config_path="config/config.yaml", hf_token=hf_token)
    detector = DepressionDetector(config)
    generator = SubmissionGenerator(
        output_dir=config['submission']['output_dir']
    )
    
    # Load persona
    try:
        model, tokenizer = loader.load_persona(persona_id)
        system_prompt = loader.get_system_prompt(persona_id)
    except Exception as e:
        print(f"Error loading persona {persona_id}: {e}")
        return None, None
    
    # Initialize conversation manager
    conv_manager = ConversationManager(model, tokenizer, system_prompt, config)
    
    # Get conversation strategies
    strategies = get_conversation_strategies()
    
    # Select strategy based on run_id (different strategy per run)
    strategy_idx = (run_id - 1) % len(strategies)
    selected_strategy = strategies[strategy_idx]
    
    print(f"Using strategy: {selected_strategy['name']}")
    print(f"System prompt: {system_prompt[:100]}...\n")
    
    # Run conversation
    if max_turns is None:
        max_turns = config['conversation'].get('max_turns', 50)
    questions = selected_strategy['questions']
    if not questions:
        raise RuntimeError(f"strategy {selected_strategy['name']} has no questions")
    
    for turn in range(max_turns):
        # Select question (cycle through strategy questions)
        question = questions[turn % len(questions)]
        
        print(f"Turn {turn + 1}: {question}")
        
        # Send message
        try:
            response = conv_manager.send_message(question)
            print(f"Persona: {response[:200]}...\n")
        except Exception as e:
            print(f"Error in conversation: {e}")
            break
        
        # Analyze conversation for early stopping
        conversation_log = conv_manager.get_conversation_log(persona_id)
        prediction = detector.analyze_conversation(
            conversation_log['conversation']
        )
        
        # Check if we should stop early
        if detector.should_stop_early(prediction):
            print(f"\nEarly stopping at turn {turn + 1}")
            print(f"Confidence: {prediction.confidence:.2f}")
            print(f"BDI Score: {prediction.bdi_score} ({format_bdi_severity(prediction.bdi_score)})")
            print(f"Detected Symptoms: {', '.join(prediction.key_symptoms) if prediction.key_symptoms else 'None'}")
            break
    
    # Final analysis
    conversation_log = conv_manager.get_conversation_log(persona_id)
    final_prediction = detector.analyze_conversation(
        conversation_log['conversation']
    )
    
    print(f"\n{'='*60}")
    print("Final Prediction:")
    print(f"{'='*60}")
    print(f"BDI Score: {final_prediction.bdi_score} ({format_bdi_severity(final_prediction.bdi_score)})")
    print(f"Key Symptoms: {', '.join(final_prediction.key_symptoms) if final_prediction.key_symptoms else 'None'}")
    print(f"Confidence: {final_prediction.confidence:.2f}")
    print(f"Total Turns: {final_prediction.turn_count}")
    print(f"{'='*60}\n")
    
    # Prepare submission data
    result = {
        "LLM": str(persona_id),
        "bdi-score": final_prediction.bdi_score,
        "key-symptoms": final_prediction.key_symptoms
    }
    
    return conversation_log, result


def process_persona(persona_id: int, run_id: int, config: dict, hf_token: str):
    """Process a single persona."""
    conversation_log, result = run_automated_conversation(
        persona_id, run_id, config, hf_token
    )
    
    if conversation_log is None or result is None:
        print(f"Failed to process persona {persona_id}")
        return None, None
    
    return conversation_log, result


def process_all_personas(run_id: int, config: dict, hf_token: str, persona_range: tuple = (1, 20)):
    """Process all personas in the specified range."""
    start_id, end_id = persona_range
    
    all_conversations = []
    all_results = []
    
    for persona_id in range(start_id, end_id + 1):
        try:
            conv_log, result = process_persona(persona_id, run_id, config, hf_token)
            if conv_log and result:
                all_conversations.append(conv_log)
                all_results.append(result)
        except Exception as e:
            print(f"Error processing persona {persona_id}: {e}")
            continue
    
    return all_conversations, all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="eRisk 2026 Task 1: Conversational Depression Detection"
    )
    parser.add_argument(
        "--persona-id",
        type=int,
        help="Persona ID to process (1-20)"
    )
    parser.add_argument(
        "--all-personas",
        action="store_true",
        help="Process all personas"
    )
    parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Run ID (1-3)"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Mark as manual run"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum conversation turns"
    )
    
    args = parser.parse_args()
    
    # Validate run_id
    if not validate_run_id(args.run_id):
        print(f"Invalid run_id: {args.run_id}. Must be between 1 and 3.")
        sys.exit(1)
    
    # Load configuration
    load_dotenv()
    config = load_config()
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        print("Error: HF_TOKEN not found in environment variables")
        sys.exit(1)
    
    # Initialize submission generator
    generator = SubmissionGenerator(
        output_dir=config['submission']['output_dir']
    )
    
    # Process personas
    if args.all_personas:
        print("Processing all personas...")
        conversations, results = process_all_personas(
            args.run_id, config, hf_token
        )
        
        # Generate submission files
        if conversations and results:
            generator.generate_submission(
                conversations, results, args.run_id, args.manual
            )
            generator.validate_submission(args.run_id, args.manual)
    
    elif args.persona_id:
        if not validate_persona_id(args.persona_id):
            print(f"Invalid persona_id: {args.persona_id}. Must be between 1 and 20.")
            sys.exit(1)
        
        conversation_log, result = process_persona(
            args.persona_id, args.run_id, config, hf_token
        )
        
        if conversation_log and result:
            # For single persona, still generate files (will contain one entry)
            generator.generate_submission(
                [conversation_log], [result], args.run_id, args.manual
            )
            generator.validate_submission(args.run_id, args.manual)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
