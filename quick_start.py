"""
Quick start script for eRisk 2026 Task 1.
Demonstrates basic usage of the system.
"""

import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

from src.persona_loader import PersonaLoader
from src.conversation_manager import ConversationManager
from src.depression_detector import DepressionDetector
from src.submission_generator import SubmissionGenerator
from src.utils import format_bdi_severity


def main():
    """Quick start example."""
    print("="*60)
    print("eRisk 2026 Task 1 - Quick Start Example")
    print("="*60)
    
    # Load configuration
    load_dotenv()
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Error: HF_TOKEN not found in .env file")
        sys.exit(1)
    
    # Initialize components
    print("\n1. Initializing components...")
    loader = PersonaLoader(config_path="config/config.yaml", hf_token=hf_token)
    detector = DepressionDetector(config)
    generator = SubmissionGenerator()
    
    # Load persona (example: persona 1)
    persona_id = 1
    print(f"\n2. Loading persona {persona_id}...")
    try:
        model, tokenizer = loader.load_persona(persona_id)
        system_prompt = loader.get_system_prompt(persona_id)
        print(f"   System prompt: {system_prompt[:80]}...")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Note: Ensure you have access to the model and persona adapters")
        sys.exit(1)
    
    # Initialize conversation
    print("\n3. Starting conversation...")
    conv_manager = ConversationManager(model, tokenizer, system_prompt, config)
    
    # Example conversation
    questions = [
        "Hello! How are you feeling today?",
        "What's been on your mind lately?",
        "How have you been sleeping?",
        "What do you do to relax or have fun?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n   Turn {i}: {question}")
        try:
            response = conv_manager.send_message(question)
            print(f"   Persona: {response[:150]}...")
        except Exception as e:
            print(f"   Error: {e}")
            break
    
    # Analyze conversation
    print("\n4. Analyzing conversation for depression indicators...")
    conversation_log = conv_manager.get_conversation_log(persona_id)
    prediction = detector.analyze_conversation(conversation_log['conversation'])
    
    print(f"\n   Results:")
    print(f"   - BDI-II Score: {prediction.bdi_score} ({format_bdi_severity(prediction.bdi_score)})")
    print(f"   - Confidence: {prediction.confidence:.2%}")
    print(f"   - Turn Count: {prediction.turn_count}")
    
    if prediction.key_symptoms:
        print(f"   - Detected Symptoms: {', '.join(prediction.key_symptoms)}")
    else:
        print(f"   - Detected Symptoms: None")
    
    # Generate submission files
    print("\n5. Generating submission files...")
    result = {
        "LLM": str(persona_id),
        "bdi-score": prediction.bdi_score,
        "key-symptoms": prediction.key_symptoms
    }
    
    interactions_path, results_path = generator.generate_submission(
        [conversation_log], [result], run_id=1, is_manual=False
    )
    
    print(f"   ✓ Interactions file: {interactions_path}")
    print(f"   ✓ Results file: {results_path}")
    
    # Validate
    if generator.validate_submission(run_id=1):
        print("\n   ✓ Submission files are valid!")
    
    print("\n" + "="*60)
    print("Quick start completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run 'streamlit run demo/app.py' for interactive demo")
    print("2. Run 'python main.py --all-personas --run-id 1' for batch processing")
    print("3. Check submissions/ directory for generated files")


if __name__ == "__main__":
    main()
