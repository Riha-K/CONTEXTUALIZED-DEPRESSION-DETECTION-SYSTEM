# CLEF eRisk 2026 - Task 1: Conversational Depression Detection

A complete research project for detecting depression through conversational interactions with LLM personas using Small Language Models (SLM).

## Overview

This project implements a system for the CLEF eRisk 2026 Task 1 challenge, which involves:
- Interacting with LLM personas (LoRA adapters) fine-tuned on **meta-llama/Meta-Llama-3-8B-Instruct** (SLM: 8B parameters)
- Detecting depression through natural conversation without directly asking about mental health
- Using **SLM-based analysis** (`all-MiniLM-L6-v2` for semantic similarity)
- Predicting BDI-II scores and identifying key depressive symptoms
- Submitting conversation logs and classification results
- **Runs entirely locally** - CPU or GPU support with automatic detection

## Project Structure

```
erisk-ppt/
├── src/
│   ├── __init__.py
│   ├── persona_loader.py      # Load LoRA adapters and personas
│   ├── conversation_manager.py # Manage conversations with personas
│   ├── depression_detector.py  # Analyze conversations and predict depression
│   ├── submission_generator.py # Generate submission JSON files
│   └── utils.py                # Utility functions
├── demo/
│   ├── app.py                  # Streamlit web demo
│   └── static/                 # Static assets
├── config/
│   └── config.yaml             # Configuration file
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── main.py                    # Main execution script
└── README.md                  # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your Hugging Face token
```

3. Run the demo:
```bash
streamlit run demo/app.py
```

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify SLM usage and setup:**
```bash
python scripts/verify_slm_usage.py  # Verify SLM models
python scripts/test_setup.py        # Test configuration
```

3. **Run quick start example:**
```bash
python quick_start.py
```

4. **Launch web demo (runs locally):**
```bash
streamlit run demo/app.py
```

**Note**: The system automatically detects GPU/CPU and runs entirely locally. See [LOCAL_EXECUTION.md](LOCAL_EXECUTION.md) for details.

## Usage

### Command Line Interface

```bash
# Interact with a specific persona
python main.py --persona-id 1 --run-id 1

# Process all available personas
python main.py --all-personas --run-id 1

# Manual run (human-in-the-loop)
python main.py --persona-id 1 --run-id 1 --manual
```

### Web Demo

Launch the Streamlit demo for interactive testing:
```bash
streamlit run demo/app.py
```

The demo provides:
- Interactive chat interface
- Real-time depression analysis
- Submission file generation
- Conversation statistics

## Submission Format

The system generates two JSON files per run:
- `interactions_run<id>.json`: Full conversation logs
- `results_run<id>.json`: Depression predictions with BDI-II scores and symptoms

## Task Constraints

- Must use official system prompt verbatim (no modifications)
- Cannot directly ask about depression
- Early detection is rewarded (fewer messages = better score)
- Up to 3 runs per persona (max 1 manual run)

## References

- [Task Details](https://erisk.irlab.org/Task1LLMs.html)
- [Hugging Face Collection](https://huggingface.co/collections/irlab-udc/erisk2026)
- [BDI-II Symptoms Reference](https://erisk.irlab.org/Task3.html)
