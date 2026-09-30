# Project Summary: eRisk 2026 Task 1

## Overview

This project implements a complete research system for **CLEF eRisk 2026 Task 1: Conversational Depression Detection with LLM Personas**. The system detects depression through natural conversations with fine-tuned LLM personas without directly asking about mental health.

## Key Features

✅ **Complete Implementation**
- LoRA adapter loading and persona management
- Natural conversation interface
- Depression detection using SLM-based analysis
- BDI-II scoring (0-63)
- Symptom identification (up to 4 key symptoms)
- Submission file generation
- Interactive web demo

✅ **Compliance with Task Requirements**
- Uses official system prompt verbatim
- No direct questions about depression
- Early detection optimization
- Proper JSON submission format
- Support for automated and manual runs

✅ **Production Ready**
- Modular architecture
- Comprehensive error handling
- Configuration management
- Validation and testing
- Documentation

## Project Structure

```
erisk-ppt/
├── src/                          # Core modules
│   ├── persona_loader.py         # Load LoRA adapters
│   ├── conversation_manager.py   # Manage conversations
│   ├── depression_detector.py    # Detect depression
│   ├── submission_generator.py  # Generate submissions
│   └── utils.py                  # Utilities
├── demo/
│   └── app.py                    # Streamlit web demo
├── scripts/
│   ├── test_setup.py             # Setup verification
│   ├── download_personas.py     # Download adapters
│   ├── setup.sh                  # Linux/Mac setup
│   └── setup.bat                 # Windows setup
├── config/
│   └── config.yaml               # Configuration
├── main.py                       # CLI entry point
├── quick_start.py                # Quick start example
├── requirements.txt              # Dependencies
├── .env                          # Environment variables
└── Documentation files...
```

## Technology Stack

- **Base Model**: meta-llama/Meta-Llama-3-8B-Instruct
- **Fine-tuning**: LoRA adapters (PEFT)
- **Analysis**: Sentence Transformers for semantic similarity
- **Interface**: Streamlit for web demo
- **Language**: Python 3.8+

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   - Edit `.env` and add your Hugging Face token

3. **Test setup:**
   ```bash
   python scripts/test_setup.py
   ```

4. **Run demo:**
   ```bash
   streamlit run demo/app.py
   ```

5. **Process personas:**
   ```bash
   python main.py --all-personas --run-id 1
   ```

## Detection Methodology

### Symptom Detection
- **Keyword Matching**: Pattern-based detection for 21 BDI-II symptoms
- **Semantic Analysis**: Sentence transformers for context understanding
- **Negation Handling**: Detects negations that invalidate symptoms
- **Confidence Scoring**: Weighted combination of signals

### BDI-II Scoring
- Base score from depression indicators
- Symptom contribution (proportional to detection strength)
- Severity modifiers
- Clamped to 0-63 range

### Early Stopping
- Stops when confidence ≥ threshold (default: 0.85)
- Minimum turns required (default: 3)
- Optimizes for early accurate detection

## Conversation Strategies

The system uses different strategies per run:

1. **General Wellbeing**: Broad questions about feelings
2. **Sleep & Energy**: Physical symptoms focus
3. **Activities & Interests**: Behavioral changes focus
4. **Social & Relationships**: Interpersonal focus
5. **Thoughts & Feelings**: Cognitive symptoms focus
6. **Daily Life**: Routine and functioning focus

## Submission Format

### Interactions File (`interactions_run<id>.json`)
```json
[
  {
    "LLM": "1",
    "conversation": [
      {"role": "user", "message": "..."},
      {"role": "assistant", "message": "..."}
    ]
  }
]
```

### Results File (`results_run<id>.json`)
```json
[
  {
    "LLM": "1",
    "bdi-score": 18,
    "key-symptoms": ["Sadness", "Loss of Interest", "Fatigue"]
  }
]
```

## Evaluation Metrics

The task evaluates:
- **Accuracy**: Correct depression identification
- **Early Detection**: Fewer turns = better score (if accurate)
- **Symptom Detection**: Correct identification of BDI-II symptoms
- **BDI Score Accuracy**: Closeness to ground truth BDI-II scores

## Key Constraints

⚠️ **Important Rules:**
- Must use official system prompt verbatim
- Cannot directly ask about depression
- Must infer from language, tone, and thoughts
- Early detection is rewarded
- Up to 3 runs per persona (max 1 manual)

## Files Generated

- `interactions_run<id>.json`: Full conversation logs
- `results_run<id>.json`: Depression predictions
- Files saved in `./submissions/` directory

## Performance Considerations

- **Memory**: 8-bit quantization for efficient GPU usage
- **Speed**: Early stopping reduces computation
- **Caching**: Personas cached for efficiency
- **Scalability**: Batch processing support

## Documentation

- **README.md**: Main documentation
- **USAGE.md**: Detailed usage guide
- **ARCHITECTURE.md**: System architecture
- **EXAMPLES.md**: Code examples
- **BDI_SYMPTOMS.md**: BDI-II symptoms reference

## Future Enhancements

Potential improvements:
1. Multi-turn context understanding
2. Adaptive strategy selection
3. Ensemble detection methods
4. Fine-tuning on validation data
5. Real-time confidence updates

## Support

For issues or questions:
1. Check documentation files
2. Run `python scripts/test_setup.py` for diagnostics
3. Review task guidelines: https://erisk.irlab.org/Task1LLMs.html

## License

This project is developed for CLEF eRisk 2026 Task 1 research purposes.

## Acknowledgments

- CLEF eRisk 2026 organizers
- Hugging Face for model hosting
- Meta for LLaMA 3 base model
