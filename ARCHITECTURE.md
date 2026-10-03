# Architecture Documentation

## System Overview

The eRisk 2026 Task 1 system is designed to detect depression through conversational interactions with LLM personas. The architecture follows a modular design with clear separation of concerns.

## Components

### 1. PersonaLoader (`src/persona_loader.py`)

**Responsibility**: Load and manage LLM personas (LoRA adapters)

**Key Features**:
- Loads base model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Loads LoRA adapters from Hugging Face collection
- Caches loaded personas for efficiency
- Retrieves official system prompts

**Methods**:
- `load_persona(persona_id)`: Load a specific persona
- `get_system_prompt(persona_id)`: Get official system prompt
- `unload_persona(persona_id)`: Free memory

### 2. ConversationManager (`src/conversation_manager.py`)

**Responsibility**: Manage conversations with personas

**Key Features**:
- Formats conversations with system prompt
- Handles message exchange
- Maintains conversation history
- Generates conversation logs in submission format

**Methods**:
- `send_message(user_message)`: Send message and get response
- `get_conversation_log(persona_id)`: Get formatted log
- `reset()`: Clear conversation history

### 3. DepressionDetector (`src/depression_detector.py`)

**Responsibility**: Analyze conversations and predict depression

**Key Features**:
- Detects BDI-II symptoms using keyword and semantic matching
- Calculates BDI-II scores (0-63)
- Identifies key symptoms (max 4)
- Implements early stopping logic

**Methods**:
- `analyze_conversation(conversation_history)`: Analyze and predict
- `should_stop_early(prediction)`: Determine if early stopping is appropriate

**Detection Strategy**:
- Keyword pattern matching for 21 BDI-II symptoms
- Semantic similarity using sentence transformers
- Sentiment and severity analysis
- Confidence scoring

### 4. SubmissionGenerator (`src/submission_generator.py`)

**Responsibility**: Generate submission files

**Key Features**:
- Creates interactions JSON files
- Creates results JSON files
- Validates submission format
- Handles manual vs automated runs

**Methods**:
- `generate_interactions_file()`: Create conversation log
- `generate_results_file()`: Create predictions file
- `validate_submission()`: Validate format

### 5. Utils (`src/utils.py`)

**Responsibility**: Utility functions

**Features**:
- Conversation strategies
- BDI severity formatting
- Validation helpers
- Configuration loading

## Data Flow

```
1. Load Persona
   PersonaLoader → Load base model + LoRA adapter
   
2. Initialize Conversation
   ConversationManager → Setup with system prompt
   
3. Interact
   User → ConversationManager → Persona Model → Response
   
4. Analyze
   Conversation History → DepressionDetector → Prediction
   
5. Generate Submission
   Prediction + Logs → SubmissionGenerator → JSON Files
```

## Configuration

Configuration is managed through:
- `config/config.yaml`: Main configuration
- `.env`: Environment variables (tokens, paths)

## Conversation Strategies

The system uses different strategies per run:

1. **General Wellbeing**: Broad questions about feelings
2. **Sleep & Energy**: Focus on physical symptoms
3. **Activities & Interests**: Focus on behavioral changes
4. **Social & Relationships**: Focus on interpersonal
5. **Thoughts & Feelings**: Focus on cognitive symptoms
6. **Daily Life**: Focus on routine and functioning

## Detection Algorithm

### Symptom Detection

1. **Keyword Matching**: Match against symptom-specific patterns
2. **Semantic Similarity**: Use sentence transformers for context
3. **Negation Handling**: Detect negations that invalidate symptoms
4. **Confidence Scoring**: Weighted combination of signals

### BDI-II Scoring

- Base score from depression indicators
- Symptom contribution (proportional to detection strength)
- Severity modifiers (very, extremely, etc.)
- Clamped to 0-63 range

### Early Stopping

Stops conversation when:
- Confidence ≥ threshold (default: 0.85)
- Minimum turns completed (default: 3)
- Balances accuracy vs. efficiency

## File Structure

```
erisk-ppt/
├── src/                    # Core modules
├── demo/                   # Streamlit demo
├── scripts/                # Utility scripts
├── config/                 # Configuration
├── submissions/            # Generated files (gitignored)
├── main.py                 # CLI entry point
└── requirements.txt        # Dependencies
```

## Dependencies

### Core
- `torch`: PyTorch for model inference
- `transformers`: Hugging Face transformers
- `peft`: Parameter-efficient fine-tuning (LoRA)

### Analysis
- `sentence-transformers`: Semantic similarity
- `scikit-learn`: ML utilities
- `numpy`: Numerical operations

### Interface
- `streamlit`: Web demo
- `pyyaml`: Configuration parsing
- `python-dotenv`: Environment variables

## Performance Considerations

### Memory Management
- 8-bit quantization for base model
- LoRA adapters loaded on-demand
- Cache clearing after processing

### GPU Usage
- Automatic device mapping
- Batch processing support
- CPU fallback available

### Optimization
- Early stopping reduces computation
- Caching prevents redundant loads
- Efficient tokenization

## Extensibility

The modular design allows easy extension:

1. **New Detection Methods**: Extend `DepressionDetector`
2. **New Strategies**: Add to `get_conversation_strategies()`
3. **New Models**: Extend `PersonaLoader`
4. **New Formats**: Extend `SubmissionGenerator`

## Security & Privacy

- Tokens stored in `.env` (gitignored)
- No data persistence beyond submissions
- Conversation logs only for evaluation
- No external API calls beyond Hugging Face

## Future Improvements

1. **Multi-turn Context**: Better context understanding
2. **Adaptive Strategies**: Dynamic strategy selection
3. **Ensemble Methods**: Combine multiple detection approaches
4. **Fine-tuning**: Fine-tune detection on validation data
5. **Real-time Feedback**: Live confidence updates
