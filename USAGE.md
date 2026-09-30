# Usage Guide

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts/test_setup.py
```

### 2. Configuration

Ensure your `.env` file contains:
```
HUGGINGFACE_TOKEN=your_token_here
```

### 3. Run Demo

```bash
streamlit run demo/app.py
```

## Command Line Usage

### Process a Single Persona

```bash
python main.py --persona-id 1 --run-id 1
```

### Process All Personas

```bash
python main.py --all-personas --run-id 1
```

### Manual Run

```bash
python main.py --persona-id 1 --run-id 1 --manual
```

### Options

- `--persona-id`: Persona ID (1-20)
- `--all-personas`: Process all available personas
- `--run-id`: Run identifier (1-3, required)
- `--manual`: Mark as manual run (optional)
- `--max-turns`: Maximum conversation turns (optional)

## Web Demo

The Streamlit demo provides an interactive interface:

1. **Load Persona**: Select persona ID and click "Load Persona"
2. **Chat**: Type messages to interact with the persona
3. **Analyze**: Click "Analyze Conversation" to get depression predictions
4. **Export**: Generate and download submission files

## Submission Files

Submission files are generated in `./submissions/`:

- `interactions_run<id>.json`: Conversation logs
- `results_run<id>.json`: Depression predictions

### File Format

**interactions_run1.json:**
```json
[
  {
    "LLM": "1",
    "conversation": [
      {"role": "user", "message": "Hello. How are you?"},
      {"role": "assistant", "message": "I'm okay, I guess."}
    ]
  }
]
```

**results_run1.json:**
```json
[
  {
    "LLM": "1",
    "bdi-score": 18,
    "key-symptoms": ["Sadness", "Loss of Interest", "Fatigue"]
  }
]
```

## Automated Strategies

The system uses different conversation strategies for each run:

1. **Run 1**: General wellbeing questions
2. **Run 2**: Sleep and energy focus
3. **Run 3**: Activities and interests focus

Each strategy avoids directly asking about depression and instead infers from natural conversation.

## Early Stopping

The system implements early stopping when:
- Confidence threshold is reached (default: 0.85)
- Minimum turns completed (default: 3)

This optimizes for early detection, which is rewarded in evaluation.

## Troubleshooting

### Model Access Issues

If you get access errors:
1. Request access to `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face
2. Ensure your token has proper permissions
3. Check that personas are available in the collection

### Memory Issues

If running out of GPU memory:
- Set `LOAD_IN_8BIT: true` in `config/config.yaml`
- Process personas one at a time
- Use CPU mode (slower but uses less memory)

### Persona Not Found

Personas are released weekly. If a persona isn't available:
- Check the Hugging Face collection: https://huggingface.co/collections/irlab-udc/erisk2026
- Wait for the weekly release
- Process only available personas

## Best Practices

1. **Test First**: Use the demo to test with one persona before batch processing
2. **Validate**: Always validate submission files before uploading
3. **Early Detection**: Aim for accurate predictions with fewer turns
4. **No Direct Questions**: Never ask directly about depression
5. **Natural Flow**: Maintain natural conversation flow

## Evaluation Metrics

The task evaluates:
- **Accuracy**: Correct depression identification
- **Early Detection**: Fewer turns = better score (if accurate)
- **Symptom Detection**: Correct identification of BDI-II symptoms
- **BDI Score Accuracy**: Closeness to ground truth BDI-II scores
