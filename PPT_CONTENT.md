# PowerPoint Presentation Content
## eRisk 2026 Task 1: Conversational Depression Detection

---

## SLIDE 1: Title Slide

### 🧠 eRisk 2026 Task 1
# Conversational Depression Detection with SLM-Based Analysis

**CLEF eRisk 2026 Challenge**

**Project Overview:**
- Detect depression through natural AI conversations
- Predict BDI-II scores and identify symptoms
- Early detection optimization
- Fully automated and manual run support

**Team/Author:** [Your Name/Team]
**Date:** February 2026

---

## SLIDE 2: Models Used - Small Language Models (SLMs)

### 🤖 Model Architecture

**Conversation Model: Meta-Llama-3-8B-Instruct**
```
┌─────────────────────────────────────┐
│  Base Model: LLaMA 3 8B Instruct   │
│  Parameters: 8 Billion (SLM)       │
│  Fine-tuning: LoRA Adapters         │
│  Memory: ~8GB VRAM (8-bit quant)    │
│  Purpose: Persona Conversations     │
└─────────────────────────────────────┘
```

**Detection Model: all-MiniLM-L6-v2**
```
┌─────────────────────────────────────┐
│  Sentence Transformer              │
│  Parameters: 22 Million (SLM)      │
│  Size: ~90MB                       │
│  Purpose: Semantic Analysis        │
└─────────────────────────────────────┘
```

**Why SLMs?**
- ✅ Efficient inference (< 10B parameters)
- ✅ Runs locally (CPU or GPU)
- ✅ Privacy-preserving (no external APIs)
- ✅ Cost-effective (no API costs)
- ✅ Fast response times

**Comparison:**
- SLM (8B): Fast, efficient, local
- LLM (70B+): Slow, expensive, requires cloud

---

## SLIDE 3: System Architecture

### 🏗️ Architecture Diagram

```
                    USER INTERFACE
              (Web Demo / CLI / API)
                         │
                         ▼
              ┌──────────────────────┐
              │   PersonaLoader      │
              │  • Load LoRA         │
              │  • Manage Personas   │
              │  • System Prompts    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ ConversationManager │
              │  • Format Messages  │
              │  • Generate Response │
              │  • Track History     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  DepressionDetector   │
              │  • Symptom Detection │
              │  • BDI-II Scoring    │
              │  • Early Stopping    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ SubmissionGenerator  │
              │  • JSON Formatting   │
              │  • Validation        │
              └──────────────────────┘
```

**Key Components:**

1. **PersonaLoader**
   - Loads base model (Meta-Llama-3-8B)
   - Manages 20 LoRA persona adapters
   - Auto device detection (GPU/CPU)

2. **ConversationManager**
   - Formats conversations
   - Manages message history
   - Generates persona responses

3. **DepressionDetector**
   - Keyword pattern matching
   - Semantic similarity (SLM)
   - BDI-II score calculation
   - Early stopping logic

4. **SubmissionGenerator**
   - Creates interaction logs
   - Generates prediction files
   - Validates format

---

## SLIDE 4: What It Does - Functionality

### 🔍 System Workflow & Capabilities

**1. Persona Interaction**
```
User: "How are you feeling today?"
     ↓
Persona: "Not great, honestly. Just feeling down."
     ↓
System: Analyzes response for depression indicators
```

**2. Depression Detection Process**
- **Symptom Detection** (21 BDI-II symptoms):
  - Keyword matching: "sad", "tired", "hopeless"
  - Semantic analysis: Context understanding
  - Confidence scoring: Weighted signals
  
- **BDI-II Scoring** (0-63):
  ```
  0-13:   Minimal depression
  14-19:  Mild depression
  20-28:  Moderate depression
  29-63:  Severe depression
  ```

- **Key Symptoms**: Identifies top 4 symptoms

**3. Early Detection**
- Monitors confidence after each turn
- Stops when confidence ≥ 0.85
- Optimizes for early accurate detection
- Reduces unnecessary conversation turns

**4. Output Generation**
```json
{
  "LLM": "1",
  "bdi-score": 18,
  "key-symptoms": [
    "Sadness",
    "Loss of Interest", 
    "Fatigue"
  ]
}
```

**Example Detection:**
- Conversation: 5 turns
- Detected: Moderate depression (BDI: 22)
- Symptoms: Sadness, Fatigue, Loss of Interest, Sleep Issues
- Confidence: 87%

---

## SLIDE 5: SLM Usage & How to Use

### 🚀 Implementation & Usage Guide

**SLM Advantages:**
- ✅ **Local Execution**: Runs entirely on your machine
- ✅ **Privacy**: No data sent to external services
- ✅ **Efficiency**: Fast inference, low memory
- ✅ **Cost**: No API fees
- ✅ **Accessibility**: Works on CPU or GPU

**System Requirements:**
- **GPU Mode**: 8GB+ VRAM (recommended)
- **CPU Mode**: 16GB+ RAM (works but slower)
- **Storage**: 20GB+ free space
- **Python**: 3.8+

**How to Use:**

**1. Quick Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Verify SLM usage
python scripts/verify_slm_usage.py

# Run web demo
streamlit run demo/app.py
```

**2. Command Line:**
```bash
# Process single persona
python main.py --persona-id 1 --run-id 1

# Process all personas
python main.py --all-personas --run-id 1
```

**3. Programmatic Usage:**
```python
from src.persona_loader import PersonaLoader
from src.depression_detector import DepressionDetector

# Load persona (SLM)
loader = PersonaLoader(hf_token="token")
model, tokenizer = loader.load_persona(1)

# Detect depression
detector = DepressionDetector(config)
prediction = detector.analyze_conversation(conversation)
print(f"BDI Score: {prediction.bdi_score}")
```

**Features:**
- ✅ 6 conversation strategies
- ✅ Real-time analysis
- ✅ Automatic early stopping
- ✅ Complete submission pipeline
- ✅ Validation and error handling

**Performance:**
- GPU: ~1-2 seconds per response
- CPU: ~8-12 seconds per response
- Early stopping: Reduces total time by 40-60%

---

## Visual Design Suggestions

### Color Scheme:
- **Primary**: Blue (#2563EB) - Trust, technology
- **Secondary**: Green (#10B981) - Health, detection
- **Accent**: Orange (#F59E0B) - Warning, attention
- **Background**: White/Light Gray

### Slide Layout:
- **Slide 1**: Title with gradient background, centered
- **Slide 2**: Two-column layout (models side-by-side)
- **Slide 3**: Architecture diagram (centered, large)
- **Slide 4**: Workflow diagram with examples
- **Slide 5**: Code snippets with syntax highlighting

### Icons to Use:
- 🧠 Brain (depression detection)
- 🤖 Robot (AI/ML)
- 📊 Chart (analysis)
- 🔍 Magnifying glass (detection)
- ⚡ Lightning (speed/efficiency)
- 🏠 House (local execution)

### Fonts:
- **Title**: Bold, Sans-serif (Arial, Calibri)
- **Body**: Clean Sans-serif (Arial, Calibri)
- **Code**: Monospace (Consolas, Courier New)

---

## Speaker Notes

### Slide 1: Introduction
- Start with the challenge importance
- Emphasize real-world applications
- Mention CLEF eRisk competition

### Slide 2: Models
- Explain why SLMs over LLMs
- Highlight efficiency and privacy
- Show model size comparison
- Emphasize local execution capability

### Slide 3: Architecture
- Walk through each component
- Explain data flow
- Show how components interact
- Highlight modularity

### Slide 4: Functionality
- Demonstrate with example conversation
- Show detection process step-by-step
- Explain BDI-II scoring
- Highlight early detection feature

### Slide 5: Usage
- Show practical examples
- Demonstrate ease of use
- Highlight performance metrics
- Emphasize local execution benefits

---

## Additional Slides (Optional)

### Slide 6: Results & Evaluation
- Example predictions
- Accuracy metrics
- Early detection performance
- Comparison with baselines

### Slide 7: Future Work
- Multi-turn context improvement
- Adaptive strategy selection
- Ensemble methods
- Fine-tuning on validation data

### Slide 8: Conclusion
- Key achievements
- SLM advantages demonstrated
- Real-world applicability
- Open for questions
