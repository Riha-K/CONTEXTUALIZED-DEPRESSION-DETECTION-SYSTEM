# eRisk 2026 Task 1: Conversational Depression Detection
## Presentation Slides

---

## SLIDE 1: Project Overview

### 🧠 eRisk 2026 - Task 1: Conversational Depression Detection

**What is this project?**

- **Research Challenge**: CLEF eRisk 2026 Task 1
- **Objective**: Detect depression through natural conversations with AI personas
- **Approach**: Interact with fine-tuned LLM personas and analyze responses
- **Goal**: Predict BDI-II scores and identify depressive symptoms

**Key Features:**
- ✅ No direct questions about mental health
- ✅ Early detection optimization
- ✅ Automated and manual run support
- ✅ Real-time analysis and prediction
- ✅ Complete submission file generation

**Challenge Details:**
- 20 LLM personas released weekly
- Each persona simulates different depression severity levels
- Must detect depression through natural conversation flow
- Evaluation based on accuracy and early detection

---

## SLIDE 2: Models Used (SLM-Based)

### 🤖 Small Language Models (SLMs) Architecture

**1. Conversation Model: Meta-Llama-3-8B-Instruct**
- **Type**: Small Language Model (SLM)
- **Parameters**: 8 Billion
- **Purpose**: Base model for persona conversations
- **Fine-tuning**: LoRA adapters (Parameter-Efficient Fine-Tuning)
- **Size**: ~16GB (with 8-bit quantization: ~8GB VRAM)
- **Why SLM?**: Efficient, runs locally, fast inference

**2. Detection Model: all-MiniLM-L6-v2**
- **Type**: Sentence Transformer (SLM)
- **Parameters**: 22 Million
- **Purpose**: Semantic similarity analysis for symptom detection
- **Size**: ~90MB
- **Why SLM?**: Lightweight, fast, accurate semantic matching

**Key Advantages:**
- ✅ All models are SLMs (< 10B parameters)
- ✅ Runs entirely locally (CPU or GPU)
- ✅ No external API dependencies
- ✅ Privacy-preserving (all processing local)
- ✅ Cost-effective (no API costs)

---

## SLIDE 3: System Architecture

### 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                       │
│  (Streamlit Web Demo / CLI / Programmatic API)          │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  PersonaLoader        │
         │  - Load LoRA Adapters │
         │  - Manage Personas    │
         │  - System Prompts     │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ ConversationManager    │
         │  - Format Messages    │
         │  - Generate Responses  │
         │  - Track History      │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  DepressionDetector   │
         │  - Symptom Detection  │
         │  - BDI-II Scoring     │
         │  - Early Stopping     │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │ SubmissionGenerator   │
         │  - JSON Formatting    │
         │  - Validation         │
         └───────────────────────┘
```

**Component Details:**

1. **PersonaLoader**: Loads and manages LoRA adapters
   - Base model: Meta-Llama-3-8B-Instruct
   - LoRA adapters: 20 personas (released weekly)
   - Auto device detection (GPU/CPU)

2. **ConversationManager**: Handles interactions
   - Formats conversations with system prompts
   - Manages conversation history
   - Generates persona responses

3. **DepressionDetector**: Analyzes conversations
   - Keyword pattern matching (21 BDI-II symptoms)
   - Semantic similarity analysis (SentenceTransformer)
   - BDI-II score calculation (0-63)
   - Early stopping logic

4. **SubmissionGenerator**: Creates output files
   - Interactions JSON (conversation logs)
   - Results JSON (predictions)

---

## SLIDE 4: What It Does

### 🔍 System Functionality & Workflow

**1. Persona Interaction**
- Load persona adapter from Hugging Face
- Initialize conversation with official system prompt
- Engage in natural conversation (no direct mental health questions)
- Track all interactions for submission

**2. Depression Detection**
- **Symptom Detection**: Identifies 21 BDI-II symptoms
  - Keyword pattern matching
  - Semantic similarity analysis
  - Context understanding
- **BDI-II Scoring**: Calculates depression severity (0-63)
  - Minimal (0-13)
  - Mild (14-19)
  - Moderate (20-28)
  - Severe (29-63)
- **Key Symptoms**: Identifies up to 4 most prominent symptoms

**3. Early Detection Optimization**
- Monitors confidence after each turn
- Stops early when confidence threshold reached (0.85)
- Balances accuracy vs. efficiency
- Rewards early accurate detection

**4. Submission Generation**
- **Interactions File**: Complete conversation logs
- **Results File**: BDI-II scores and key symptoms
- Validates format before submission
- Supports up to 3 runs per persona

**Output Example:**
```json
{
  "LLM": "1",
  "bdi-score": 18,
  "key-symptoms": ["Sadness", "Loss of Interest", "Fatigue"]
}
```

---

## SLIDE 5: SLM Usage & How to Use

### 🚀 Implementation & Usage

**Why SLMs?**
- ✅ **Efficiency**: Fast inference, low memory footprint
- ✅ **Local Execution**: Runs entirely on your machine
- ✅ **Privacy**: No data sent to external APIs
- ✅ **Cost-Effective**: No API costs
- ✅ **Accessibility**: Works on CPU or GPU

**Local Execution:**
- **GPU Mode**: Automatic detection, 8-bit quantization (~7GB VRAM)
- **CPU Mode**: Automatic fallback, float32 precision (~14GB RAM)
- **Auto-Detection**: No configuration needed

**How to Use:**

**1. Setup:**
```bash
pip install -r requirements.txt
python scripts/verify_slm_usage.py  # Verify SLM models
python scripts/test_setup.py        # Test configuration
```

**2. Web Demo (Interactive):**
```bash
streamlit run demo/app.py
```
- Load personas
- Chat interactively
- Real-time analysis
- Export submission files

**3. Command Line (Batch Processing):**
```bash
# Single persona
python main.py --persona-id 1 --run-id 1

# All personas
python main.py --all-personas --run-id 1
```

**4. Programmatic Usage:**
```python
from src.persona_loader import PersonaLoader
from src.conversation_manager import ConversationManager
from src.depression_detector import DepressionDetector

# Load persona
loader = PersonaLoader(hf_token="your_token")
model, tokenizer = loader.load_persona(1)

# Interact
conv_manager = ConversationManager(...)
response = conv_manager.send_message("How are you feeling?")

# Analyze
detector = DepressionDetector(config)
prediction = detector.analyze_conversation(conversation)
```

**Key Features:**
- ✅ Multiple conversation strategies (6 different approaches)
- ✅ Real-time analysis and confidence scoring
- ✅ Automatic early stopping
- ✅ Complete submission file generation
- ✅ Validation and error handling

---

## Summary Slide (Bonus)

### 📊 Project Highlights

**Technology Stack:**
- **Base Model**: Meta-Llama-3-8B-Instruct (SLM)
- **Detection**: all-MiniLM-L6-v2 (SLM)
- **Framework**: PyTorch, Transformers, PEFT
- **Interface**: Streamlit, CLI, Python API

**Key Achievements:**
- ✅ 100% SLM-based (no large models)
- ✅ Fully local execution (CPU/GPU)
- ✅ Early detection optimization
- ✅ Multiple conversation strategies
- ✅ Complete submission pipeline

**Project Structure:**
- Modular architecture
- Comprehensive documentation
- Easy to extend and customize
- Production-ready code

**Next Steps:**
1. Download persona adapters
2. Run test conversations
3. Generate predictions
4. Submit results

---

## Visual Elements for Slides

**Slide 1**: 
- CLEF eRisk logo
- Project title with brain icon
- Key statistics (20 personas, BDI-II scoring)

**Slide 2**:
- Model comparison chart (8B vs 22M parameters)
- SLM vs LLM comparison
- Architecture diagram showing model flow

**Slide 3**:
- System architecture diagram (as shown above)
- Component interaction flow
- Data flow visualization

**Slide 4**:
- Workflow diagram
- Example conversation snippet
- BDI-II severity chart
- Detection process visualization

**Slide 5**:
- Code snippets (syntax highlighted)
- Usage flowchart
- Performance metrics
- System requirements

---

## Notes for Presenter

**Slide 1**: Emphasize the challenge and research importance
**Slide 2**: Highlight why SLMs are chosen (efficiency, privacy, cost)
**Slide 3**: Walk through architecture components and their roles
**Slide 4**: Demonstrate with example conversation and detection
**Slide 5**: Show live demo or code examples

**Key Talking Points:**
- All models are SLMs (< 10B parameters)
- Runs entirely locally (no cloud dependencies)
- Privacy-preserving (all data stays local)
- Early detection optimization
- Multiple conversation strategies
- Production-ready implementation
