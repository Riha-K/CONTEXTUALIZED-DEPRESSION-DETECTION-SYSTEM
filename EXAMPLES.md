# Code Examples

## Basic Usage

### Example 1: Single Persona Interaction

```python
import os
import yaml
from dotenv import load_dotenv
from src.persona_loader import PersonaLoader
from src.conversation_manager import ConversationManager
from src.depression_detector import DepressionDetector

# Load config
load_dotenv()
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize
loader = PersonaLoader(config_path="config/config.yaml", hf_token=os.getenv('HF_TOKEN'))
detector = DepressionDetector(config)

# Load persona
model, tokenizer = loader.load_persona(1)
system_prompt = loader.get_system_prompt(1)

# Start conversation
conv_manager = ConversationManager(model, tokenizer, system_prompt, config)
response = conv_manager.send_message("How are you feeling today?")
print(response)

# Analyze
conversation_log = conv_manager.get_conversation_log(1)
prediction = detector.analyze_conversation(conversation_log['conversation'])
print(f"BDI Score: {prediction.bdi_score}")
print(f"Symptoms: {prediction.key_symptoms}")
```

### Example 2: Automated Conversation Strategy

```python
from src.utils import get_conversation_strategies

strategies = get_conversation_strategies()
strategy = strategies[0]  # General wellbeing

for question in strategy['questions']:
    response = conv_manager.send_message(question)
    print(f"Q: {question}")
    print(f"A: {response}\n")
    
    # Check for early stopping
    log = conv_manager.get_conversation_log(1)
    pred = detector.analyze_conversation(log['conversation'])
    
    if detector.should_stop_early(pred):
        print("Early stopping triggered!")
        break
```

### Example 3: Batch Processing

```python
from src.submission_generator import SubmissionGenerator

generator = SubmissionGenerator()
all_conversations = []
all_results = []

for persona_id in range(1, 21):
    # Load and interact with persona
    model, tokenizer = loader.load_persona(persona_id)
    # ... conversation logic ...
    
    # Collect results
    all_conversations.append(conversation_log)
    all_results.append(result)

# Generate submission
generator.generate_submission(
    all_conversations, 
    all_results, 
    run_id=1, 
    is_manual=False
)
```

### Example 4: Custom Detection Logic

```python
class CustomDetector(DepressionDetector):
    def analyze_conversation(self, conversation_history):
        # Call parent method
        prediction = super().analyze_conversation(conversation_history)
        
        # Add custom logic
        # ... your custom analysis ...
        
        return prediction
```

### Example 5: Manual Run with Custom Questions

```python
# For manual runs, you can interactively ask questions
questions = [
    "Tell me about your day",
    "What's been challenging lately?",
    "How do you feel about yourself?",
    # ... more questions ...
]

for question in questions:
    response = conv_manager.send_message(question)
    print(response)
    
    # Manual analysis
    log = conv_manager.get_conversation_log(1)
    pred = detector.analyze_conversation(log['conversation'])
    
    # Make decision based on prediction
    if pred.bdi_score > 20:
        print("Moderate to severe depression detected")
        break
```

## Advanced Usage

### Custom Conversation Strategies

```python
def create_custom_strategy():
    return {
        "name": "custom_focus",
        "questions": [
            "What's your daily routine like?",
            "How do you spend your free time?",
            # ... custom questions ...
        ]
    }
```

### Real-time Analysis

```python
# Analyze after each turn
for turn in range(max_turns):
    response = conv_manager.send_message(question)
    
    # Real-time analysis
    log = conv_manager.get_conversation_log(persona_id)
    pred = detector.analyze_conversation(log['conversation'])
    
    # Display progress
    print(f"Turn {turn+1}: BDI={pred.bdi_score}, Confidence={pred.confidence:.2f}")
    
    # Adaptive questioning based on responses
    if pred.bdi_score > 15:
        # Focus on specific symptoms
        question = "Can you tell me more about that?"
    else:
        # Continue general questions
        question = next_question()
```

### Export and Validation

```python
from src.submission_generator import SubmissionGenerator

generator = SubmissionGenerator()

# Generate files
interactions_path, results_path = generator.generate_submission(
    conversations, results, run_id=1, is_manual=False
)

# Validate
if generator.validate_submission(run_id=1):
    print("Files are valid and ready for submission!")
else:
    print("Validation failed - check file format")
```

## Integration Examples

### With Streamlit (Custom UI)

```python
import streamlit as st
from src.persona_loader import PersonaLoader

st.title("Custom Depression Detection")

persona_id = st.selectbox("Persona", range(1, 21))
loader = PersonaLoader(...)
model, tokenizer = loader.load_persona(persona_id)

# Your custom UI logic
```

### With Flask API

```python
from flask import Flask, request, jsonify
from src.persona_loader import PersonaLoader
from src.conversation_manager import ConversationManager

app = Flask(__name__)
loader = PersonaLoader(...)

@app.route('/chat', methods=['POST'])
def chat():
    persona_id = request.json['persona_id']
    message = request.json['message']
    
    model, tokenizer = loader.load_persona(persona_id)
    conv_manager = ConversationManager(...)
    response = conv_manager.send_message(message)
    
    return jsonify({'response': response})
```

### With Jupyter Notebook

```python
# In a Jupyter notebook
from src.persona_loader import PersonaLoader
from src.conversation_manager import ConversationManager
from IPython.display import display, Markdown

loader = PersonaLoader(...)
model, tokenizer = loader.load_persona(1)

conv_manager = ConversationManager(...)

# Interactive conversation
user_input = input("Your message: ")
response = conv_manager.send_message(user_input)
display(Markdown(f"**Persona:** {response}"))
```
