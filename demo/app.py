"""
Streamlit web demo for eRisk 2026 Task 1.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import yaml
import json
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.persona_loader import PersonaLoader
from src.conversation_manager import ConversationManager
from src.depression_detector import DepressionDetector
from src.submission_generator import SubmissionGenerator
from src.utils import format_bdi_severity, validate_persona_id

# Page configuration
st.set_page_config(
    page_title="eRisk 2026 - Depression Detection Demo",
    page_icon="🧠",
    layout="wide"
)

# Load environment variables
load_dotenv()


@st.cache_resource
def load_config():
    """Load configuration."""
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource
def initialize_components():
    """Initialize components."""
    config = load_config()
    hf_token = os.getenv('HF_TOKEN')
    
    loader = PersonaLoader(config_path="config/config.yaml", hf_token=hf_token)
    detector = DepressionDetector(config)
    
    return loader, detector, config


def main():
    """Main demo application."""
    st.title("🧠 eRisk 2026 - Conversational Depression Detection")
    st.markdown("**Task 1: Detect depression through natural conversation with LLM personas**")
    cfg = load_config()
    model_name = cfg.get("model", {}).get("base_model", "SLM")
    st.caption("Running with **open SLM**: " + model_name + " + all-MiniLM-L6-v2 (no gated access) • Device: " + str(cfg.get("model", {}).get("device", "auto")))
    
    # Initialize components
    try:
        loader, detector, config = initialize_components()
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        st.info("Please ensure:")
        st.info("1. Configuration file exists at config/config.yaml")
        st.info("2. HF_TOKEN is set in .env file")
        st.info("3. All dependencies are installed")
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        persona_id = st.number_input(
            "Persona ID",
            min_value=1,
            max_value=20,
            value=1,
            help="Select persona ID (1-20)"
        )
        
        if st.button("Load Persona", type="primary"):
            if validate_persona_id(persona_id):
                st.caption("First load: ~30–60 sec (360M model). Run 'python scripts/download_base_model.py' once to pre-download. Do not refresh.")
                with st.spinner(f"Loading persona {persona_id}..."):
                    try:
                        model, tokenizer = loader.load_persona(persona_id)
                        system_prompt = loader.get_system_prompt(persona_id)
                        
                        st.session_state['persona_loaded'] = True
                        st.session_state['persona_id'] = persona_id
                        st.session_state['model'] = model
                        st.session_state['tokenizer'] = tokenizer
                        st.session_state['system_prompt'] = system_prompt
                        st.session_state['conv_manager'] = ConversationManager(
                            model, tokenizer, system_prompt, config
                        )
                        st.success(f"Persona {persona_id} loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading persona: {e}")
            else:
                st.error("Invalid persona ID")
        
        st.divider()
        if cfg.get("model", {}).get("use_open_slm"):
            st.success("Using **open-access SLM** (SmolLM2-360M). No gated access. Fast load on CPU.")
        
        with st.expander("Where is the persona? How does loading work?"):
            st.markdown("""
            **Where the persona comes from**
            - **Base model:** Loaded from **Hugging Face** (`config/config.yaml` → `base_model`, e.g. `SmolLM2-360M-Instruct`).
            - **Open SLM mode (current):** One shared model for all Persona IDs. Each ID uses a slightly different **system prompt** for variety.
            - **Official task mode:** Base model = LLaMA 8B + **LoRA adapters** per persona from `irlab-udc/erisk2026` (persona1 … persona20).

            **What happens when you click Load Persona**
            1. **First time:** App downloads the base model from Hugging Face (if not cached), then loads it into RAM. This can take 1–2 min on CPU.
            2. **Later:** Model is already in memory (or in cache), so loading is faster (~30 sec).
            3. A **ConversationManager** is created with that model + tokenizer + system prompt, and the chat is ready.
            """)
        
        st.divider()
        
        st.subheader("BDI-II Severity Levels")
        st.info("""
        - **0-13**: Minimal
        - **14-19**: Mild
        - **20-28**: Moderate
        - **29-63**: Severe
        """)
        
        st.divider()
        
        st.subheader("Task Constraints")
        st.warning("""
        ⚠️ **Important Rules:**
        - Cannot directly ask about depression
        - Must infer from language, tone, and thoughts
        - Early detection is rewarded
        - Use official system prompt verbatim
        """)
    
    # Main content area
    if 'persona_loaded' not in st.session_state or not st.session_state['persona_loaded']:
        st.info("👈 Please load a persona from the sidebar to begin")
        
        # Show available personas
        st.subheader("Available Personas")
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        
        for i in range(1, 21):
            with cols[(i - 1) % 4]:
                st.button(f"Persona {i}", key=f"persona_{i}", disabled=True)
        
        return
    
    # Conversation interface
    st.header(f"Conversation with Persona {st.session_state['persona_id']}")
    
    # Initialize conversation history in session state
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    
    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state['messages']:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    
    # User input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state['messages'].append({"role": "user", "content": user_input})
        
        # Get response from persona
        conv_manager = st.session_state['conv_manager']
        with st.spinner("Persona is thinking..."):
            try:
                response = conv_manager.send_message(user_input)
                st.session_state['messages'].append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error getting response: {e}")
                st.session_state['messages'].pop()  # Remove failed user message
        
        st.rerun()
    
    # Analysis panel
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Real-time Analysis")
        if st.button("Analyze Conversation", type="primary"):
            conv_manager = st.session_state['conv_manager']
            conversation_log = conv_manager.get_conversation_log(st.session_state['persona_id'])
            
            if conversation_log['conversation']:
                prediction = detector.analyze_conversation(conversation_log['conversation'])
                
                st.metric("BDI-II Score", prediction.bdi_score)
                st.metric("Severity", format_bdi_severity(prediction.bdi_score))
                st.metric("Confidence", f"{prediction.confidence:.2%}")
                st.metric("Turn Count", prediction.turn_count)
                
                if prediction.key_symptoms:
                    st.subheader("Detected Symptoms")
                    for symptom in prediction.key_symptoms:
                        st.write(f"• {symptom}")
                else:
                    st.info("No symptoms detected yet")
                
                # Store prediction in session state
                st.session_state['current_prediction'] = prediction
            else:
                st.warning("No conversation yet. Start chatting to analyze!")
    
    with col2:
        st.subheader("Export & Submission")
        
        if st.button("Generate Submission Files"):
            if 'current_prediction' in st.session_state:
                conv_manager = st.session_state['conv_manager']
                persona_id = st.session_state['persona_id']
                
                conversation_log = conv_manager.get_conversation_log(persona_id)
                prediction = st.session_state['current_prediction']
                
                result = {
                    "LLM": str(persona_id),
                    "bdi-score": prediction.bdi_score,
                    "key-symptoms": prediction.key_symptoms
                }
                
                generator = SubmissionGenerator()
                interactions_path, results_path = generator.generate_submission(
                    [conversation_log], [result], run_id=1, is_manual=False
                )
                
                st.success("Submission files generated!")
                
                # Download buttons
                with open(interactions_path, 'r') as f:
                    st.download_button(
                        "Download Interactions JSON",
                        f.read(),
                        file_name=f"interactions_run1.json",
                        mime="application/json"
                    )
                
                with open(results_path, 'r') as f:
                    st.download_button(
                        "Download Results JSON",
                        f.read(),
                        file_name=f"results_run1.json",
                        mime="application/json"
                    )
            else:
                st.warning("Please analyze conversation first")
        
        if st.button("Reset Conversation"):
            st.session_state['messages'] = []
            st.session_state['conv_manager'].reset()
            if 'current_prediction' in st.session_state:
                del st.session_state['current_prediction']
            st.rerun()
    
    # Show conversation statistics
    if st.session_state['messages']:
        st.divider()
        st.subheader("Conversation Statistics")
        col1, col2, col3 = st.columns(3)
        
        user_messages = [m for m in st.session_state['messages'] if m['role'] == 'user']
        assistant_messages = [m for m in st.session_state['messages'] if m['role'] == 'assistant']
        
        with col1:
            st.metric("User Messages", len(user_messages))
        with col2:
            st.metric("Persona Responses", len(assistant_messages))
        with col3:
            total_chars = sum(len(m['content']) for m in st.session_state['messages'])
            st.metric("Total Characters", total_chars)


if __name__ == "__main__":
    main()
