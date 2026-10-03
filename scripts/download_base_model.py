"""
Pre-download the base model so "Load Persona" is faster (only loads into RAM, no download).
Run once: python scripts/download_base_model.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv

load_dotenv()


def main():
    print("Pre-downloading base model (this makes 'Load Persona' faster later)...")
    
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if not config or "model" not in config:
        print("config is missing the model section")
        sys.exit(1)
    
    model_name = config["model"]["base_model"]
    hf_token = os.getenv("HF_TOKEN")
    
    print(f"Model: {model_name}")
    
    # Download tokenizer (small, fast)
    from transformers import AutoTokenizer
    print("Downloading tokenizer...")
    AutoTokenizer.from_pretrained(model_name, token=hf_token)
    print("Tokenizer cached.")
    
    # Download model (this is the big part – same as first "Load Persona")
    from transformers import AutoModelForCausalLM
    import torch
    
    device = config["model"].get("device", "cpu")
    print(f"Downloading model weights (device={device})...")
    
    if device == "cpu":
        AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    
    print("Done. Base model is cached. Next 'Load Persona' will only load into RAM (faster).")


if __name__ == "__main__":
    main()
