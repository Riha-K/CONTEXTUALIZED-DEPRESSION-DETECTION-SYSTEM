"""
Test script to verify setup and configuration.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
        
        import transformers
        print(f"[OK] Transformers {transformers.__version__}")
        
        import peft
        print(f"[OK] PEFT {peft.__version__}")
        
        import streamlit
        print(f"[OK] Streamlit {streamlit.__version__}")
        
        import yaml
        print("[OK] PyYAML")
        
        import huggingface_hub
        print(f"[OK] Hugging Face Hub {huggingface_hub.__version__}")
        
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        return False

def test_config():
    """Test configuration files."""
    print("\nTesting configuration...")
    
    config_path = Path("config/config.yaml")
    if config_path.exists():
        print("[OK] config/config.yaml exists")
    else:
        print("[FAIL] config/config.yaml not found")
        return False
    
    env_path = Path(".env")
    if env_path.exists():
        print("[OK] .env file exists")
        load_dotenv()
        token = os.getenv('HF_TOKEN')
        if token:
            print("[OK] HF_TOKEN found in .env")
        else:
            print("[FAIL] HF_TOKEN not found in .env")
            return False
    else:
        print("[FAIL] .env file not found")
        return False
    
    return True

def test_huggingface_auth():
    """Test Hugging Face authentication."""
    print("\nTesting Hugging Face authentication...")
    try:
        from huggingface_hub import login, whoami
        load_dotenv()
        token = os.getenv('HF_TOKEN')
        
        if token:
            login(token=token)
            user_info = whoami()
            print(f"[OK] Authenticated as: {user_info.get('name', 'Unknown')}")
            return True
        else:
            print("[FAIL] No HF_TOKEN found")
            return False
    except Exception as e:
        print(f"[FAIL] Authentication failed: {e}")
        return False

def test_model_access():
    """Test access to base model."""
    print("\nTesting model access...")
    try:
        import yaml
        from huggingface_hub import model_info
        load_dotenv()
        token = os.getenv('HF_TOKEN')

        config_path = Path("config/config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        model_name = (cfg.get("model") or {}).get(
            "base_model", "HuggingFaceTB/SmolLM2-360M-Instruct"
        )
        model_info(model_name, token=token)
        print(f"[OK] Can access base model: {model_name}")
        return True
    except Exception as e:
        print(f"[FAIL] Cannot access base model: {e}")
        print("  Note: You may need to request access to the model on Hugging Face")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("eRisk 2026 Task 1 - Setup Test")
    print("="*60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Hugging Face Auth", test_huggingface_auth()))
    results.append(("Model Access", test_model_access()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! Setup is complete.")
    else:
        print("\n[ERROR] Some tests failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
