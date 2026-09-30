"""
Script to download and cache all persona adapters.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from tqdm import tqdm

def download_persona(persona_id: int, hf_token: str, collection: str = "irlab-udc/erisk2026"):
    """Download a single persona adapter."""
    persona_name = f"persona{persona_id}"
    repo_id = f"{collection}/{persona_name}"
    
    print(f"Downloading {persona_name}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            token=hf_token,
            local_dir=f"./personas/{persona_name}",
            local_dir_use_symlinks=False
        )
        print(f"✓ {persona_name} downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download {persona_name}: {e}")
        return False

def main():
    """Download all available personas."""
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        print("Error: HF_TOKEN not found in .env file")
        return
    
    login(token=hf_token)
    
    # Create personas directory
    Path("./personas").mkdir(exist_ok=True)
    
    print("Downloading persona adapters...")
    print("="*60)
    
    # Download personas 1-20
    success_count = 0
    for persona_id in tqdm(range(1, 21), desc="Downloading"):
        if download_persona(persona_id, hf_token):
            success_count += 1
    
    print("="*60)
    print(f"Downloaded {success_count}/20 personas")
    
    if success_count < 20:
        print("\nNote: Some personas may not be available yet.")
        print("Personas are released weekly, check the Hugging Face collection for availability.")

if __name__ == "__main__":
    main()
