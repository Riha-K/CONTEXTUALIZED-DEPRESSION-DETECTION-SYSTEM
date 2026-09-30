# Local Execution Guide

## SLM Usage Verification ✅

This project uses **Small Language Models (SLMs)** exclusively:

1. **Conversation Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
   - **Size**: 8B parameters (Small Language Model)
   - **Purpose**: Base model for persona conversations
   - **Fine-tuning**: LoRA adapters (lightweight)

2. **Detection Model**: `all-MiniLM-L6-v2` (SentenceTransformer)
   - **Size**: 22M parameters (Small Language Model)
   - **Purpose**: Semantic similarity for symptom detection
   - **Usage**: Analyzes conversation text for depression indicators

Both models are SLMs and can run locally!

## System Requirements

### Minimum Requirements (CPU Mode)
- **RAM**: 16GB+ (for 8B model in float32)
- **Storage**: 20GB+ free space
- **Python**: 3.8+
- **OS**: Windows/Linux/Mac

### Recommended (GPU Mode)
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8+ or 12.1+
- **RAM**: 16GB+
- **Storage**: 20GB+ free space

## Local Execution Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` file:
```env
HUGGINGFACE_TOKEN=your_token_here
```

### 3. Verify Setup

```bash
python scripts/test_setup.py
```

This will check:
- ✅ Python packages installed
- ✅ Configuration files present
- ✅ Hugging Face authentication
- ✅ Model access permissions

## Running Locally

### CPU Mode (No GPU)

The system automatically detects if GPU is available:
- If **no GPU**: Uses CPU with float32 precision
- If **GPU available**: Uses GPU with 8-bit quantization

**CPU Mode Performance**:
- Model loading: ~2-5 minutes
- Inference: ~5-15 seconds per response
- Memory usage: ~12-16GB RAM

**To force CPU mode**, edit `config/config.yaml`:
```yaml
model:
  device: "cpu"  # Force CPU mode
  load_in_8bit: false  # Not available on CPU
```

### GPU Mode (Recommended)

**GPU Mode Performance**:
- Model loading: ~30-60 seconds
- Inference: ~1-3 seconds per response
- Memory usage: ~6-8GB VRAM (with 8-bit quantization)

**Automatic GPU detection**:
- System auto-detects CUDA availability
- Uses GPU if available, falls back to CPU

## Running the Demo Locally

### Web Demo (Streamlit)

```bash
streamlit run demo/app.py
```

Then open: `http://localhost:8501`

### Command Line

```bash
# Single persona
python main.py --persona-id 1 --run-id 1

# All personas
python main.py --all-personas --run-id 1
```

### Quick Start Example

```bash
python quick_start.py
```

## Troubleshooting Local Execution

### Issue: Out of Memory (OOM)

**Solution**:
1. Use CPU mode (slower but uses less memory)
2. Process one persona at a time
3. Clear cache between runs:
   ```python
   loader.clear_cache()
   torch.cuda.empty_cache()  # If using GPU
   ```

### Issue: Model Download Fails

**Solution**:
1. Ensure HF_TOKEN is set correctly
2. Request access to `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face
3. Check internet connection
4. Try downloading manually:
   ```bash
   python scripts/download_personas.py
   ```

### Issue: Slow Performance on CPU

**Solution**:
- This is expected - CPU inference is slower
- Consider using GPU if available
- Reduce `max_turns` in config for faster testing
- Use early stopping (already enabled)

### Issue: CUDA Not Found

**Solution**:
- System will automatically use CPU
- No action needed - it's handled automatically
- To verify: Check console output for "Loading on device: cpu"

## Model Sizes (Local Storage)

- **Base Model**: ~16GB (downloaded automatically)
- **LoRA Adapters**: ~50-100MB each (20 personas = ~1-2GB)
- **Sentence Transformer**: ~90MB (downloaded automatically)
- **Total**: ~18-20GB

Models are cached in Hugging Face cache directory:
- Windows: `C:\Users\<username>\.cache\huggingface\`
- Linux/Mac: `~/.cache/huggingface/`

## Performance Benchmarks

### CPU Mode (Intel i7, 16GB RAM)
- First load: ~3 minutes
- Subsequent loads: ~30 seconds (cached)
- Per message: ~8-12 seconds
- Memory: ~14GB peak

### GPU Mode (NVIDIA RTX 3060, 12GB VRAM)
- First load: ~45 seconds
- Subsequent loads: ~10 seconds (cached)
- Per message: ~1-2 seconds
- Memory: ~7GB VRAM peak

## Verification Checklist

Before running locally, verify:

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file configured with HF_TOKEN
- [ ] Hugging Face access granted for base model
- [ ] Sufficient disk space (~20GB)
- [ ] Sufficient RAM (16GB+ recommended)
- [ ] Test setup passes (`python scripts/test_setup.py`)

## Notes

- **All models run locally** - no external API calls
- **SLMs only** - no large language models
- **Automatic device detection** - works on CPU or GPU
- **8-bit quantization** - reduces GPU memory usage
- **Model caching** - faster subsequent runs

## Support

If you encounter issues:
1. Run `python scripts/test_setup.py` for diagnostics
2. Check console output for device detection messages
3. Verify model access on Hugging Face
4. Check system resources (RAM/VRAM)
