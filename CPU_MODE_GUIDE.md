# CPU Mode Execution Guide

## ✅ Configuration Complete

Your project is now configured for **CPU mode** execution. The configuration has been updated:

- **Device**: CPU (forced)
- **8-bit Quantization**: Disabled (not available on CPU)
- **Model Loading**: Will use float32 precision

## 🚀 Streamlit Demo Running

The Streamlit demo should now be starting. Once it's ready:

1. **Open your browser** and go to:
   ```
   http://localhost:8501
   ```

2. **If the page doesn't open automatically**, check the terminal output for the URL.

## 📝 Important Notes for CPU Mode

### Performance Expectations:
- **Model Loading**: ~2-5 minutes (first time)
- **Subsequent Loads**: ~30-60 seconds (cached)
- **Per Message**: ~8-15 seconds per response
- **Memory Usage**: ~12-16GB RAM

### First Run:
- The first time you load a persona, it will download the model (~16GB)
- This only happens once - models are cached
- Ensure you have:
  - ✅ Stable internet connection
  - ✅ ~20GB free disk space
  - ✅ 16GB+ RAM available

### Using the Demo:

1. **Load a Persona**:
   - Select persona ID (1-20) from sidebar
   - Click "Load Persona"
   - Wait for model to load (first time takes longer)

2. **Start Chatting**:
   - Type messages in the chat input
   - Persona will respond (may take 8-15 seconds on CPU)
   - Continue conversation naturally

3. **Analyze Conversation**:
   - Click "Analyze Conversation" button
   - View BDI-II score and detected symptoms
   - See confidence level

4. **Export Results**:
   - Click "Generate Submission Files"
   - Download JSON files for submission

## ⚠️ Troubleshooting

### If Streamlit doesn't start:
```bash
# Check if port 8501 is available
# Or specify a different port:
streamlit run demo/app.py --server.port 8502
```

### If model loading fails:
- Check your `.env` file has `HF_TOKEN` set
- Ensure you have access to `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face
- Verify internet connection

### If out of memory:
- Close other applications
- Process one persona at a time
- Restart the demo if needed

### Slow performance:
- This is normal for CPU mode
- First load is always slower
- Subsequent interactions are faster
- Consider using GPU if available for better performance

## 🔄 Switching Back to GPU Mode

If you want to use GPU mode later, edit `config/config.yaml`:

```yaml
model:
  device: "auto"  # or "cuda"
  load_in_8bit: true
```

## 📊 Expected Behavior

**CPU Mode Characteristics:**
- ✅ Works without GPU
- ✅ Uses more RAM (~14GB)
- ✅ Slower inference (~8-15 sec/response)
- ✅ More stable (no GPU memory issues)
- ✅ Better for testing and development

**When to Use CPU:**
- No GPU available
- Testing and development
- Lower memory requirements
- More stable execution

## 🎯 Quick Commands

**Start Streamlit:**
```bash
streamlit run demo/app.py
```

**Check Status:**
```bash
# In another terminal
python scripts/test_setup.py
```

**Stop Streamlit:**
- Press `Ctrl+C` in the terminal
- Or close the terminal window

## 💡 Tips

1. **Be Patient**: First model load takes time
2. **One at a Time**: Load one persona at a time
3. **Monitor RAM**: Keep an eye on memory usage
4. **Save Work**: Export results frequently
5. **Test First**: Try persona 1 before processing all

---

**Your Streamlit demo should be running now!** 🎉

Open http://localhost:8501 in your browser to start using it.
