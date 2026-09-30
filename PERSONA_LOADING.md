# Where Is the Persona? How Does Loading Work?

## Faster option (current setup)

- **Model:** **SmolLM2-360M-Instruct** (360M parameters) – loads in **~30–60 seconds** on CPU (first time). Much faster than 0.5B/1.5B.
- **Pre-download (optional):** Run once so the model is already on disk when you open the app:
  ```bash
  python scripts/download_base_model.py
  ```
  Then when you click "Load Persona", only the "load into RAM" step runs (no download), so it feels faster.

---

## Why does loading take so long?

**Main reasons:**

1. **Download (first time only)**  
   The model (e.g. Qwen2-0.5B, ~1 GB) is downloaded from Hugging Face. This happens once; after that it’s cached (e.g. under `~/.cache/huggingface/`). Speed depends on your internet.

2. **Loading into RAM (every time you start the app)**  
   Even a 0.5B parameter model in float32 is ~2 GB in memory. The app has to:
   - Read files from disk
   - Build tensors and move them into RAM  
   On **CPU** this is much slower than on GPU (no fast VRAM). Expect **1–3 minutes** for the first load on a typical machine.

3. **Running on CPU**  
   With `device: "cpu"` in config, everything runs on the CPU. No GPU means slower load and slower replies. Using a GPU (if available) cuts load time a lot.

4. **One-time library startup**  
   The first time `torch` / `transformers` are used in the process, there is extra one-time overhead. Later “Load Persona” in the same session reuses the model already in memory, so it’s fast.

**Rough times (CPU):**

| Phase              | First time | Same session later |
|--------------------|------------|---------------------|
| Download model     | 1–5 min    | 0 (cached)          |
| Load into RAM     | 1–2 min    | 0 (already loaded)  |
| **Total**          | **2–7 min**| **~0**              |

After you restart the Streamlit app, the “first load” (download + load into RAM) happens again. So the **first** “Load Persona” after each app start is the slow one.

---

## Where the persona comes from

| Setting | Source | What gets loaded |
|--------|--------|-------------------|
| **Open SLM mode** (current) | `config/config.yaml` → `base_model: "SmolLM2-360M-Instruct"` | One **base model** from **Hugging Face**. No per-persona adapters. Persona 1–20 share this model; each ID uses a different **system prompt** for variety. |
| **Official task mode** (`use_open_slm: false`) | Base: `meta-llama/Meta-Llama-3-8B-Instruct`<br>Personas: `irlab-udc/erisk2026/persona1` … `persona20` | **Base model** from Hugging Face + **LoRA adapter** for the chosen persona (e.g. `persona3`) from the eRisk collection. Requires gated LLaMA access. |

So:
- **Persona** = the model (and, in open SLM mode, the system prompt) used to reply in the chat.
- **Where it loads from:** Hugging Face (base model; in official mode, also the persona LoRA from the eRisk collection).

---

## How loading works (step by step)

When you click **Load Persona** in the Streamlit app:

1. **Demo UI** (`demo/app.py`)
   - Calls `loader.load_persona(persona_id)` and `loader.get_system_prompt(persona_id)`.

2. **PersonaLoader** (`src/persona_loader.py`)
   - **If base not yet loaded:** `_load_base_model()` runs:
     - Reads `base_model` from `config/config.yaml`.
     - Downloads model (and tokenizer) from Hugging Face if not already cached.
     - Loads weights into RAM (CPU or GPU). **This is the slow part**, especially the first time.
   - **Open SLM mode:** Returns the same base model + tokenizer for every persona ID; no extra adapter.
   - **Official task mode:** Loads the LoRA adapter for that persona from `irlab-udc/erisk2026/persona{N}` and attaches it to the base model.
   - **System prompt:** In open SLM mode it comes from a short list in code (one prompt per persona ID). In official mode it can be read from the adapter repo (e.g. README).

3. **Back in the app**
   - Creates a `ConversationManager` with that model, tokenizer, and system prompt.
   - Stores them in `st.session_state` so the chat uses this “persona” for the rest of the session.

So:
- **Where is the persona?** In the **model** (and system prompt) that the app loads from **Hugging Face** (and, in official mode, from the eRisk persona LoRA repos).
- **How is it loading?** By calling Hugging Face to download/cache the base model (and optionally the persona adapter), then loading that into memory and wiring it into the `ConversationManager` for the chat.

---

## Why the first load is slow

- **Download:** First time, the base model (e.g. Qwen2-0.5B) is downloaded from Hugging Face (~hundreds of MB to ~1 GB). Cached under something like `~/.cache/huggingface/`.
- **Load into RAM:** The model is then loaded into memory (CPU or GPU). On CPU this can take 1–2 minutes for a 0.5B model.
- **Later loads:** Same model is reused from cache and from in-memory cache in the app, so “Load Persona” is much faster (e.g. ~30 s or less) until you restart the app.

---

## Code flow (file references)

```
demo/app.py
  → loader.load_persona(persona_id)     ← you click "Load Persona"
  → loader.get_system_prompt(persona_id)

src/persona_loader.py
  → _load_base_model()                  ← downloads/loads from Hugging Face
  → if use_open_slm: return (base_model, tokenizer)
  → else: PeftModel.from_pretrained(..., adapter_path)  ← persona LoRA

config/config.yaml
  → base_model, use_open_slm, device    ← where model name and mode come from
```

So: **persona is the model (and prompt) loaded from Hugging Face; loading is “download (if needed) → load into memory → use for chat”.**
