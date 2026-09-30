# CLEF eRisk 2026 — Task 1: Conversational Depression Detection

Talk with an LLM persona and estimate depression without asking about it directly. The run predicts a BDI-II score (0–63), picks the main symptoms, and writes the eRisk submission files.

Symptom matching uses `all-MiniLM-L6-v2`. The checked-in config talks with **SmolLM2-360M-Instruct** on CPU (`config/config.yaml`). To use the official LoRA personas, set `use_open_slm: false` and point `base_model` at `meta-llama/Meta-Llama-3-8B-Instruct`. Personas come from the Hugging Face collection `irlab-udc/erisk2026`.

## Setup

```bash
pip install -r requirements.txt
```

If the model or a persona repo is gated, put a Hugging Face token in `.env`:

```text
HF_TOKEN=your_token
```

Check the environment, then open the demo:

```bash
python scripts/test_setup.py
streamlit run demo/app.py
```

## Run

```bash
python quick_start.py
python main.py --persona-id 1 --run-id 1
python main.py --all-personas --run-id 1
python main.py --persona-id 1 --run-id 1 --manual
```

`--manual` keeps a person in the loop for that run.

## Output

Each run writes two files under `submissions/`:

- `interactions_run<id>.json` — full conversation
- `results_run<id>.json` — BDI-II score and symptoms

## Task rules

- Use the official system prompt as given
- Do not ask about depression directly
- An earlier correct decision is scored better than a long chat
- At most 3 runs per persona, and at most one of those may be manual

## Layout

```text
config/config.yaml              # model, turn limits, BDI settings
src/persona_loader.py           # base model and LoRA personas
src/conversation_manager.py     # multi-turn chat
src/depression_detector.py      # similarity scoring and BDI-II
src/submission_generator.py     # submission JSON
demo/app.py                     # Streamlit demo
main.py                         # command-line run
```

Longer notes: [USAGE.md](USAGE.md), [ARCHITECTURE.md](ARCHITECTURE.md), [LOCAL_EXECUTION.md](LOCAL_EXECUTION.md).

## References

- [Task details](https://erisk.irlab.org/Task1LLMs.html)
- [Hugging Face collection](https://huggingface.co/collections/irlab-udc/erisk2026)
