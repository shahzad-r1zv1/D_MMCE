<div align="center">

# D-MMCE

### Dynamic Multi-Model Consensus Engine

**Aggregate weak LLM learners into a single Globally Optimal answer.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

<br>

<img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
<img src="https://img.shields.io/badge/Anthropic-Claude_3.5-D4A574?style=for-the-badge" alt="Anthropic">
<img src="https://img.shields.io/badge/Google-Gemini_1.5-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Google">
<img src="https://img.shields.io/badge/Ollama-Any_Local_LLM-000000?style=for-the-badge" alt="Ollama">

---

*One question. Four models. Four prompt framings. Peer review. Semantic clustering. Stability-verified synthesis.*

</div>

<br>

## The Problem

A single LLM can hallucinate, overfit to its training bias, or land on a **Local Optimum** — an answer that looks correct in isolation but fails under scrutiny. D-MMCE treats every individual model output as a "weak learner" and applies an ensemble pipeline inspired by boosting, adversarial validation, and immune-system cross-reactivity to converge on a **Global Optimum**.

## How It Works

```
 User Query
     │
     ▼
┌─────────────────────┐
│  Prompt Perturbator  │  → 4 semantic variants (original, step-by-step, adversarial, summary)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Model Pool        │  → asyncio.gather() fans out P × 4 concurrent API calls
│  GPT-4o │ Claude    │    across OpenAI, Anthropic, Gemini, and any local
│  Gemini │ Ollama *  │    Ollama models (Llama, Mistral, Phi, Codellama…)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Peer Reviewer     │  → Pairwise cross-examination ("You are a logic auditor…")
│   (Immune System)   │    Generates a Contradiction Matrix with penalty scores
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Semantic Clusterer   │  → sentence-transformers embeddings + HDBSCAN
│ (Consensus Finder)   │    Largest cluster = Consensus; outliers = Local Optima discarded
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    Meta-Judge        │  → Lowest-penalty model synthesises the final answer
│  (Stability Loop)    │    Re-runs until cosine similarity ≥ threshold → Global Optimum
└─────────────────────┘
         │
         ▼
   Final Verdict
   (Globally Optimal Answer)
```

### Local vs. Global Optimality

| Concept | In D-MMCE |
|---|---|
| **Local Optimum** | A model response that may be plausible alone but diverges from ensemble consensus — it falls outside the main semantic cluster or is flagged by peer reviewers. |
| **Global Optimum** | The synthesised answer that (1) belongs to the densest semantic cluster, (2) survives peer review, and (3) remains **stable** across successive synthesis rounds. |
| **Stability Loop** | If re-generating the synthesis yields a semantically different answer (cosine sim < threshold), the system treats it as a Local Optimum and retries. Convergence = Global Optimum. |

## Architecture

```
D_MMCE/
├── main.py                          # CLI entry point
├── server.py                        # FastAPI web server + SSE streaming
├── requirements.txt                 # All dependencies
├── .env.example                     # API key template
├── static/
│   └── index.html                   # Single-page web UI
└── d_mmce/
    ├── __init__.py                  # Public API: D_MMCE, FinalVerdict
    ├── schemas.py                   # Typed dataclasses for pipeline data
    ├── observer.py                  # Observer Pattern: EventBus + EventType
    ├── prompt_perturbator.py        # Diversity Injection (4 prompt variants)
    ├── peer_reviewer.py             # Cross-Examination / Contradiction Matrix
    ├── semantic_clusterer.py        # Embeddings + HDBSCAN consensus clustering
    ├── meta_judge.py                # Meta-Judge + Stability Loop
    ├── orchestrator.py              # D_MMCE class — full async pipeline
    └── providers/
        ├── __init__.py              # Auto-registers all providers
        ├── base.py                  # Strategy interface (ModelProvider ABC)
        ├── factory.py               # Factory Pattern + @register decorator
        ├── openai_provider.py       # GPT-4o wrapper
        ├── anthropic_provider.py    # Claude 3.5 Sonnet wrapper
        ├── gemini_provider.py       # Gemini 1.5 Pro wrapper
        └── ollama_provider.py       # Any local model via Ollama HTTP
```

### Design Patterns

| Pattern | Where | Why |
|---|---|---|
| **Strategy** | `ModelProvider` ABC | Each LLM backend is swappable; the orchestrator is provider-agnostic. |
| **Factory** | `ProviderFactory` + `@register` | Providers self-register on import — no central switch statement. |
| **Observer** | `EventBus` | Pipeline stages emit typed events without coupling to loggers, UIs, or dashboards. |

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/D_MMCE.git
cd D_MMCE
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   GOOGLE_API_KEY=AI...
#   OLLAMA_BASE_URL=http://localhost:11434
```

> **Tip:** You only need _at least one_ provider configured. The engine gracefully skips unavailable providers.

### 3a. Run via CLI

```bash
python main.py "What causes the Northern Lights?"
python main.py --providers openai,anthropic "Explain quantum entanglement"
python main.py --stability-threshold 0.9 --max-reruns 5 -v "Is P=NP?"
```

### 3b. Run via Web UI

```bash
python server.py
# Open http://localhost:8000
```

The web UI provides:
- ⚡ **Real-time pipeline visualization** — watch events stream from each stage
- ⚙ **Settings panel** — configure API keys, toggle providers, tune stability threshold & max re-runs
- 🦙 **Local LLM picker** — auto-discovers all models installed in Ollama and lets you select any combination
- 📊 **Stability gauge** — visual indicator of convergence quality
- 📋 **Audit trail** — full transparency into every pipeline decision

## Using Local LLMs (Ollama)

D-MMCE can use **any model** installed on your local [Ollama](https://ollama.com) instance — Llama 3.1, Mistral, Phi, CodeLlama, Gemma, Qwen, DeepSeek, or any other. Multiple local models can participate in the ensemble simultaneously, each registered as a distinct provider (`ollama:mistral`, `ollama:phi3`, etc.).

### Setup

```bash
# 1. Install Ollama (https://ollama.com/download)
# 2. Pull any models you want to use
ollama pull llama3.1
ollama pull mistral
ollama pull phi3
ollama pull codellama

# 3. Ensure the server is running
ollama serve
```

### CLI usage

```bash
# Add one local model alongside cloud providers
python main.py --ollama-model mistral "Explain transformers"

# Add multiple local models (each becomes a separate ensemble member)
python main.py --ollama-model mistral --ollama-model phi3 "Compare sorting algorithms"

# Use ONLY local models (no cloud APIs needed)
python main.py --providers ollama:mistral --ollama-model phi3 --ollama-model codellama "Write a merge sort"

# Use the ollama:model shorthand in --providers
python main.py --providers openai,ollama:mistral "What is dark matter?"
```

### Web UI usage

1. Open Settings (⚙ button)
2. Scroll to the **🦙 Local LLMs** section
3. The UI auto-discovers all models installed on your Ollama server
4. Click to select/deselect any model — selected models join the ensemble alongside cloud providers
5. Hit **Save Settings** and run your query

### Programmatic usage

```python
from d_mmce import D_MMCE
from d_mmce.providers.ollama_provider import OllamaProvider

# Create providers for specific local models
mistral = OllamaProvider(model="mistral")
phi3    = OllamaProvider(model="phi3")

engine = D_MMCE(providers=[mistral, phi3])
verdict = await engine.run("Explain gradient descent")
```

### List available models

```python
from d_mmce.providers.ollama_provider import OllamaProvider

models = await OllamaProvider.list_local_models()
for m in models:
    print(f"{m['short_name']:20s}  {m['parameter_size']:>5s}  {m['quantization']}")
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--providers` | all registered | Comma-separated list of providers (`ollama:model` syntax supported) |
| `--ollama-model` | — | Add a local Ollama model; can be repeated for multiple models |
| `--review-provider` | `openai` | Model used for peer review critiques |
| `--embedding-model` | `all-MiniLM-L6-v2` | sentence-transformers model for clustering |
| `--stability-threshold` | `0.85` | Cosine similarity required for convergence |
| `--max-reruns` | `3` | Maximum Stability Loop iterations |
| `-v, --verbose` | off | Enable DEBUG logging |

## Programmatic Usage

```python
import asyncio
from d_mmce import D_MMCE

async def main():
    engine = D_MMCE(
        stability_threshold=0.85,
        max_stability_reruns=3,
    )

    verdict = await engine.run("What is dark matter?")

    print(verdict.answer)
    print(f"Stability: {verdict.stability_score:.2%}")
    print(f"Re-runs:   {verdict.num_reruns}")
    for step in verdict.audit_trail:
        print(f"  • {step}")

asyncio.run(main())
```

### Subscribe to pipeline events

```python
from d_mmce.observer import EventType

def my_handler(event):
    print(f"[{event.event_type.name}] {event.message}")

engine = D_MMCE()
engine.event_bus.subscribe(EventType.PEER_CRITIQUE, my_handler)
engine.event_bus.subscribe(EventType.STABILITY_CHECK, my_handler)
```

## Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Async orchestration | `asyncio.gather()` for parallel inference |
| LLM SDKs | `openai`, `anthropic`, `google-generativeai`, `httpx` (Ollama) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Clustering | `hdbscan` with cosine-similarity fallback |
| Web server | FastAPI + SSE streaming |
| Frontend | Vanilla HTML/CSS/JS — zero build step |

## How the Pipeline Prevents Hallucinations

1. **Diversity Injection** — Multiple prompt framings prevent a single framing from biasing all models toward the same error.
2. **Heterogeneous Models** — Different training data, architectures, and alignment approaches mean correlated failures are unlikely.
3. **Peer Review** — The "logic auditor" prompt explicitly asks for _failure points and factual inaccuracies_. Models that hallucinate get penalised in the Contradiction Matrix.
4. **Semantic Clustering** — Outlier responses (hallucinations that no other model agrees with) are mathematically identified and discarded.
5. **Stability Loop** — Even after synthesis, if re-generation produces a different answer, the system recognises it hasn't converged and retries.

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built to prove that the wisdom of the crowd applies to machines too.</sub>
</div>

