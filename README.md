<div align="center">

# 🧠 D-MMCE

### Dynamic Multi-Model Consensus Engine

**Aggregate "weak" LLM learners into a single Globally Optimal answer through ensemble intelligence.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![asyncio](https://img.shields.io/badge/asyncio-parallel-yellow?style=for-the-badge)](https://docs.python.org/3/library/asyncio.html)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)

<br>

<img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
<img src="https://img.shields.io/badge/Anthropic-Claude_3.5-D4A574?style=for-the-badge" alt="Anthropic">
<img src="https://img.shields.io/badge/Google-Gemini_1.5-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Google">
<img src="https://img.shields.io/badge/Ollama-Local_LLMs-000000?style=for-the-badge" alt="Ollama">

---

*One question · Multiple models · Multiple prompt framings · Peer review · Semantic clustering · Stability-verified synthesis*

**→ Works with cloud APIs, local LLMs via Ollama, or any combination.**

</div>

<br>

## 🎯 The Problem

A single LLM can hallucinate, overfit to its training bias, or land on a **Local Optimum** — an answer that looks correct in isolation but fails under scrutiny.

D-MMCE treats every individual model output as a "weak learner" and applies an ensemble pipeline inspired by boosting, adversarial validation, and immune-system cross-reactivity to converge on a **Global Optimum**.

> **No API keys?** No problem. D-MMCE works entirely with local Ollama models — just `ollama pull` any models and go.

---

## ⚡ Key Features

| Feature | Description |
|---|---|
| 🔀 **Diversity Injection** | Automatically generates 4 semantic prompt variants (original, step-by-step, adversarial, summary) to reduce framing bias |
| 🚀 **Parallel Inference** | `asyncio.as_completed()` fans out all model × variant calls concurrently with live streaming as each completes |
| 📡 **Token Streaming** | True SSE streaming from Ollama — tokens appear live in the UI as they're generated, eliminating timeout issues |
| 🛡️ **Peer Review (Immune System)** | Cross-examination layer where models critique each other's outputs, generating a Contradiction Matrix |
| 🎯 **Semantic Clustering** | `sentence-transformers` embeddings + HDBSCAN identify the consensus cluster; outliers (Local Optima) are discarded |
| ⚖️ **Meta-Judge + Stability Loop** | The lowest-penalty model synthesises the final answer, then re-verifies for convergence |
| ⚡ **Concurrency Control** | Configurable semaphore limits parallel tasks + exponential-backoff retry for transient failures |
| 🦙 **Local LLM Support** | Auto-discovers all Ollama models — use Llama, Qwen, Mistral, Phi, DeepSeek, or any other local model |
| 🌐 **Live Web UI** | Real-time pipeline visualization with streaming response cards, stability gauge, and full audit trail |
| 📜 **Run History** | SQLite-backed history of every pipeline run — browse, compare, and review past results |
| 📝 **Debug Logging** | Every pipeline run is traced to `d_mmce.log` for easy debugging |

---

## 🔬 How It Works

```
 User Query
     │
     ▼
┌─────────────────────────┐
│   Prompt Perturbator    │  → 4 semantic variants (original, step-by-step,
│   (Diversity Injection) │    adversarial, summary)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│      Model Pool         │  → asyncio.as_completed() fans out P × 4
│   ┌───────┬───────┐     │    concurrent calls. Responses stream to the
│   │GPT-4o │Claude │     │    UI live as each model finishes.
│   │Gemini │Ollama*│     │    * Any local model: Llama, Qwen, Mistral…
│   └───────┴───────┘     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│     Peer Reviewer       │  → "You are a logic auditor…"
│    (Immune System)      │    Pairwise cross-examination generates a
│                         │    Contradiction Matrix with penalty scores
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Semantic Clusterer    │  → sentence-transformers embeddings + HDBSCAN
│   (Consensus Finder)    │    Densest cluster = Consensus
│                         │    Outliers = Local Optima → discarded
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│      Meta-Judge         │  → Lowest-penalty model synthesises the answer
│   (Stability Loop)      │    Re-runs until cosine_sim ≥ threshold
│                         │    Convergence → Global Optimum ✅
└───────────┬─────────────┘
            │
            ▼
     Final Verdict
   ✅ Globally Optimal Answer
```

### Local vs. Global Optimality

| Concept | In D-MMCE |
|---|---|
| **Local Optimum** | A response that seems plausible alone but diverges from ensemble consensus — falls outside the main semantic cluster or is flagged by peer reviewers |
| **Global Optimum** | The synthesised answer that (1) belongs to the densest semantic cluster, (2) survives peer review, and (3) remains **stable** across successive synthesis rounds |
| **Stability Loop** | If re-generating the synthesis yields a semantically different answer (cosine sim < threshold), the system treats it as a Local Optimum and retries. Convergence = Global Optimum |

---

## 🚀 Quick Start

### 1. Clone & Install

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

### 2. Configure (pick any option)

<details>
<summary><b>Option A: Cloud APIs</b> — set API keys in <code>.env</code></summary>

```bash
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   GOOGLE_API_KEY=AI...
```

</details>

<details>
<summary><b>Option B: Local LLMs only</b> — no API keys needed</summary>

```bash
# 1. Install Ollama → https://ollama.com/download
# 2. Pull any models
ollama pull llama3.2
ollama pull qwen3:4b
ollama pull mistral

# 3. Ensure Ollama is running
ollama serve
```

D-MMCE auto-discovers all installed Ollama models — no configuration needed.

</details>

<details>
<summary><b>Option C: Mixed</b> — cloud + local together</summary>

Set API keys in `.env` AND have Ollama running. D-MMCE will use everything available.

</details>

> **Tip:** You need *at least one* working provider. The engine auto-detects what's available and gracefully skips the rest.

### 3. Run

**Web UI (recommended):**

```bash
python server.py
# Open http://localhost:8000
```

**CLI:**

```bash
python main.py "What causes the Northern Lights?"
```

---

## 🌐 Web UI

The web interface provides a full real-time view of the D-MMCE pipeline:

| Feature | Description |
|---|---|
| 🎲 **Pipeline Stages** | Animated chip indicators for each stage: Diversify → Infer → Peer Review → Cluster → Synthesize → Verdict |
| 📡 **Live Event Feed** | Streaming log of every pipeline event as it happens |
| ✍️ **Token Streaming** | Live typing effect — tokens appear in response cards as they're generated by Ollama models |
| 🧠 **Model Response Cards** | Individual outputs from each model × prompt variant, rendered live as they arrive |
| 🏆 **Globally Optimal Answer** | The final synthesised result with stability score |
| 📊 **Stability Gauge** | Circular SVG gauge showing convergence quality (green ≥ 85%, yellow ≥ 60%, red < 60%) |
| 📋 **Audit Trail** | Full step-by-step breakdown of every pipeline decision |
| 📜 **Run History** | Browse, view details, and delete past runs — all stored in local SQLite |
| ⚙️ **Settings Panel** | Configure API keys, toggle cloud providers, select local Ollama models, tune stability threshold, max re-runs, concurrency, retries, and streaming |
| 🦙 **Local LLM Picker** | Auto-discovers all Ollama models with parameter size, quantization, and family info |

---

## 🦙 Using Local LLMs (Ollama)

D-MMCE works with **any model** installed on your local [Ollama](https://ollama.com) instance. Multiple local models participate simultaneously, each as a distinct ensemble member (`ollama:qwen3:4b`, `ollama:llama3.2:3b`, etc.).

### Setup

```bash
# Install Ollama → https://ollama.com/download
# Pull models
ollama pull llama3.2:3b
ollama pull qwen3:4b
ollama pull mistral
ollama pull phi3
ollama pull codellama

# Start the server
ollama serve
```

### CLI Usage

```bash
# Auto-discover local models (no flags needed — D-MMCE finds them)
python main.py "Explain transformers"

# Explicitly add specific local models
python main.py --ollama-model mistral --ollama-model phi3 "Compare sorting algorithms"

# Mix cloud + local
python main.py --providers openai --ollama-model mistral "What is dark matter?"
```

### Web UI Usage

1. Open the Settings panel (⚙️ button)
2. Scroll to **🦙 Local LLMs** — the UI auto-discovers all installed Ollama models
3. Click to select/deselect any model
4. Save & run your query — selected models join the ensemble

### Programmatic Usage

```python
import asyncio
from d_mmce.providers.ollama_provider import OllamaProvider

async def main():
    # List all available local models
    models = await OllamaProvider.list_local_models()
    for m in models:
        print(f"{m['name']:20s}  {m['parameter_size']:>5s}  {m['quantization']}")

asyncio.run(main())
```

---

## 📖 CLI Reference

| Flag | Default | Description |
|---|---|---|
| `query` | *(required)* | The question or task to process |
| `--providers` | auto-discover all | Comma-separated list (e.g. `openai,anthropic`, `ollama:mistral`) |
| `--ollama-model` | — | Add a local Ollama model; repeatable for multiple models |
| `--review-provider` | `auto` | Model for peer reviews (`auto` = best available provider) |
| `--embedding-model` | `all-MiniLM-L6-v2` | sentence-transformers model for semantic clustering |
| `--stability-threshold` | `0.85` | Cosine similarity required for Stability Loop convergence |
| `--max-reruns` | `3` | Maximum synthesis re-runs in the Stability Loop |
| `--max-concurrent` | `4` | Maximum parallel inference tasks (semaphore limit) |
| `--max-retries` | `2` | Retries per task with exponential backoff |
| `--no-streaming` | off | Disable token streaming (use batch mode) |
| `-v, --verbose` | off | Enable DEBUG logging |

---

## 🐍 Programmatic API

```python
import asyncio
from d_mmce import D_MMCE

async def main():
    engine = D_MMCE(
        stability_threshold=0.85,
        max_stability_reruns=3,
        review_provider_name="auto",    # picks best available
        max_concurrent_tasks=4,         # limit parallel calls
        max_retries=2,                  # retry with exponential backoff
        enable_streaming=True,          # token-by-token streaming
    )

    verdict = await engine.run("What is dark matter?")

    print(verdict.answer)
    print(f"Stability: {verdict.stability_score:.2%}")
    print(f"Re-runs:   {verdict.num_reruns}")
    for step in verdict.audit_trail:
        print(f"  • {step}")

asyncio.run(main())
```

### Subscribe to Pipeline Events

```python
from d_mmce.observer import EventType

def on_response(event):
    data = event.payload
    print(f"[{data['provider']}] {data['variant']}: {data['text'][:100]}...")

engine = D_MMCE()
engine.event_bus.subscribe(EventType.MODEL_RESPONSE, on_response)
engine.event_bus.subscribe(EventType.STABILITY_CHECK, lambda e: print(e.message))
```

### Available Event Types

| Event | Emitted when… |
|---|---|
| `PROMPT_PERTURBED` | Prompt variants are generated |
| `MODEL_RESPONSE` | A model returns a complete response (includes full text in payload) |
| `TOKEN_CHUNK` | A single token is streamed from a provider (includes `provider`, `variant`, `token` in payload) |
| `PEER_CRITIQUE` | A peer review critique is completed |
| `CLUSTER_FORMED` | Semantic clustering identifies the consensus cluster |
| `OUTLIER_DISCARDED` | A response is classified as an outlier |
| `SYNTHESIS_STARTED` | Meta-Judge begins synthesis |
| `STABILITY_CHECK` | A stability loop iteration completes |
| `STABILITY_RERUN` | Synthesis is re-run due to instability |
| `FINAL_VERDICT` | Final answer is produced (includes answer in payload) |

---

## 🏗️ Architecture

```
D_MMCE/
├── main.py                          # CLI entry point
├── server.py                        # FastAPI + SSE streaming server
├── requirements.txt                 # All dependencies
├── pytest.ini                       # Test configuration
├── .env.example                     # API key template
├── .gitignore                       # Standard Python gitignore
├── d_mmce.log                       # Runtime debug log (auto-created)
├── d_mmce_history.db                # Run history database (auto-created)
├── static/
│   └── index.html                   # Single-page web UI (zero build step)
├── tests/
│   ├── conftest.py                  # Shared fixtures & MockProvider
│   ├── test_unit.py                 # Unit tests (schemas, observer, perturbator, factory)
│   ├── test_integration.py          # Integration & E2E tests (full pipeline)
│   └── test_features.py             # Streaming, concurrency/retry, history tests
└── d_mmce/
    ├── __init__.py                  # Public API: D_MMCE, FinalVerdict
    ├── schemas.py                   # Typed dataclasses (ModelResponse, etc.)
    ├── observer.py                  # Observer pattern: EventBus + EventType
    ├── prompt_perturbator.py        # Diversity injection (4 prompt variants)
    ├── peer_reviewer.py             # Cross-examination / Contradiction Matrix
    ├── semantic_clusterer.py        # Embeddings + HDBSCAN consensus clustering
    ├── meta_judge.py                # Meta-Judge + Stability Loop
    ├── orchestrator.py              # D_MMCE class — full async pipeline
    ├── history.py                   # SQLite-backed run history (aiosqlite)
    └── providers/
        ├── __init__.py              # Auto-registers all providers
        ├── base.py                  # Strategy interface (ModelProvider ABC)
        ├── factory.py               # Factory + @register + auto-discovery
        ├── openai_provider.py       # OpenAI GPT-4o
        ├── anthropic_provider.py    # Anthropic Claude 3.5 Sonnet
        ├── gemini_provider.py       # Google Gemini 1.5 Pro
        └── ollama_provider.py       # Any local model via Ollama HTTP API
```

### Design Patterns

| Pattern | Where | Why |
|---|---|---|
| **Strategy** | `ModelProvider` ABC | Each LLM backend is a swappable strategy; the orchestrator is provider-agnostic |
| **Factory** | `ProviderFactory` + `@register` | Providers self-register on import — no central switch statement |
| **Observer** | `EventBus` + `EventType` | Pipeline stages emit typed events for UIs, loggers, and dashboards without coupling |
| **Async Orchestrator** | `D_MMCE.run()` | `asyncio.as_completed()` for parallel inference with live event streaming |

---

## 🛡️ How the Pipeline Prevents Hallucinations

| Layer | Mechanism |
|---|---|
| **1. Diversity Injection** | Multiple prompt framings prevent a single framing from biasing all models toward the same error |
| **2. Heterogeneous Models** | Different training data, architectures, and alignment approaches make correlated failures unlikely |
| **3. Peer Review** | The "logic auditor" prompt explicitly asks for *failure points and factual inaccuracies*. Hallucinating models get penalised in the Contradiction Matrix |
| **4. Semantic Clustering** | Outlier responses (hallucinations no other model agrees with) are mathematically identified and discarded |
| **5. Stability Loop** | Even after synthesis, if re-generation produces a different answer, the system recognises it hasn't converged and retries |

---

## 🛠️ Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Async orchestration | `asyncio.as_completed()` with semaphore-gated concurrency and exponential-backoff retry |
| Token streaming | Ollama NDJSON streaming via `httpx.AsyncClient.stream()` with per-token `TOKEN_CHUNK` events |
| LLM SDKs | `openai`, `anthropic`, `google-generativeai`, `httpx` (Ollama) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Clustering | `hdbscan` with cosine-similarity fallback |
| Web server | FastAPI + SSE (Server-Sent Events) streaming |
| Frontend | Vanilla HTML/CSS/JS — zero build step, dark-themed UI |
| Run history | `aiosqlite` → `d_mmce_history.db` (auto-created, env-configurable via `D_MMCE_HISTORY_DB`) |
| Logging | Python `logging` → console + `d_mmce.log` file |
| Testing | `pytest` + `pytest-asyncio` — 68 tests, ~7s, no network calls |

---

## 🧪 Testing

The test suite uses **mock providers** — no API keys or Ollama required. Tests run in ~7 seconds.

### Run all tests

```bash
pytest                    # run everything
pytest -v                 # verbose output
pytest tests/test_unit.py # unit tests only
pytest tests/test_integration.py  # integration & E2E only
pytest tests/test_features.py     # streaming, concurrency, history tests
```

### Test structure

| File | Tests | What it covers |
|---|---|---|
| `tests/conftest.py` | — | `MockProvider`, `MockStreamingProvider`, `FlakyProvider`, shared fixtures |
| `tests/test_unit.py` | 29 | Schemas, `EventBus` pub/sub, `PromptPerturbator`, `ModelProvider.generate()`, peer review parsing & penalty calculation, `ProviderFactory` registry |
| `tests/test_integration.py` | 20 | `PeerReviewer` full flow, `SemanticClusterer` consensus & outlier detection, `MetaJudge` synthesis & stability convergence, **Full E2E pipeline** |
| `tests/test_features.py` | 19 | **Token Streaming** (supports_streaming, generate_stream, TOKEN_CHUNK events), **Concurrency** (semaphore limits, retry with backoff, exhausted retries), **History DB** (save, list, get, delete, clear, pagination, ordering), **E2E history integration** |

### Key E2E tests

| Test | Validates |
|---|---|
| `test_full_pipeline` | Complete pipeline returns a `FinalVerdict` with a non-empty answer |
| `test_pipeline_emits_all_event_types` | All expected event types fire during a run |
| `test_pipeline_model_response_has_payload` | `MODEL_RESPONSE` events carry `text`, `provider`, `variant` |
| `test_pipeline_final_verdict_has_answer_payload` | `FINAL_VERDICT` event includes the full answer |
| `test_pipeline_deterministic_converges` | Stability Loop converges (score ≥ 0.85) with deterministic inputs |
| `test_pipeline_no_providers_returns_error` | Graceful error when no providers are available |

---

## 🔧 Troubleshooting

### Debug Log

Every pipeline run is logged to **`d_mmce.log`** in the project root. Check it for:
- Which providers were discovered and their availability status
- Which model the Meta-Judge selected
- Response lengths and latencies
- Errors, timeouts, and stack traces

```bash
# View the last 50 lines
tail -50 d_mmce.log                     # macOS/Linux
Get-Content d_mmce.log -Tail 50         # Windows PowerShell
```

### Common Issues

| Issue | Solution |
|---|---|
| `No providers available` | Set API keys in `.env` or start Ollama with at least one pulled model |
| `ReadTimeout` on large models | Reduce **Max Concurrent Tasks** in settings (default 4). Large models (20B+) benefit from concurrency of 1–2. The retry system will auto-recover from transient timeouts |
| `404 Not Found` from Ollama | The model tag doesn't match what's installed. Run `ollama list` and use the exact tag (e.g. `qwen3:4b` not `qwen3`) |
| Response cards not showing | Open browser DevTools (F12) → Console. Look for `[D-MMCE]` logs. Hard-refresh with `Ctrl+Shift+R` |
| Token streaming not working | Ensure **Enable Token Streaming** is checked in settings. Only Ollama providers support true streaming; cloud providers use batch mode |
| History not loading | The SQLite database is created automatically at `d_mmce_history.db`. Set `D_MMCE_HISTORY_DB` env var to change the location |
| Old code still running | Kill stale Python processes and restart: `Get-Process python* \| Stop-Process -Force` (Windows) or `pkill -f server.py` (Linux/macOS), then `python server.py` |

### Browser Console Debugging

The web UI logs all pipeline events to the browser console with the `[D-MMCE]` prefix:
```
[D-MMCE] MODEL_RESPONSE { provider: "ollama:qwen3:4b", hasText: true }
[D-MMCE] FINAL_VERDICT  { hasPayload: true, answer_length: 1275 }
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built to prove that the wisdom of the crowd applies to machines too.**

<sub>D-MMCE • Dynamic Multi-Model Consensus Engine</sub>

</div>
