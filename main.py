"""
D-MMCE — Dynamic Multi-Model Consensus Engine
===============================================
CLI entry point.  Run with::

    python main.py "What causes the Northern Lights?"
    python main.py --providers openai,anthropic "Explain quantum entanglement"

Environment variables required (set the ones for providers you want to use):
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GOOGLE_API_KEY
    OLLAMA_BASE_URL  (default: http://localhost:11434)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import textwrap

from dotenv import load_dotenv

load_dotenv()  # Load .env file if present

from d_mmce.orchestrator import D_MMCE
from d_mmce.providers import ProviderFactory
from d_mmce.providers.ollama_provider import OllamaProvider
from d_mmce.schemas import FinalVerdict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D-MMCE: Dynamic Multi-Model Consensus Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py "What is photosynthesis?"
              python main.py --providers openai,anthropic "Explain gravity"
              python main.py --ollama-model mistral "Summarise quantum computing"
              python main.py --ollama-model mistral --ollama-model phi3 "Compare ML frameworks"
              python main.py --stability-threshold 0.9 --max-reruns 5 "Is P=NP?"
        """),
    )
    parser.add_argument("query", help="The question or task for the engine.")
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        help=(
            "Comma-separated list of provider names to use (default: all registered). "
            "Use 'ollama:modelname' to target a specific local model."
        ),
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        action="append",
        default=None,
        dest="ollama_models",
        help=(
            "Add a local Ollama model to the pool.  Can be repeated "
            "(e.g. --ollama-model mistral --ollama-model phi3)."
        ),
    )
    parser.add_argument(
        "--review-provider",
        type=str,
        default="auto",
        help="Provider used for peer reviews (default: auto = best available).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model for clustering.",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.85,
        help="Cosine-similarity threshold for the Stability Loop (default: 0.85).",
    )
    parser.add_argument(
        "--max-reruns",
        type=int,
        default=3,
        help="Maximum synthesis re-runs in the Stability Loop (default: 3).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_verdict(verdict: FinalVerdict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  D-MMCE — FINAL VERDICT (Globally Optimal Answer)")
    print(sep)
    print(f"\n{verdict.answer}\n")
    print(sep)
    print(f"  Stability Score : {verdict.stability_score:.4f}")
    print(f"  Re-runs         : {verdict.num_reruns}")
    print(f"  Audit Trail     :")
    for line in verdict.audit_trail:
        print(f"    • {line}")
    print(sep)


async def _main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)

    # Build provider list
    if args.providers:
        provider_names = [n.strip() for n in args.providers.split(",")]
        providers = []
        for name in provider_names:
            try:
                providers.append(ProviderFactory.create(name))
            except KeyError as e:
                print(f"WARNING: {e}", file=sys.stderr)
    else:
        # Auto-discover all providers including local Ollama models
        providers = await ProviderFactory.create_all_async()

    # Add explicitly-selected local Ollama models
    if args.ollama_models:
        for model_tag in args.ollama_models:
            tag = model_tag.strip()
            if tag:
                providers.append(OllamaProvider(model=tag))
                print(f"Added local model: ollama:{tag}")

    if not providers:
        print(
            "ERROR: No providers available. Set API keys and/or start Ollama.",
            file=sys.stderr,
        )
        sys.exit(1)

    engine = D_MMCE(
        providers=providers,
        review_provider_name=args.review_provider,
        embedding_model=args.embedding_model,
        stability_threshold=args.stability_threshold,
        max_stability_reruns=args.max_reruns,
    )

    print(f"Running D-MMCE with providers: {[p.name for p in providers]}")
    print(f"Query: {args.query}\n")

    verdict = await engine.run(args.query)
    _print_verdict(verdict)


if __name__ == "__main__":
    asyncio.run(_main())
