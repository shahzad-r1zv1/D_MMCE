"""
D-MMCE Package
===============
Dynamic Multi-Model Consensus Engine — aggregates weak LLM learners
to find a Globally Optimal output.
"""

from d_mmce.orchestrator import D_MMCE
from d_mmce.schemas import FinalVerdict

__all__ = ["D_MMCE", "FinalVerdict"]

