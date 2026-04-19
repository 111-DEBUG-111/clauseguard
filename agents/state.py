"""
ClauseGuard — LangGraph Shared State Schema
============================================
This module defines the TypedDict that forms the backbone of the
LangGraph state machine.  Every agent node reads from and writes
to this structure so that data flows cleanly between steps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


# ---------------------------------------------------------------------------
# Per-clause result — what the pipeline produces for each clause
# ---------------------------------------------------------------------------

class ClauseResult(TypedDict):
    """Structured output for a single analysed clause."""
    clause: str                        # Original clause text
    risk_category: str                 # One of the 5 categories
    confidence: float                  # 0-1 float
    explanation: str                   # Plain-English reasoning
    key_risk_terms: List[str]          # Critical terms flagged by LLM
    severity: str                      # "High" | "Medium" | "Low"


# ---------------------------------------------------------------------------
# Top-level graph state
# ---------------------------------------------------------------------------

class ClauseGuardState(TypedDict):
    """
    Shared mutable state for the entire LangGraph pipeline.

    Fields
    ------
    raw_input : str
        The raw text supplied by the user — either a single clause
        or a full contract document.
    mode : str
        "single"  → analyse one clause
        "document" → extract multiple clauses, then analyse each
    clauses : List[str]
        Clause strings extracted by the Clause Extraction Agent.
        Populated only in "document" mode.
    current_clause : str
        The clause currently being classified.
    results : List[ClauseResult]
        Accumulated results — one entry per clause.
    document_summary : Optional[str]
        High-level summary generated after all clauses are processed.
    error : Optional[str]
        Set if any agent encounters a fatal error.
    metadata : Dict[str, Any]
        Arbitrary key/value bag for diagnostics (token counts, timings…).
    """

    raw_input: str
    mode: str
    clauses: List[str]
    current_clause: str
    results: List[ClauseResult]
    document_summary: Optional[str]
    error: Optional[str]
    metadata: Dict[str, Any]


def initial_state(raw_input: str, mode: str = "single") -> ClauseGuardState:
    """Return a blank-slate state dict ready for graph execution."""
    return ClauseGuardState(
        raw_input=raw_input,
        mode=mode,
        clauses=[],
        current_clause="",
        results=[],
        document_summary=None,
        error=None,
        metadata={},
    )
