"""
ClauseGuard — LangGraph Pipeline Definition
=============================================
Defines two compiled graphs:

  single_clause_graph   — 3-node chain for one clause
  document_graph        — iterative multi-clause pipeline

Graph topology (single mode):
  [START] → clause_extraction → risk_classification → explanation → [END]

Graph topology (document mode):
  [START] → clause_extraction → [loop over clauses] → document_summary → [END]
  Inside loop: risk_classification → explanation → next_clause_or_exit

Usage
-----
    from agents.graph import run_single_clause, run_document

    result = run_single_clause("The customer shall indemnify...")
    results = run_document(open("contract.txt").read())
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from agents.agent_nodes import (
    clause_extraction_agent,
    document_summary_agent,
    explanation_agent,
    risk_classification_agent,
)
from agents.state import ClauseGuardState, ClauseResult, initial_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_after_extraction(state: ClauseGuardState) -> str:
    """After extraction, check for errors and decide next step."""
    if state.get("error"):
        return END
    clauses = state.get("clauses", [])
    if not clauses:
        return END
    return "risk_classification"


def _route_after_explanation(state: ClauseGuardState) -> str:
    """
    After explaining one clause, decide whether to:
      - move to the next unprocessed clause, or
      - proceed to the document summary (document mode), or
      - finish (single mode).
    """
    if state.get("error"):
        return END

    mode = state.get("mode", "single")
    if mode == "single":
        return END

    # Document mode: have all clauses been classified?
    clauses = state.get("clauses", [])
    results = state.get("results", [])
    if len(results) < len(clauses):
        return "advance_clause"
    return "document_summary"


# ---------------------------------------------------------------------------
# Helper node: advance the "current_clause" pointer in document mode
# ---------------------------------------------------------------------------

def advance_clause_node(state: ClauseGuardState) -> dict:
    """Set current_clause to the next unprocessed clause."""
    clauses = state.get("clauses", [])
    results = state.get("results", [])
    next_idx = len(results)           # results length == number processed so far
    if next_idx < len(clauses):
        return {"current_clause": clauses[next_idx]}
    return {}


# ---------------------------------------------------------------------------
# Build: Single-Clause Graph
# ---------------------------------------------------------------------------

def _build_single_clause_graph() -> Any:
    """
    Simple linear graph for a single clause:
      extraction → classification → explanation
    """
    g = StateGraph(ClauseGuardState)

    g.add_node("clause_extraction",    clause_extraction_agent)
    g.add_node("risk_classification",  risk_classification_agent)
    g.add_node("explanation",          explanation_agent)

    g.add_edge(START, "clause_extraction")
    g.add_conditional_edges(
        "clause_extraction",
        _route_after_extraction,
        {
            "risk_classification": "risk_classification",
            END: END,
        },
    )
    g.add_edge("risk_classification", "explanation")
    g.add_edge("explanation", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Build: Document Graph
# ---------------------------------------------------------------------------

def _build_document_graph() -> Any:
    """
    Iterative graph for multi-clause documents:
      extraction → (classify → explain → advance) × N → summary
    """
    g = StateGraph(ClauseGuardState)

    g.add_node("clause_extraction",   clause_extraction_agent)
    g.add_node("risk_classification", risk_classification_agent)
    g.add_node("explanation",         explanation_agent)
    g.add_node("advance_clause",      advance_clause_node)
    g.add_node("document_summary",    document_summary_agent)

    g.add_edge(START, "clause_extraction")

    g.add_conditional_edges(
        "clause_extraction",
        _route_after_extraction,
        {
            "risk_classification": "risk_classification",
            END: END,
        },
    )

    g.add_edge("risk_classification", "explanation")

    g.add_conditional_edges(
        "explanation",
        _route_after_explanation,
        {
            "advance_clause":    "advance_clause",
            "document_summary":  "document_summary",
            END:                 END,
        },
    )

    # After advancing, re-classify the new current_clause
    g.add_edge("advance_clause", "risk_classification")
    g.add_edge("document_summary", END)

    return g.compile()


# Pre-compile graphs at import time for performance
_SINGLE_GRAPH   = _build_single_clause_graph()
_DOCUMENT_GRAPH = _build_document_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_single_clause(clause_text: str) -> ClauseResult:
    """
    Analyse a single clause through the agentic pipeline.

    Parameters
    ----------
    clause_text : str
        Raw clause text as entered by the user.

    Returns
    -------
    ClauseResult
        Structured result with category, confidence, explanation, etc.
    """
    state = initial_state(raw_input=clause_text, mode="single")
    final_state: ClauseGuardState = _SINGLE_GRAPH.invoke(state)

    results = final_state.get("results", [])
    if not results:
        # Return a safe fallback if the graph failed
        return ClauseResult(
            clause=clause_text,
            risk_category="Standard Clause",
            confidence=0.0,
            explanation="Analysis failed — check logs for details.",
            key_risk_terms=[],
            severity="Low",
        )
    return results[0]


def run_document(document_text: str) -> Dict[str, Any]:
    """
    Analyse a full contract document through the agentic pipeline.

    Parameters
    ----------
    document_text : str
        Full contract text.

    Returns
    -------
    dict with keys:
        results           : List[ClauseResult]
        document_summary  : dict (parsed JSON from summary agent)
        error             : str | None
    """
    state = initial_state(raw_input=document_text, mode="document")
    final_state: ClauseGuardState = _DOCUMENT_GRAPH.invoke(state)

    summary_raw = final_state.get("document_summary", "{}")
    try:
        summary = json.loads(summary_raw) if isinstance(summary_raw, str) else summary_raw
    except json.JSONDecodeError:
        summary = {"raw": summary_raw}

    return {
        "results":          final_state.get("results", []),
        "document_summary": summary,
        "error":            final_state.get("error"),
    }
