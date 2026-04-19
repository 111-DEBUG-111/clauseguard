"""
ClauseGuard — Individual Agent Nodes
======================================
Each function here is a LangGraph node.  They share the signature:
    node_fn(state: ClauseGuardState) -> dict   # partial state update

Nodes:
  1. clause_extraction_agent  — splits a contract into individual clauses
  2. risk_classification_agent — classifies a single clause
  3. explanation_agent         — enriches result with plain-English reasoning
  4. document_summary_agent    — synthesises a contract-level risk report
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from agents.llm_provider import get_llm
from agents.prompts import (
    clause_extraction_prompt,
    classification_prompt,
    explanation_prompt,
    summary_prompt,
)
from agents.state import ClauseGuardState, ClauseResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_safely(text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from an LLM response.

    Handles three common LLM quirks:
      1. Markdown code fences wrapping the JSON (```json ... ```)
      2. Unescaped control characters (raw newlines) inside string values —
         Groq llama models occasionally emit these in long summary fields.
      3. Trailing commas or other minor formatting issues (via strict=False).
    """
    # ── Step 1: strip markdown fences ────────────────────────────────────────
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    cleaned = cleaned.rstrip("` \n")

    # ── Step 2: strict JSON parse (fastest path) ──────────────────────────────
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ── Step 3: non-strict parse (allows control chars in strings) ────────────
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        pass

    # ── Step 4: sanitise unescaped control characters then retry ─────────────
    # Replace literal newlines / tabs inside JSON string values with their
    # escaped equivalents, leaving structural newlines (between keys) intact.
    def _escape_ctrl(m: re.Match) -> str:
        return m.group(0).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    sanitised = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', _escape_ctrl, cleaned, flags=re.DOTALL)
    try:
        return json.loads(sanitised)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed after all fallbacks.\nRaw text:\n%s\nError: %s", text, exc)
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc


def _invoke_chain(prompt, llm, **kwargs) -> Dict[str, Any]:
    """Format prompt, call LLM, parse JSON response."""
    chain = prompt | llm
    response = chain.invoke(kwargs)
    return _parse_json_safely(response.content)


# ---------------------------------------------------------------------------
# Agent 1: Clause Extraction
# ---------------------------------------------------------------------------

def clause_extraction_agent(state: ClauseGuardState) -> dict:
    """
    Split a full contract document into individual clause strings.

    Skips extraction in "single" mode — the raw_input IS the clause.

    State mutations
    ---------------
    clauses        : populated with extracted clause strings
    current_clause : set to the first clause (for single-turn pipelines)
    """
    if state["mode"] == "single":
        # No extraction needed; treat the whole input as one clause
        return {
            "clauses": [state["raw_input"]],
            "current_clause": state["raw_input"],
        }

    logger.info("Clause Extraction Agent — extracting from document …")
    llm = get_llm(temperature=0.0)

    try:
        parsed = _invoke_chain(
            clause_extraction_prompt,
            llm,
            document_text=state["raw_input"],
        )
        clauses: list[str] = parsed.get("clauses", [])
        logger.info("Extracted %d clauses.", len(clauses))
        return {
            "clauses": clauses,
            "current_clause": clauses[0] if clauses else "",
            "metadata": {**state.get("metadata", {}), "extracted_count": len(clauses)},
        }
    except Exception as exc:
        logger.exception("Clause extraction failed.")
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Agent 2: Risk Classification
# ---------------------------------------------------------------------------

def risk_classification_agent(state: ClauseGuardState) -> dict:
    """
    Classify a single clause into one of the 5 risk categories.

    Reads  : state["current_clause"]
    Writes : partial ClauseResult (no explanation yet) into state["results"]

    Note
    ----
    In document mode this agent is called *per clause* inside the graph
    loop.  See graph.py for the iteration pattern.
    """
    clause = state["current_clause"]
    if not clause:
        return {"error": "No clause text to classify."}

    logger.info("Risk Classification Agent — clause[:80]: %s …", clause[:80])
    llm = get_llm(temperature=0.0)

    try:
        parsed = _invoke_chain(
            classification_prompt,
            llm,
            clause_text=clause,
        )

        # Build a partial ClauseResult (explanation will be filled next)
        result: ClauseResult = {
            "clause": clause,
            "risk_category": parsed.get("risk_category", "Standard Clause"),
            "confidence": float(parsed.get("confidence", 0.5)),
            "explanation": parsed.get("explanation", ""),
            "key_risk_terms": parsed.get("key_risk_terms", []),
            "severity": parsed.get("severity", "Low"),
        }

        existing = state.get("results", [])
        return {"results": existing + [result]}

    except Exception as exc:
        logger.exception("Classification failed for clause: %s", clause[:80])
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Agent 3: Explanation (Enrichment)
# ---------------------------------------------------------------------------

def explanation_agent(state: ClauseGuardState) -> dict:
    """
    Enrich the most-recently classified clause result with a plain-English
    explanation and negotiation advice.

    Reads  : last item in state["results"]
    Writes : updates the last ClauseResult with richer explanation fields
    """
    results = state.get("results", [])
    if not results:
        return {}

    last: ClauseResult = results[-1]

    logger.info("Explanation Agent — enriching result for: %s …", last["clause"][:60])
    llm = get_llm(temperature=0.1)   # slight creativity for natural prose

    try:
        parsed = _invoke_chain(
            explanation_prompt,
            llm,
            clause_text=last["clause"],
            risk_category=last["risk_category"],
            severity=last["severity"],
            key_risk_terms=", ".join(last["key_risk_terms"]),
        )

        # Merge the richer explanation into the result
        enriched = {
            **last,
            "explanation": parsed.get("plain_english_summary", last["explanation"]),
            "risk_bearer": parsed.get("risk_bearer", "N/A"),
            "negotiation_advice": parsed.get("negotiation_advice", ""),
        }

        return {"results": results[:-1] + [enriched]}

    except Exception as exc:
        logger.exception("Explanation enrichment failed.")
        # Non-fatal — keep existing explanation
        return {}


# ---------------------------------------------------------------------------
# Agent 4: Document Summary
# ---------------------------------------------------------------------------

def document_summary_agent(state: ClauseGuardState) -> dict:
    """
    After all clauses are classified, produce a contract-level risk summary.

    Reads  : state["results"]  (all ClauseResult entries)
    Writes : state["document_summary"]
    """
    results = state.get("results", [])
    if not results:
        return {"document_summary": "No clauses were analysed."}

    logger.info("Document Summary Agent — summarising %d results …", len(results))
    llm = get_llm(temperature=0.0)

    # Provide the LLM with a compact version of results to manage context length
    compact = [
        {
            "clause_preview": r["clause"][:200],
            "risk_category": r["risk_category"],
            "confidence": r["confidence"],
            "severity": r["severity"],
            "key_risk_terms": r["key_risk_terms"],
        }
        for r in results
    ]

    try:
        parsed = _invoke_chain(
            summary_prompt,
            llm,
            results_json=json.dumps(compact, indent=2),
        )
        return {"document_summary": json.dumps(parsed, indent=2)}
    except Exception as exc:
        logger.exception("Document summary failed.")
        return {"document_summary": f"Summary generation failed: {exc}"}
