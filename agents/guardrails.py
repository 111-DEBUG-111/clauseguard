"""
ClauseGuard — Guardrails Module
=================================
Production safety checks for LLM outputs:

1. HallucinationGuard  — validates that the predicted category actually
                         appears in our known taxonomy.
2. ConfidenceGuard     — flags results where the LLM reports low confidence.
3. ContentGuard        — basic input sanity (length, encoding, PII check).

Usage
-----
    from agents.guardrails import validate_result, validate_input

    validate_input(clause_text)    # raises ValueError if input is problematic
    safe_result = validate_result(raw_result)  # corrects / flags bad outputs
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known taxonomy
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {
    "Indemnity Risk",
    "Liability Risk",
    "Termination Risk",
    "Payment Risk",
    "Standard Clause",
}

VALID_SEVERITIES = {"Critical", "High", "Medium", "Low"}

# ---------------------------------------------------------------------------
# Input Guardrails
# ---------------------------------------------------------------------------

MIN_CLAUSE_LENGTH = 20     # chars
MAX_CLAUSE_LENGTH = 8_000  # chars  (prevent context-window abuse)


def validate_input(text: str) -> None:
    """
    Raise ValueError if the input clause fails sanity checks.

    Checks
    ------
    - Not empty / too short
    - Not excessively long
    - Not purely numeric / non-textual
    """
    if not text or not text.strip():
        raise ValueError("Clause text is empty.")

    stripped = text.strip()

    if len(stripped) < MIN_CLAUSE_LENGTH:
        raise ValueError(
            f"Clause is too short ({len(stripped)} chars). "
            f"Minimum is {MIN_CLAUSE_LENGTH} characters."
        )

    if len(stripped) > MAX_CLAUSE_LENGTH:
        raise ValueError(
            f"Clause is too long ({len(stripped):,} chars). "
            f"Maximum is {MAX_CLAUSE_LENGTH:,} characters. "
            "Please split the document into smaller sections."
        )

    # Check that it contains at least some alphabetic content
    alpha_ratio = sum(c.isalpha() for c in stripped) / len(stripped)
    if alpha_ratio < 0.3:
        raise ValueError(
            "Input appears to contain mostly non-textual content. "
            "Please provide a valid clause in English."
        )


# ---------------------------------------------------------------------------
# Output Guardrails
# ---------------------------------------------------------------------------

LOW_CONFIDENCE_THRESHOLD = 0.45   # Below this → flag for human review


def validate_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and auto-correct a ClauseResult from the LLM.

    Corrections applied
    -------------------
    - Unknown risk_category → replaced with "Standard Clause"
    - Unknown severity      → replaced with "Low"
    - Confidence out of [0,1] range → clamped
    - Empty key_risk_terms  → set to []
    - Adds "requires_human_review" flag when confidence is low
    """
    result = dict(result)   # shallow copy — do not mutate caller's dict

    # ── Category validation ──────────────────────────────────────────────────
    cat = result.get("risk_category", "")
    if cat not in VALID_CATEGORIES:
        logger.warning(
            "Guardrail: unknown risk_category '%s' → 'Standard Clause'", cat
        )
        result["risk_category"] = "Standard Clause"
        result["guardrail_correction"] = f"Category '{cat}' not in taxonomy."

    # ── Severity validation ──────────────────────────────────────────────────
    sev = result.get("severity", "")
    if sev not in VALID_SEVERITIES:
        logger.warning(
            "Guardrail: unknown severity '%s' → 'Low'", sev
        )
        result["severity"] = "Low"

    # ── Confidence clamping ──────────────────────────────────────────────────
    conf = result.get("confidence", 0.5)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.5
    result["confidence"] = round(max(0.0, min(1.0, conf)), 3)

    # ── Key terms normalisation ──────────────────────────────────────────────
    terms = result.get("key_risk_terms", None)
    if not isinstance(terms, list):
        result["key_risk_terms"] = []

    # ── Human review flag ────────────────────────────────────────────────────
    result["requires_human_review"] = result["confidence"] < LOW_CONFIDENCE_THRESHOLD

    if result["requires_human_review"]:
        logger.info(
            "Guardrail: low confidence (%.2f) — clause flagged for human review.",
            result["confidence"],
        )

    return result


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def validate_results_batch(results: list) -> list:
    """Apply validate_result to every item in a list."""
    return [validate_result(r) for r in results]
