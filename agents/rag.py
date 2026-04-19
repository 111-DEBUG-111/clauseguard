"""
ClauseGuard — RAG (Retrieval-Augmented Generation) Module  [OPTIONAL / ADVANCED]
==================================================================================
Augments the risk classification pipeline with a retrieval step:
  1. A local FAISS vector store is built from a legal knowledge base
     (sample risk rules shipped in data/legal_kb.json).
  2. Before classifying a clause, similar precedents + risk rules are
     retrieved and injected into the classification prompt as context.

This module is entirely OPTIONAL.  The main pipeline works without it.

Usage
-----
    from agents.rag import get_rag_context
    context = get_rag_context("indemnify and hold harmless")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Graceful optional imports ─────────────────────────────────────────────────
try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False
    logger.warning("RAG dependencies not installed (langchain-community, faiss-cpu). RAG disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# Sample Legal Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_KB = [
    {
        "id": "IND-001",
        "category": "Indemnity Risk",
        "rule": "Mutual indemnification clauses are lower risk than unilateral ones. Watch for broad 'including but not limited to' language.",
        "example": "Each party shall indemnify the other from third-party claims arising from its own negligence.",
        "risk_level": "Medium",
    },
    {
        "id": "IND-002",
        "category": "Indemnity Risk",
        "rule": "Uncapped indemnification obligations represent high financial exposure. Always check for monetary caps.",
        "example": "Vendor shall indemnify Client from all claims, damages, costs and expenses without limitation.",
        "risk_level": "High",
    },
    {
        "id": "LIA-001",
        "category": "Liability Risk",
        "rule": "Exclusions of consequential damages are standard but check if they are mutual. One-sided exclusions favour the drafter.",
        "example": "In no event shall Provider be liable for indirect, incidental, or consequential damages.",
        "risk_level": "Medium",
    },
    {
        "id": "LIA-002",
        "category": "Liability Risk",
        "rule": "Liability caps below 6 months of contract value are considered extremely restrictive.",
        "example": "Total liability shall not exceed the fees paid in the preceding 30 days.",
        "risk_level": "High",
    },
    {
        "id": "TER-001",
        "category": "Termination Risk",
        "rule": "Termination for convenience clauses without notice or penalties create business continuity risk.",
        "example": "Either party may terminate this agreement immediately without cause or notice.",
        "risk_level": "High",
    },
    {
        "id": "TER-002",
        "category": "Termination Risk",
        "rule": "Cure periods shorter than 10 business days are considered aggressive and risk contract loss for minor breaches.",
        "example": "Failure to cure a breach within 3 days shall result in immediate termination.",
        "risk_level": "High",
    },
    {
        "id": "PAY-001",
        "category": "Payment Risk",
        "rule": "Interest rates above 1.5% per month on late payments are aggressive; check jurisdiction usury laws.",
        "example": "Late payments will accrue interest at 2% per month.",
        "risk_level": "Medium",
    },
    {
        "id": "PAY-002",
        "category": "Payment Risk",
        "rule": "Automatic price escalation clauses without caps represent open-ended financial commitments.",
        "example": "Fees shall increase annually by the greater of 5% or CPI.",
        "risk_level": "Medium",
    },
    {
        "id": "STD-001",
        "category": "Standard Clause",
        "rule": "Governing law and jurisdiction clauses are standard but ensure the chosen jurisdiction is favourable.",
        "example": "This Agreement shall be governed by the laws of the State of Delaware.",
        "risk_level": "Low",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store (lazy-initialised singleton)
# ─────────────────────────────────────────────────────────────────────────────

_vector_store: Optional[object] = None  # FAISS instance


def _build_vector_store() -> Optional[object]:
    """Build or load a FAISS index from the legal knowledge base."""
    if not _RAG_AVAILABLE:
        return None

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY")
    if not api_key:
        logger.warning("No API key for embeddings — RAG disabled.")
        return None

    # Load from disk if already built
    index_path = Path("data/rag_index")
    if index_path.exists():
        try:
            embeddings = OpenAIEmbeddings()
            vs = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded RAG index from disk.")
            return vs
        except Exception as exc:
            logger.warning("Failed to load RAG index: %s — rebuilding.", exc)

    # Build from SAMPLE_KB
    docs = []
    for entry in SAMPLE_KB:
        content = (
            f"[{entry['category']}] {entry['rule']}\n"
            f"Example: {entry['example']}\n"
            f"Risk Level: {entry['risk_level']}"
        )
        docs.append(Document(page_content=content, metadata=entry))

    try:
        embeddings = OpenAIEmbeddings()
        vs = FAISS.from_documents(docs, embeddings)
        index_path.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(index_path))
        logger.info("Built and saved RAG index with %d documents.", len(docs))
        return vs
    except Exception as exc:
        logger.warning("RAG index build failed: %s", exc)
        return None


def _get_or_build_store() -> Optional[object]:
    global _vector_store
    if _vector_store is None:
        _vector_store = _build_vector_store()
    return _vector_store


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_rag_context(clause_text: str, k: int = 3) -> str:
    """
    Retrieve the top-k relevant legal rules / precedents for a clause.

    Returns a formatted string ready to be injected into an LLM prompt.
    Returns an empty string if RAG is unavailable.
    """
    store = _get_or_build_store()
    if store is None:
        return ""

    try:
        docs = store.similarity_search(clause_text, k=k)
        if not docs:
            return ""

        lines = ["Relevant legal precedents and risk rules:"]
        for i, doc in enumerate(docs, 1):
            lines.append(f"\n[Rule {i}]\n{doc.page_content}")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return ""
