#!/usr/bin/env python3
"""
ClauseGuard — CLI Quick-Test Script
=====================================
Run this script to verify the agentic pipeline without starting Streamlit.

Usage
-----
    # Single clause (hard-coded examples)
    python run_pipeline.py --mode single

    # Full document from file
    python run_pipeline.py --mode document --file data/raw/sample_contract.txt

    # Full document from stdin
    echo "Contract text..." | python run_pipeline.py --mode document --stdin

Environment
-----------
    export XAI_API_KEY="xai-..."       # for Grok
    export OPENAI_API_KEY="sk-..."     # for OpenAI fallback
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from agents.graph import run_single_clause, run_document
from agents.guardrails import validate_input, validate_results_batch


# ─────────────────────────────────────────────────────────────────────────────
# Sample data
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CLAUSES = {
    "indemnity": (
        "The Vendor shall indemnify, defend, and hold harmless the Client, "
        "its officers, directors, employees, and agents from and against any "
        "and all claims, damages, losses, costs, and expenses (including "
        "reasonable attorneys' fees) arising out of or relating to any breach "
        "of this Agreement by the Vendor or the Vendor's negligent acts."
    ),
    "liability": (
        "IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER PARTY FOR ANY "
        "INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE, OR CONSEQUENTIAL "
        "DAMAGES, REGARDLESS OF THE CAUSE OF ACTION OR THE THEORY OF LIABILITY. "
        "EACH PARTY'S TOTAL CUMULATIVE LIABILITY ARISING OUT OF OR RELATED TO "
        "THIS AGREEMENT SHALL NOT EXCEED THE GREATER OF (A) THE TOTAL FEES PAID "
        "BY CLIENT IN THE TWELVE MONTHS PRECEDING THE CLAIM OR (B) USD 50,000."
    ),
    "termination": (
        "Either party may terminate this Agreement for cause if the other party "
        "materially breaches any provision of this Agreement and fails to cure "
        "such breach within ten (10) business days after receiving written notice "
        "describing the breach in reasonable detail. Either party may also "
        "terminate this Agreement for convenience upon thirty (30) days' "
        "prior written notice to the other party."
    ),
    "payment": (
        "Client shall pay all invoices within thirty (30) days of the invoice date. "
        "Any amounts not paid by the due date shall accrue late payment interest "
        "at the rate of one and one-half percent (1.5%) per month or the maximum "
        "rate permitted by applicable law, whichever is less, from the due date "
        "until the date of actual payment."
    ),
    "standard": (
        "This Agreement shall be governed by and construed in accordance with the "
        "laws of the State of New York, without regard to its conflict of law "
        "principles. The parties irrevocably consent to the exclusive jurisdiction "
        "of the state and federal courts located in New York County, New York."
    ),
}

SAMPLE_DOCUMENT = "\n\n".join(SAMPLE_CLAUSES.values())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ClauseGuard v2 — Agentic AI Pipeline")
    parser.add_argument(
        "--mode",
        choices=["single", "document"],
        default="single",
        help="Analysis mode",
    )
    parser.add_argument(
        "--clause-type",
        choices=list(SAMPLE_CLAUSES.keys()),
        default="indemnity",
        help="Which sample clause to use in single mode",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a .txt contract file (document mode)",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read document from stdin (document mode)",
    )
    args = parser.parse_args()

    # ── Single clause mode ────────────────────────────────────────────────────
    if args.mode == "single":
        clause = SAMPLE_CLAUSES[args.clause_type]
        print(f"\n{'='*70}")
        print(f"  CLAUSEGUARD v2 — SINGLE CLAUSE ANALYSIS")
        print(f"  Sample type: {args.clause_type.upper()}")
        print(f"{'='*70}")
        print(f"\nClause:\n{clause}\n")

        validate_input(clause)
        result = run_single_clause(clause)

        print(f"\n{'─'*70}")
        print(f"  RESULT")
        print(f"{'─'*70}")
        print(json.dumps(result, indent=2, default=str))

    # ── Document mode ─────────────────────────────────────────────────────────
    else:
        if args.stdin:
            document_text = sys.stdin.read()
        elif args.file:
            with open(args.file, "r", encoding="utf-8") as fh:
                document_text = fh.read()
        else:
            print("No file or --stdin provided; using built-in sample document.\n")
            document_text = SAMPLE_DOCUMENT

        print(f"\n{'='*70}")
        print(f"  CLAUSEGUARD v2 — FULL DOCUMENT ANALYSIS")
        print(f"  Document length: {len(document_text):,} chars")
        print(f"{'='*70}\n")

        output = run_document(document_text)
        results   = output["results"]
        summary   = output["document_summary"]
        error     = output["error"]

        if error:
            print(f"⚠️  Pipeline error: {error}")

        # Apply guardrails
        results = validate_results_batch(results)

        print(f"\n✅ Analysed {len(results)} clauses\n")

        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['risk_category']}  ({r['severity']})  conf={r['confidence']:.2f}")
            print(f"       {r['clause'][:100]}…")
            print(f"       Terms: {', '.join(r['key_risk_terms'][:4])}")
            if r.get("requires_human_review"):
                print(f"       ⚠️  LOW CONFIDENCE — FLAG FOR HUMAN REVIEW")
            print()

        if summary:
            print(f"\n{'─'*70}")
            print("  DOCUMENT SUMMARY")
            print(f"{'─'*70}")
            print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
