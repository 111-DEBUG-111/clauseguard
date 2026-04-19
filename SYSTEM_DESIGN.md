# ClauseGuard v2 — Agentic AI System Design
### LangGraph + Grok LLM · Complete Architecture & Implementation Guide

---

## 1. Architecture Overview

### 1.1 High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                   │
│  ┌──────────────────┐          ┌───────────────────────────────┐    │
│  │  Single Clause   │          │   Full Contract Document      │    │
│  │  (text area)     │          │   (.txt upload / paste)       │    │
│  └────────┬─────────┘          └──────────────┬────────────────┘    │
└───────────┼──────────────────────────────────┼─────────────────────┘
            │                                  │
            ▼                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   LANGGRAPH AGENTIC PIPELINE                         │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  AGENT 1: CLAUSE EXTRACTION                                  │   │
│  │  Skipped in single mode.                                     │   │
│  │  LLM segments document into self-contained clauses.          │   │
│  │  Returns { clauses: [...], total_count: N }                  │   │
│  └───────────────────────┬──────────────────────────────────────┘   │
│                           │  clauses[]                               │
│  ┌────────────────────────▼──────────────────────────────────────┐  │
│  │  AGENT 2: RISK CLASSIFICATION  (once per clause)              │  │
│  │  LLM classifies into 5 categories.                            │  │
│  │  Returns: category, confidence 0-1, severity, key terms.      │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                           │  partial ClauseResult                    │
│  ┌────────────────────────▼──────────────────────────────────────┐  │
│  │  AGENT 3: EXPLANATION / ENRICHMENT                            │  │
│  │  Plain-English explanation for non-lawyers.                   │  │
│  │  Identifies risk bearer + negotiation advice.                 │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
│                           │  enriched ClauseResult                   │
│  ┌────────────────────────▼──────────────────────────────────────┐  │
│  │  AGENT 4: DOCUMENT SUMMARY  (document mode only)              │  │
│  │  Synthesises all clause results.                              │  │
│  │  Overall risk rating: Critical / High / Medium / Low          │  │
│  │  Risk distribution, top risks, recommendations.              │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                                   │
│                                                                      │
│  Per-clause JSON:                                                    │
│  {                                                                   │
│    "clause": "...",                                                  │
│    "risk_category": "Indemnity Risk",                               │
│    "confidence": 0.92,                                               │
│    "explanation": "...",                                             │
│    "key_risk_terms": ["indemnify", "hold harmless", "attorneys"],   │
│    "severity": "High",                                               │
│    "risk_bearer": "Vendor",                                          │
│    "negotiation_advice": "...",                                      │
│    "requires_human_review": false                                    │
│  }                                                                   │
│                                                                      │
│  Document JSON:  Executive summary + risk distribution dashboard     │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 Classical ML vs Agentic LLM Comparison

| Dimension | Phase 1 (TF-IDF + LR) | Phase 2 (Agentic LLM) |
|---|---|---|
| **Semantic understanding** | Bag-of-words; no meaning | Full contextual understanding |
| **Generalisation** | Fails on unseen vocab | Zero-shot on any legal text |
| **Interpretability** | Feature coefficients only | Natural-language explanation |
| **Training data** | Requires labelled dataset | No retraining needed |
| **Compound clauses** | Single label for complex text | Reasons about multiple intents |
| **Document-level synthesis** | None | Executive summary + patterns |
| **Negotiation support** | None | Risk bearer + negotiation advice |
| **RAG augmentation** | Not possible | Retrieves legal precedents |
| **Guardrails** | None | Validates outputs, flags low conf |

---

## 2. Agent Design

### Agent 1 — Clause Extraction Agent
```
Input:  Full contract document (raw text)
Task:   Segment document into self-contained contractual clauses
Output: { "clauses": ["...", "..."], "total_count": N }
Temp:   0.0  (deterministic extraction)
Skip:   Bypassed in single-clause mode
```

### Agent 2 — Risk Classification Agent
```
Input:  Single clause text
Task:   Classify into 5 risk categories + confidence + severity + key terms
Output: { risk_category, confidence, explanation, key_risk_terms, severity }
Temp:   0.0  (consistent, repeatable classification)
Runs:   Once per clause (iterated by LangGraph loop)
```

### Agent 3 — Explanation Agent
```
Input:  Clause text + preliminary classification from Agent 2
Task:   Generate plain-English explanation + risk bearer + negotiation advice
Output: { plain_english_summary, risk_bearer, negotiation_advice }
Temp:   0.1  (slight creativity for natural prose)
Runs:   Immediately after each classification
```

### Agent 4 — Document Summary Agent
```
Input:  All ClauseResult objects from Agents 2 + 3
Task:   Synthesise overall contract risk profile
Output: { overall_risk_rating, risk_distribution, executive_summary,
          highest_risk_clauses, recommendations }
Temp:   0.0  (factual, deterministic summary)
Runs:   Once, after all clauses are processed
```

---

## 3. LangGraph Flow

### 3.1 Shared State Schema

```python
class ClauseGuardState(TypedDict):
    raw_input:        str                  # User's input text
    mode:             str                  # "single" | "document"
    clauses:          List[str]            # Extracted clause strings
    current_clause:   str                  # Clause being processed now
    results:          List[ClauseResult]   # Accumulated results
    document_summary: Optional[str]        # Final executive summary JSON
    error:            Optional[str]        # Error string if pipeline fails
    metadata:         Dict[str, Any]       # Diagnostic info (tokens, timing)
```

### 3.2 Single-Clause Graph

```
[START] → clause_extraction → risk_classification → explanation → [END]
                    │
              (error) → [END]
```

### 3.3 Document Graph (with iteration loop)

```
[START]
   │
   ▼
clause_extraction ──(error/empty)──► [END]
   │
   ▼
risk_classification ◄──────────────────────────┐
   │                                            │
   ▼                                            │
explanation                                     │
   │                                            │
   ├──(more clauses remain)──► advance_clause ──┘
   │
   └──(all done)──► document_summary → [END]
```

### 3.4 Conditional Routing

```python
# After clause_extraction:
if error or clauses == []  →  END
else                       →  risk_classification

# After explanation:
if mode == "single"                    →  END
elif len(results) < len(clauses)       →  advance_clause   # iterate
else                                   →  document_summary  # done
```

---

## 4. Prompt Design

### 4.1 Classification Prompt (Agent 2)

**System:**
```
You are an expert legal risk analyst specialising in commercial contract review.
Your job is to classify a given clause into exactly one of the following risk categories:

  1. Indemnity Risk   — clauses requiring one party to compensate for losses/damages
  2. Liability Risk   — clauses limiting, capping, or disclaiming liability  
  3. Termination Risk — clauses defining conditions for ending the contract
  4. Payment Risk     — clauses related to payment terms, fees, penalties
  5. Standard Clause  — routine boilerplate with minimal inherent risk

Reasoning approach:
  1. Identify the primary legal purpose of the clause.
  2. Detect risk-indicative language (key terms below).
  3. Assign the category matching the dominant legal purpose.
  4. Estimate confidence on a 0-1 scale.

Key terms:
  Indemnity:    "indemnify", "hold harmless", "defend"
  Liability:    "limit", "cap", "disclaim", "in no event shall"
  Termination:  "terminate", "notice", "breach", "cure period"
  Payment:      "invoice", "fee", "interest", "penalty"

Return ONLY a valid JSON object (no markdown fences, no prose):
{
  "risk_category": "<one of the 5 categories>",
  "confidence": <float 0.0–1.0>,
  "explanation": "<2-3 sentence explanation>",
  "key_risk_terms": ["term1", "term2", ...],
  "severity": "<High|Medium|Low>"
}
```

**Human:**
```
Classify the following contractual clause:

<clause>
{clause_text}
</clause>
```

**Design principles:**
- XML delimiters prevent prompt injection
- Explicit "no markdown fences" prevents JSON parse failures
- Numbered reasoning steps guide chain-of-thought
- Low temperature (0.0) ensures repeatability

### 4.2 Explanation Prompt (Agent 3)

**System:**
```
You are a legal risk communication specialist.
Given a contract clause and its preliminary risk assessment, explain the
risks in plain English that a business executive can understand.

Your explanation must:
- Describe what the clause means in practical terms.
- Explain WHY it is risky (if it is).
- State WHO bears the risk (which party).
- Suggest what a negotiator might push back on.
- Be concise (3-5 sentences max).

Return ONLY a valid JSON object:
{
  "plain_english_summary": "<3-5 sentence explanation>",
  "risk_bearer": "<Party A | Party B | Both | N/A>",
  "negotiation_advice": "<1-2 sentence actionable advice>"
}
```

### 4.3 Clause Extraction Prompt (Agent 1)

**System:**
```
You are a skilled legal document analyst.
Extract individual, self-contained contractual clauses from the contract.

Rules:
- Each clause must express a single contractual obligation, right, or condition.
- Do NOT split one clause across multiple entries.
- Ignore headings, TOC entries, and signature blocks.
- Return ONLY valid JSON (no markdown fences).

Output: { "clauses": ["...", "..."], "total_count": <int> }
```

### 4.4 Document Summary Prompt (Agent 4)

**System:**
```
You are a senior legal risk officer providing an executive summary.

Given clause-level risk results:
1. Identify the highest-risk clauses.
2. Highlight patterns or concentrations of risk.
3. Give an overall risk rating: Critical / High / Medium / Low.
4. Provide concise recommendations.

Return ONLY valid JSON:
{
  "overall_risk_rating": "<Critical|High|Medium|Low>",
  "total_clauses": <int>,
  "risk_distribution": { ... },
  "highest_risk_clauses": [...],
  "executive_summary": "<3-4 sentence summary>",
  "recommendations": ["Rec 1", ...]
}
```

---

## 5. Output Format

### Per-Clause Result

```json
{
  "clause": "The Vendor shall indemnify and hold harmless the Client...",
  "risk_category": "Indemnity Risk",
  "confidence": 0.95,
  "explanation": "This clause places a broad, unlimited indemnification obligation on the Vendor. It requires the Vendor to pay all legal costs arising from any breach, even those partly caused by the Client. This represents significant financial exposure for the Vendor.",
  "key_risk_terms": ["indemnify", "hold harmless", "attorneys' fees", "unlimited"],
  "severity": "High",
  "risk_bearer": "Vendor",
  "negotiation_advice": "Push for mutual indemnification and a monetary cap tied to contract value.",
  "requires_human_review": false
}
```

### Document-Level Summary

```json
{
  "overall_risk_rating": "High",
  "total_clauses": 12,
  "risk_distribution": {
    "Indemnity Risk": 2,
    "Liability Risk": 3,
    "Termination Risk": 2,
    "Payment Risk": 2,
    "Standard Clause": 3
  },
  "highest_risk_clauses": [
    { "clause_preview": "The Vendor shall indemnify...", "risk_category": "Indemnity Risk", "severity": "High" }
  ],
  "executive_summary": "This contract carries a High overall risk rating driven by uncapped indemnification obligations and aggressive liability caps favouring the service provider.",
  "recommendations": [
    "Negotiate a monetary cap on indemnification obligations.",
    "Request mutual liability caps.",
    "Extend the cure period for breach from 3 to 15 business days."
  ]
}
```

---

## 6. Project Structure

```
clauseguard/
├── agents/
│   ├── __init__.py          # Package marker
│   ├── state.py             # LangGraph TypedDict state schema
│   ├── prompts.py           # All LLM prompt templates (centralised)
│   ├── llm_provider.py      # Grok / OpenAI factory (env-var driven)
│   ├── agent_nodes.py       # 4 agent node functions
│   ├── graph.py             # LangGraph pipeline + public API
│   ├── rag.py               # Optional FAISS RAG module
│   └── guardrails.py        # Input/output safety validation
├── src/                     # Phase 1 (preserved for fallback)
├── app.py                   # Phase 1 Streamlit app (preserved)
├── app_v2.py                # Phase 2 Agentic Streamlit app
├── run_pipeline.py          # CLI test harness
├── requirements.txt         # Phase 1 + Phase 2 dependencies
└── .env.example             # API key template
```

### Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key
export XAI_API_KEY="xai-your-key"      # Grok (preferred)
# export OPENAI_API_KEY="sk-..."        # OpenAI (fallback)

# 3. CLI test
python run_pipeline.py --mode single --clause-type indemnity
python run_pipeline.py --mode document

# 4. Launch UI
streamlit run app_v2.py
```

---

## 7. Phase 1 vs Phase 2 Comparison

### Improvements

| Aspect | Phase 1 | Phase 2 | Improvement |
|---|---|---|---|
| Semantic understanding | TF-IDF token frequency | LLM contextual reasoning | Understands legal nuance, synonyms, negation |
| Generalisation | Limited to training distribution | Zero-shot on novel text | No retraining for new clause types |
| Interpretability | Top TF-IDF features | Plain-English explanation | Lawyers + executives can understand |
| Multi-intent clauses | One label per clause | Identifies dominant risk | More accurate for complex clauses |
| Document synthesis | Simple aggregation | Intelligent summary | Actionable insights, not just counts |
| Negotiation guidance | None | Specific advice per clause | Immediately actionable |

### Trade-offs

| Trade-off | Details | Mitigation |
|---|---|---|
| Latency | 2-8s per clause vs <100ms for ML | Async processing |
| Cost | ~$0.002–0.01 per clause | Cache results, tiered model selection |
| Hallucination | LLM may invent reasoning | Guardrails + structured JSON + temp=0 |
| API dependency | Requires internet + API key | Offline Phase 1 fallback |

---

## 8. Advanced Features

### RAG — Retrieval-Augmented Generation

`agents/rag.py` builds a FAISS vector store from a legal knowledge base:
1. Similar legal risk rules retrieved via semantic similarity
2. Injected into classification prompt as contextual grounding
3. Reduces hallucination; increases accuracy on edge cases

Enable: `pip install faiss-cpu` (already in requirements.txt)

### Guardrails

`agents/guardrails.py` implements three layers:
- **Input Guard**: Rejects empty, too-short, too-long, or non-textual input
- **Output Guard**: Corrects invalid categories/severities; clamps confidence to [0,1]
- **Human-Review Flag**: `requires_human_review: true` when confidence < 0.45

### Architecture Decision Records

**ADR-001: Why LangGraph?**  
Multi-node pipeline provides separation of concerns, independent agent upgrades, inspectable state, natural iteration loop for documents, and easy extensibility.

**ADR-002: Why Grok primary, OpenAI fallback?**  
Grok-3 competitive legal reasoning + OpenAI-compatible API = zero-code provider switching via env var.

**ADR-003: Why preserve Phase 1?**  
Phase 1 provides offline fallback, enables A/B comparison, and can serve as a fast pre-filter for obvious Standard Clauses.
