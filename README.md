<div align="center">

# ⚖️ ClauseGuard

### AI-Powered Legal Contract Risk Analysis

**Phase 1 · Classical ML** &nbsp;|&nbsp; **Phase 2 · Agentic AI with LangGraph + Groq LLM**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-6c63ff)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-LLM_Backend-f55036)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 📌 Overview

**ClauseGuard** is a contract risk classification system that automatically identifies and analyses risky clauses buried in legal documents — enabling non-experts to understand their exposure before signing.

The project evolved through **two phases**:

| | Phase 1 | Phase 2 |
|---|---|---|
| **Approach** | Classical ML (TF-IDF + Logistic Regression) | Agentic AI (LangGraph + Groq LLM) |
| **Accuracy** | ~96% on labelled test set | Semantic reasoning, zero-shot generalisation |
| **Explanation** | Probability scores | Plain-English reasoning + negotiation advice |
| **Document-level** | Clause-by-clause aggregation | Executive summary with risk dashboard |
| **App** | `app.py` | `app_v2.py` |

---

## 🚨 Problem Statement

Legal contracts are dense and technical. Critical risk clauses — indemnification obligations, liability caps, termination traps, aggressive payment penalties — are often buried in pages of legalese.

**ClauseGuard solves this by:**
- Automatically extracting individual clauses from a contract
- Classifying each clause into one of 5 risk categories
- Explaining *why* a clause is risky in plain English
- Providing negotiation advice for high-risk clauses
- Generating an executive risk summary for the full document

---

## 🏗️ Architecture

### Phase 2 — LangGraph Agentic Pipeline

```
Input (clause or full contract)
        │
        ▼
┌─────────────────────────────┐
│  Agent 1: Clause Extraction │  ← Skipped in single-clause mode
│  Groq LLM segments document │
└──────────────┬──────────────┘
               │  clauses[]
               ▼  (loop per clause)
┌─────────────────────────────┐
│  Agent 2: Classification    │  ← category · confidence · severity · key terms
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Agent 3: Explanation       │  ← plain English · risk bearer · negotiation advice
└──────────────┬──────────────┘
               │  (document mode only)
               ▼
┌─────────────────────────────┐
│  Agent 4: Document Summary  │  ← overall rating · distribution · recommendations
└─────────────────────────────┘
               │
               ▼
        Structured JSON output
```

### Risk Categories

| Category | Description |
|---|---|
| 🛡️ **Indemnity Risk** | One party must compensate the other for losses or damages |
| ⚡ **Liability Risk** | Clauses that limit, cap, or disclaim liability |
| 🚫 **Termination Risk** | Conditions for ending the contract; notice periods |
| 💰 **Payment Risk** | Payment terms, late fees, price escalation |
| 📋 **Standard Clause** | Routine boilerplate with minimal inherent risk |

---

## 📂 Project Structure

```
clauseguard/
│
├── agents/                      # Phase 2 — Agentic AI
│   ├── __init__.py
│   ├── state.py                 # LangGraph TypedDict state schema
│   ├── prompts.py               # All LLM prompt templates (centralised)
│   ├── llm_provider.py          # Groq / xAI / OpenAI factory (env-driven)
│   ├── agent_nodes.py           # 4 agent node functions
│   ├── graph.py                 # LangGraph pipeline + public API
│   ├── rag.py                   # Optional RAG module (FAISS + legal KB)
│   └── guardrails.py            # Input/output safety validation
│
├── src/                         # Phase 1 — Classical ML (preserved)
│   ├── preprocessing.py
│   ├── prepare_dataset.py
│   ├── train_model.py
│   └── predict.py
│
├── data/
│   ├── raw/                     # CUAD dataset (gitignored)
│   └── processed/               # Processed CSV
│
├── models/                      # Trained Phase 1 models
│   ├── logistic_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── confusion_matrix.png
│
├── app.py                       # Phase 1 Streamlit app
├── app_v2.py                    # Phase 2 Agentic Streamlit app  ← run this
├── run_pipeline.py              # CLI test harness (no UI required)
├── requirements.txt             # All dependencies
├── .env.example                 # API key template — copy to .env
└── SYSTEM_DESIGN.md             # Full architecture + prompt documentation
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) *(keys start with `gsk_`)*

### 1. Clone & Install

```bash
git clone https://github.com/your-username/clauseguard.git
cd clauseguard

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
```

Edit `.env` — add your Groq key (no spaces, no quotes):

```env
GROQ_API_KEY=gsk_your_key_here
```

> **Free alternatives:** xAI Grok (`XAI_API_KEY=xai-...`) or OpenAI (`OPENAI_API_KEY=sk-...`)  
> The app auto-detects which key is present.

### 3. Run the App

```bash
# Phase 2 — Agentic AI (recommended)
streamlit run app_v2.py

# Phase 1 — Classical ML (offline fallback)
streamlit run app.py
```

### 4. CLI Test (no UI needed)

```bash
# Test single clause
python3 run_pipeline.py --mode single --clause-type indemnity
python3 run_pipeline.py --mode single --clause-type liability
python3 run_pipeline.py --mode single --clause-type termination
python3 run_pipeline.py --mode single --clause-type payment
python3 run_pipeline.py --mode single --clause-type standard

# Test full document
python3 run_pipeline.py --mode document

# Test with your own file
python3 run_pipeline.py --mode document --file path/to/contract.txt
```

---

## 🖥️ App Features

### Phase 2 — `app_v2.py` (Agentic AI)

| Feature | Description |
|---|---|
| **Single Clause Analysis** | Paste any clause → get category, confidence, explanation, negotiation advice |
| **Full Document Analysis** | Upload `.txt` contract → all clauses extracted and classified automatically |
| **Executive Dashboard** | Overall risk rating (Critical/High/Medium/Low) + risk distribution chart |
| **Clause Cards** | Expandable per-clause panels with colour-coded severity badges |
| **Sort & Filter** | Sort by severity or confidence; filter to risky clauses only |
| **JSON Export** | Download full structured report as `.json` |
| **Example Buttons** | One-click sample clauses for testing |

### Phase 1 — `app.py` (Classical ML)

| Feature | Description |
|---|---|
| **Single Clause** | TF-IDF + Logistic Regression → category + probability distribution |
| **Full Contract** | Upload `.txt` → clause segmentation → batch classification table |

---

## 📊 Output Schema

Every clause produces a structured JSON result:

```json
{
  "clause": "The Vendor shall indemnify, defend, and hold harmless the Client...",
  "risk_category": "Indemnity Risk",
  "confidence": 0.95,
  "explanation": "This clause places a broad indemnification obligation on the Vendor, requiring them to cover all legal costs arising from any breach. This represents significant financial exposure with no monetary cap.",
  "key_risk_terms": ["indemnify", "hold harmless", "defend", "attorneys' fees"],
  "severity": "High",
  "risk_bearer": "Vendor",
  "negotiation_advice": "Push for mutual indemnification and add a monetary cap tied to the total contract value.",
  "requires_human_review": false
}
```

Document-level summary:

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
  "executive_summary": "This contract carries a High risk rating driven by uncapped indemnification and one-sided liability caps...",
  "recommendations": [
    "Negotiate a monetary cap on indemnification obligations.",
    "Request mutual liability exclusions.",
    "Extend the termination cure period from 3 to 15 business days."
  ]
}
```

---

## 🔬 Phase 1 — Methodology

### Dataset

**CUAD** (Contract Understanding Atticus Dataset) — 510 commercial contracts with expert-labelled clause annotations.

| Step | Detail |
|---|---|
| **Transformation** | Extracted answer spans → clause text; mapped clause types → 5 risk categories |
| **Dataset size** | ~3,700 labelled clauses |
| **Preprocessing** | Lowercase, remove special chars, normalise whitespace |
| **Feature engineering** | TF-IDF, `max_features=8000`, n-gram range `(1, 3)` |
| **Models compared** | Logistic Regression vs Decision Tree |
| **Final model** | Logistic Regression with class balancing |

### Performance

| Metric | Score |
|---|---|
| **Accuracy** | 96% |
| **Macro F1** | 0.96 |
| **Weighted F1** | 0.96 |

Performance is consistent across all 5 categories.

---

## 🆚 Phase 1 vs Phase 2

| Dimension | Phase 1 (TF-IDF + LR) | Phase 2 (Agentic LLM) |
|---|---|---|
| Semantic understanding | Token frequency | Deep contextual reasoning |
| Generalisation | Limited to training vocab | Zero-shot on any legal text |
| Interpretability | Feature weights | Plain-English explanation |
| Negotiation guidance | ❌ | ✅ Per-clause advice |
| Document synthesis | Count aggregation | Intelligent executive summary |
| Latency | < 100ms | 2–8s per clause (API call) |
| Cost | Free (local) | ~$0.001–0.01 per clause |
| Offline use | ✅ | ❌ (requires API key) |

---

## 🔧 Advanced Features

### RAG — Retrieval-Augmented Generation

`agents/rag.py` augments classification with a legal knowledge base:
- FAISS vector store seeded with curated risk rules and precedents
- Retrieved rules injected into classification prompt for grounding
- Reduces hallucination on edge-case clauses

Enable: already in `requirements.txt` — no extra install needed.

### Guardrails

`agents/guardrails.py` provides production safety checks:
- **Input guard** — rejects empty, too-short, too-long, or non-textual input
- **Output guard** — corrects invalid categories/severities; clamps confidence to [0, 1]
- **Human-review flag** — `requires_human_review: true` when confidence < 0.45

---

## 🛠️ Tech Stack

### Phase 2 (Agentic AI)
- **[LangGraph](https://github.com/langchain-ai/langgraph)** — stateful multi-agent pipeline with conditional routing
- **[LangChain](https://python.langchain.com/)** — prompt templates, LLM chains
- **[Groq](https://groq.com/)** — LLM backend (llama-3.3-70b-versatile, free tier)
- **[Streamlit](https://streamlit.io/)** — web UI

### Phase 1 (Classical ML)
- **Scikit-learn** — TF-IDF, Logistic Regression, Decision Tree
- **Pandas / NumPy** — data processing
- **Matplotlib / Seaborn** — visualisation
- **Joblib** — model serialisation

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## hosted link
https://clauseguard-yvrrjxkvepdbmdse5juzxc.streamlit.app/

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built as part of the **Gen AI Capstone Project**  
Phase 1: Classical NLP · Phase 2: Agentic AI with LangGraph + Groq

</div>
