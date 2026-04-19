"""
ClauseGuard v2 — Agentic AI Streamlit Interface
=================================================
A fully redesigned front-end that drives the LangGraph pipeline
instead of calling the old TF-IDF/Logistic-Regression classifier.

Key UI improvements
-------------------
* Dark-mode glass-card design
* Colour-coded risk badges (severity + category)
* Expandable per-clause explanation panels
* Executive summary dashboard for full-document mode
* Progress bar with live clause counter
"""

from __future__ import annotations

import json
import os
import time

import streamlit as st
import pandas as pd

# ── Page config must come first ──────────────────────────────────────────────
st.set_page_config(
    page_title="ClauseGuard v2 — Agentic AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import the agent graph (deferred so Streamlit can finish page setup) ─────
from agents.graph import run_document, run_single_clause


# ─────────────────────────────────────────────────────────────────────────────
# CSS  — Dark glass-morphism design
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #111827 60%, #0a0d1a 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(12px);
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}

/* ── Risk severity badges ── */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-critical { background: #ff2d55; color: #fff; }
.badge-high     { background: #ff6b35; color: #fff; }
.badge-medium   { background: #ffd60a; color: #111; }
.badge-low      { background: #30d158; color: #111; }
.badge-standard { background: #636366; color: #fff; }

/* ── Category pill ── */
.cat-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 6px;
    border: 1px solid rgba(255,255,255,0.15);
}
.cat-indemnity   { background: rgba(255,45,85,0.2);  color: #ff6b8a; }
.cat-liability   { background: rgba(255,107,53,0.2); color: #ffa07a; }
.cat-termination { background: rgba(255,214,10,0.2); color: #ffd60a; }
.cat-payment     { background: rgba(0,122,255,0.2);  color: #64b5f6; }
.cat-standard    { background: rgba(99,99,102,0.2);  color: #aaa; }

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6c63ff, #00d2ff, #6c63ff);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer {
    0%   { background-position: 0% }
    100% { background-position: 200% }
}
.hero-header p {
    color: rgba(255,255,255,0.55);
    font-size: 1.05rem;
    margin-top: 0.4rem;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    min-width: 130px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-card .lbl {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.5);
    margin-top: 2px;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Tables ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Text areas / inputs ── */
textarea, input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #f0f0f0 !important;
    border-radius: 10px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.8rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(108,99,255,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_CSS = {
    "Indemnity Risk":   "cat-indemnity",
    "Liability Risk":   "cat-liability",
    "Termination Risk": "cat-termination",
    "Payment Risk":     "cat-payment",
    "Standard Clause":  "cat-standard",
}

SEVERITY_CSS = {
    "Critical": "badge-critical",
    "High":     "badge-high",
    "Medium":   "badge-medium",
    "Low":      "badge-low",
}

SEVERITY_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


def _badge(text: str, css_class: str) -> str:
    return f'<span class="badge {css_class}">{text}</span>'


def _cat_pill(cat: str) -> str:
    css = CATEGORY_CSS.get(cat, "cat-standard")
    return f'<span class="cat-pill {css}">{cat}</span>'


def _severity_badge(sev: str) -> str:
    css = SEVERITY_CSS.get(sev, "badge-low")
    return _badge(sev, css)


def _confidence_bar(conf: float) -> str:
    pct = int(conf * 100)
    colour = "#ff2d55" if pct >= 80 else "#ffd60a" if pct >= 50 else "#30d158"
    return (
        f'<div style="background:rgba(255,255,255,0.08);border-radius:8px;height:8px;">'
        f'<div style="width:{pct}%;background:{colour};border-radius:8px;height:100%;'
        f'transition:width 0.6s ease;"></div></div>'
        f'<span style="font-size:0.78rem;color:rgba(255,255,255,0.5);">{pct}% confidence</span>'
    )


def _api_key_warning():
    """Show a helpful warning if no API keys are configured."""
    groq   = os.getenv("GROQ_API_KEY", "")
    xai    = os.getenv("XAI_API_KEY", "")
    openai = os.getenv("OPENAI_API_KEY", "")
    if not groq and not xai and not openai:
        st.warning(
            "⚠️  No LLM API key detected.  "
            "Set **GROQ_API_KEY** (free at console.groq.com), **XAI_API_KEY** (for Grok), "
            "or **OPENAI_API_KEY** (for GPT-4o) in your .env file.",
            icon="🔑",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown("**LLM Backend**")
    llm_choice = st.selectbox(
        "Provider",
        ["Grok (xAI) — preferred", "OpenAI GPT-4o — fallback"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Analysis Mode**")
    analysis_mode = st.radio(
        "Mode",
        ["Single Clause", "Full Document"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Risk Threshold**")
    risk_threshold = st.slider(
        "Min confidence to flag",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="Clauses below this confidence will still be shown but not counted as flagged.",
    )

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "<span style='color:rgba(255,255,255,0.4);font-size:0.82rem;'>"
        "ClauseGuard v2 · Agentic AI · LangGraph + Grok<br>"
        "Powered by 4 specialised LLM agents:<br>"
        "🔍 Extraction · ⚖️ Classification · 💬 Explanation · 📊 Summary"
        "</span>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <h1>⚖️ ClauseGuard v2</h1>
  <p>Agentic AI Contract Risk Analysis · Powered by LangGraph + Grok LLM</p>
</div>
""", unsafe_allow_html=True)

_api_key_warning()

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render a single clause result card
# ─────────────────────────────────────────────────────────────────────────────

def render_clause_card(result: dict, idx: int = 0):
    cat   = result.get("risk_category", "Standard Clause")
    sev   = result.get("severity", "Low")
    conf  = float(result.get("confidence", 0.0))
    clause = result.get("clause", "")
    explanation = result.get("explanation", "")
    terms = result.get("key_risk_terms", [])
    advice = result.get("negotiation_advice", "")
    bearer = result.get("risk_bearer", "N/A")

    with st.expander(
        f"{'🔴' if sev in ('High','Critical') else '🟡' if sev=='Medium' else '🟢'} "
        f"  Clause {idx+1}  ·  {cat}  ·  {sev}",
        expanded=(sev in ("High", "Critical")),
    ):
        # Category + severity badges
        st.markdown(
            f"{_cat_pill(cat)}  {_severity_badge(sev)}",
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Confidence bar
        st.markdown(_confidence_bar(conf), unsafe_allow_html=True)
        st.markdown("")

        # Clause text
        st.markdown(
            f'<div class="glass-card" style="border-left:3px solid #6c63ff;">'
            f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.45);">CLAUSE TEXT</span><br>'
            f'<span style="font-size:0.9rem;line-height:1.6;">{clause}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            if explanation:
                st.markdown(
                    f'<div class="glass-card">'
                    f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.45);">💬 PLAIN-ENGLISH EXPLANATION</span><br>'
                    f'<span style="font-size:0.9rem;line-height:1.6;">{explanation}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if advice:
                st.markdown(
                    f'<div class="glass-card" style="border-left:3px solid #ffd60a;">'
                    f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.45);">💡 NEGOTIATION ADVICE</span><br>'
                    f'<span style="font-size:0.9rem;">{advice}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with col2:
            if terms:
                terms_html = "".join(
                    f'<span style="display:inline-block;background:rgba(108,99,255,0.2);'
                    f'color:#a78bfa;border-radius:6px;padding:2px 9px;margin:2px;font-size:0.78rem;">'
                    f'{t}</span>'
                    for t in terms
                )
                st.markdown(
                    f'<div class="glass-card">'
                    f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.45);">🔑 KEY RISK TERMS</span><br><br>'
                    f'{terms_html}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="glass-card">'
                f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.45);">👤 RISK BEARER</span><br>'
                f'<span style="font-size:1rem;font-weight:600;color:#00d2ff;">{bearer}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# MODE A: Single Clause
# ─────────────────────────────────────────────────────────────────────────────

if analysis_mode == "Single Clause":
    st.subheader("🔍 Single Clause Analysis")
    st.markdown(
        "<span style='color:rgba(255,255,255,0.5);'>Paste a clause below. "
        "The AI agents will classify, explain, and flag risks.</span>",
        unsafe_allow_html=True,
    )

    # ── Apply staged example text BEFORE the widget is instantiated ──────────
    # (Writing to session_state[key] after the widget renders raises an error.)
    if "_clause_to_load" in st.session_state:
        st.session_state["single_clause_input"] = st.session_state.pop("_clause_to_load")

    clause_input = st.text_area(
        "Clause Text",
        placeholder=(
            "e.g.  The Client shall indemnify and hold harmless the Service "
            "Provider from any claims, damages, or liabilities arising from "
            "the Client's use of the platform..."
        ),
        height=180,
        key="single_clause_input",
    )

    analyze_btn = st.button("⚖️  Analyse Clause", key="btn_single")

    if analyze_btn:
        if not clause_input.strip():
            st.warning("Please enter a clause before analysing.")
        else:
            with st.spinner("🤖 Agents are reasoning …"):
                t0 = time.time()
                try:
                    result = run_single_clause(clause_input.strip())
                    elapsed = round(time.time() - t0, 2)

                    st.success(f"✅ Analysis complete in {elapsed}s")
                    st.markdown("---")
                    render_clause_card(result, idx=0)

                    # Raw JSON toggle
                    with st.expander("🔧 Raw JSON Output"):
                        st.json(result)

                except Exception as exc:
                    st.error(f"❌ Analysis failed: {exc}")
                    st.exception(exc)

    # ── Example clauses ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Try an example:**")
    examples = {
        "Indemnity":   "The Vendor shall indemnify, defend, and hold harmless the Client, its officers, directors, and employees from and against any and all claims, damages, penalties, fines, costs, and expenses (including reasonable attorneys' fees) arising out of or relating to any breach of this Agreement by the Vendor.",
        "Liability Cap": "IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, REGARDLESS OF WHETHER SUCH DAMAGES WERE FORESEEABLE AND WHETHER OR NOT ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. EACH PARTY'S TOTAL CUMULATIVE LIABILITY SHALL NOT EXCEED THE FEES PAID IN THE SIX MONTHS PRECEDING THE CLAIM.",
        "Termination": "Either party may terminate this Agreement upon thirty (30) days' written notice. In the event of a material breach by either party, the non-breaching party may terminate immediately upon written notice if such breach remains uncured for ten (10) business days after written notice thereof.",
        "Payment":     "Client agrees to pay all invoices within thirty (30) days of receipt. Unpaid invoices shall accrue interest at a rate of 1.5% per month (or the maximum rate permitted by law, whichever is lower) from the due date until paid in full.",
        "Standard":    "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of law provisions.",
    }
    cols = st.columns(len(examples))
    for col, (label, text) in zip(cols, examples.items()):
        if col.button(label, key=f"ex_{label}"):
            # Store in a staging key; the NEXT render will copy it to the
            # widget key BEFORE the text_area is instantiated (see above).
            st.session_state["_clause_to_load"] = text
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MODE B: Full Document
# ─────────────────────────────────────────────────────────────────────────────

else:
    st.subheader("📄 Full Contract Document Analysis")
    st.markdown(
        "<span style='color:rgba(255,255,255,0.5);'>Upload a contract (.txt) "
        "or paste text. All clauses will be extracted and analysed automatically.</span>",
        unsafe_allow_html=True,
    )

    tab_upload, tab_paste = st.tabs(["📁 Upload File", "✏️ Paste Text"])

    document_text = ""

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload Contract (.txt)",
            type=["txt"],
            key="doc_upload",
        )
        if uploaded:
            document_text = uploaded.read().decode("utf-8")
            st.success(f"✅ Loaded: **{uploaded.name}** ({len(document_text):,} chars)")

    with tab_paste:
        pasted = st.text_area(
            "Paste Contract Text",
            height=250,
            placeholder="Paste your full contract here …",
            key="doc_paste",
        )
        if pasted.strip():
            document_text = pasted.strip()

    analyze_doc_btn = st.button("🔬 Analyse Full Contract", key="btn_doc")

    if analyze_doc_btn:
        if not document_text.strip():
            st.warning("Please provide contract text before analysing.")
        else:
            progress_bar = st.progress(0, text="Initialising agents …")
            status_text  = st.empty()

            with st.spinner("🤖 Running multi-agent pipeline …"):
                t0 = time.time()
                try:
                    progress_bar.progress(15, text="🔍 Clause Extraction Agent …")
                    status_text.markdown(
                        "_Agent 1 of 4: extracting clauses from document …_"
                    )

                    output = run_document(document_text.strip())
                    elapsed = round(time.time() - t0, 2)

                    progress_bar.progress(100, text="✅ Done")
                    status_text.empty()

                    results        = output.get("results", [])
                    doc_summary    = output.get("document_summary", {})
                    pipeline_error = output.get("error")

                    if pipeline_error:
                        st.error(f"Pipeline error: {pipeline_error}")

                    st.success(
                        f"✅ Analysed **{len(results)} clauses** in {elapsed}s"
                    )
                    st.markdown("---")

                    # ── Executive Summary Dashboard ───────────────────────────
                    if doc_summary and isinstance(doc_summary, dict):
                        st.markdown("### 📊 Executive Risk Dashboard")

                        overall_rating = doc_summary.get("overall_risk_rating", "N/A")
                        rating_colour  = {
                            "Critical": "#ff2d55", "High": "#ff6b35",
                            "Medium": "#ffd60a",   "Low": "#30d158",
                        }.get(overall_rating, "#aaa")

                        dist    = doc_summary.get("risk_distribution", {})
                        exec_s  = doc_summary.get("executive_summary", "")
                        recs    = doc_summary.get("recommendations", [])
                        top_cls = doc_summary.get("highest_risk_clauses", [])

                        # Overall rating
                        st.markdown(
                            f'<div class="glass-card" style="text-align:center;">'
                            f'<span style="font-size:0.85rem;color:rgba(255,255,255,0.4);">OVERALL CONTRACT RISK</span><br>'
                            f'<span style="font-size:2.5rem;font-weight:800;color:{rating_colour};">{overall_rating}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Metrics row
                        m_cols = st.columns(5)
                        cat_icons = {
                            "Indemnity Risk": "🛡️",
                            "Liability Risk": "⚡",
                            "Termination Risk": "🚫",
                            "Payment Risk": "💰",
                            "Standard Clause": "📋",
                        }
                        for col, (cat, count) in zip(m_cols, dist.items()):
                            col.markdown(
                                f'<div class="metric-card">'
                                f'<div class="val">{count}</div>'
                                f'<div class="lbl">{cat_icons.get(cat,"")} {cat}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        # Executive summary text
                        if exec_s:
                            st.markdown(
                                f'<div class="glass-card">'
                                f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.45);">📝 EXECUTIVE SUMMARY</span><br><br>'
                                f'<span style="line-height:1.7;">{exec_s}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        # Recommendations
                        if recs:
                            recs_html = "".join(f"<li style='margin:6px 0;'>{r}</li>" for r in recs)
                            st.markdown(
                                f'<div class="glass-card" style="border-left:3px solid #ffd60a;">'
                                f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.45);">💡 RECOMMENDATIONS</span>'
                                f'<ul style="margin-top:10px;padding-left:18px;line-height:1.7;">{recs_html}</ul>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown("---")

                    # ── Clause Table ─────────────────────────────────────────
                    st.markdown("### 📋 Clause-Level Results")

                    if results:
                        df_data = []
                        for r in results:
                            df_data.append({
                                "Preview": r["clause"][:120] + "…",
                                "Category": r["risk_category"],
                                "Severity": r["severity"],
                                "Confidence": f"{int(r['confidence']*100)}%",
                                "Key Terms": ", ".join(r.get("key_risk_terms", [])[:3]),
                            })

                        df = pd.DataFrame(df_data)
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        # Sorting options
                        sort_by = st.selectbox(
                            "Sort clauses by",
                            ["Severity (High→Low)", "Category", "Confidence (High→Low)"],
                            key="sort_sel",
                        )
                        if sort_by == "Severity (High→Low)":
                            results_sorted = sorted(
                                results,
                                key=lambda r: SEVERITY_ORDER.get(r.get("severity", "Low"), 0),
                                reverse=True,
                            )
                        elif sort_by == "Category":
                            results_sorted = sorted(results, key=lambda r: r.get("risk_category", ""))
                        else:
                            results_sorted = sorted(results, key=lambda r: r.get("confidence", 0), reverse=True)

                        st.markdown("---")
                        st.markdown("### 🔍 Detailed Clause Analysis")

                        only_risky = st.checkbox(
                            "Show only risky clauses (exclude Standard Clauses)",
                            value=False,
                        )

                        filtered = [
                            r for r in results_sorted
                            if not only_risky or r.get("risk_category") != "Standard Clause"
                        ]

                        for i, result in enumerate(filtered):
                            render_clause_card(result, idx=i)

                    # Raw JSON download
                    st.markdown("---")
                    st.markdown("### 📥 Export")
                    json_bytes = json.dumps(
                        {"results": results, "summary": doc_summary},
                        indent=2,
                        default=str,
                    ).encode()
                    st.download_button(
                        label="⬇️  Download Full JSON Report",
                        data=json_bytes,
                        file_name="clauseguard_report.json",
                        mime="application/json",
                    )

                except Exception as exc:
                    progress_bar.progress(0)
                    st.error(f"❌ Analysis failed: {exc}")
                    st.exception(exc)
