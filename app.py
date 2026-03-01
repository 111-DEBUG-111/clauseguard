import streamlit as st
import pandas as pd
import re
from src.predict import predict_clause

st.set_page_config(page_title="ClauseGuard", layout="wide")

st.title("ClauseGuard — Contract Risk Classification System")

st.write("Upload a contract (.txt) or paste a clause for risk analysis.")

# =============================
# Clause Segmentation Function
# =============================

def segment_clauses(text):
    clauses = re.split(r'\n\d+\.|\n[A-Z][a-z]+:|\n\n', text)
    clauses = [c.strip() for c in clauses if len(c.strip()) > 50]
    return clauses

# =============================
# Option 1: Single Clause
# =============================

st.subheader("Single Clause Analysis")

clause_input = st.text_area("Enter Clause Text", height=150)

if st.button("Analyze Clause"):
    if clause_input.strip():
        label, probs = predict_clause(clause_input)

        st.write(f"**Predicted Category:** {label}")
        st.write("**Probability Distribution:**")
        st.json(probs)
    else:
        st.warning("Please enter a clause.")

#st.divider()
st.markdown("---")

# =============================
# Option 2: Full Contract Upload
# =============================

st.subheader("Full Contract Analysis (.txt file)")

uploaded_file = st.file_uploader("Upload Contract", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    clauses = segment_clauses(text)

    results = []

    for clause in clauses:
        label, probs = predict_clause(clause)
        confidence = max(probs.values())

        results.append({
            "Clause Preview": clause[:120] + "...",
            "Predicted Risk": label,
            "Confidence": confidence
        })

    df_results = pd.DataFrame(results)

    st.write("### Clause-Level Risk Analysis")
    st.dataframe(df_results)

    high_risk_count = df_results[df_results["Predicted Risk"].isin(
        ["Liability Risk", "Termination Risk"]
    )].shape[0]

    st.write(f"Total Clauses: {len(df_results)}")
    st.write(f"High Risk Clauses Detected: {high_risk_count}")
