# ClauseGuard — Contract Risk Classification System

##  Overview

ClauseGuard is a machine learning-based web application that analyzes legal contracts and identifies potentially risky clauses.

The system processes contract text, segments it into clauses, and classifies each clause into predefined legal risk categories such as:

- Liability Risk
- Termination Risk
- Payment Risk
- Indemnity Risk
- Standard Clauses

The goal is to simplify contract analysis and help users quickly understand legal risks.

---

## Live Demo

🔗 https://clauseguard-v1.streamlit.app/

---

##  Problem Statement

Legal contracts are often complex and difficult to understand, especially for non-experts. Important risk-related clauses are buried in long documents, making manual analysis time-consuming and error-prone.

ClauseGuard aims to automate this process by:

- Extracting clauses from contracts
- Classifying them into risk categories
- Providing quick insights into potential legal risks

---

## ⚙️ Methodology

### 1. Dataset

We used the CUAD (Contract Understanding Atticus Dataset), which is originally structured as a question-answering dataset.

### 2. Data Transformation

- Extracted answer spans as clause text
- Converted clause types into 5 risk categories
- Created a structured classification dataset (~3,700 clauses)

### 3. Preprocessing

- Lowercasing text
- Removing special characters
- Cleaning and normalization

### 4. Feature Engineering

- TF-IDF vectorization
- N-gram range: (1, 3)
- Captures legal phrases like:
  - "written notice"
  - "liquidated damages"

### 5. Model Training

Two models were compared:

- Logistic Regression (Final Model)
- Decision Tree

Final Model:
- Logistic Regression with class balancing
- Accuracy: ~96%

---

##  Model Performance

| Metric | Value |
|------|------|
| Accuracy | 96% |
| Macro F1 Score | 0.96 |

The model performs consistently across all risk categories.

---

## Features

###  Single Clause Analysis
- Paste any clause
- Get risk category + confidence score

###  Full Contract Analysis
- Upload `.txt` file
- Automatic clause segmentation
- Clause-wise classification
- Risk summary output

### Interpretability
- Probability distribution for predictions
- Top feature analysis for each class

---

##  Example Use Cases
- Contract review for students
- Legal risk identification for businesses
- Quick screening of agreements

---

## Tech Stack

- Python
- Scikit-learn
- Pandas
- Streamlit
- TF-IDF Vectorization


