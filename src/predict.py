import joblib
import re
import numpy as np

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_clause(clause_text):
    cleaned = clean_text(clause_text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    class_probs = dict(zip(model.classes_, np.round(probabilities, 3)))

    return prediction, class_probs