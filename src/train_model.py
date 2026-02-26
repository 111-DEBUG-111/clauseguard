import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from preprocessing import preprocess_dataset

DATA_PATH = "data/processed/cuad_processed.csv"

# ============================
# LOAD + PREPROCESS
# ============================

df = preprocess_dataset(DATA_PATH)

X = df["cleaned_text"]
y = df["risk_category"]

# ============================
# TRAIN TEST SPLIT
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# TF-IDF
# ============================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ============================
# MODEL 1: Logistic Regression
# ============================

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_tfidf, y_train)

lr_preds = lr_model.predict(X_test_tfidf)

print("\n===== Logistic Regression =====\n")
print(classification_report(y_test, lr_preds))

# ============================
# MODEL 2: Decision Tree
# ============================

dt_model = DecisionTreeClassifier(max_depth=20)
dt_model.fit(X_train_tfidf, y_train)

dt_preds = dt_model.predict(X_test_tfidf)

print("\n===== Decision Tree =====\n")
print(classification_report(y_test, dt_preds))

# ============================
# SAVE BEST MODEL (LR)
# ============================

joblib.dump(lr_model, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved.")