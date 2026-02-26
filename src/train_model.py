import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import preprocess_dataset

# ============================
# CONFIG
# ============================

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
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# TF-IDF FEATURE ENGINEERING
# ============================

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ============================
# MODEL 1 — LOGISTIC REGRESSION
# ============================

lr_model = LogisticRegression(
    max_iter=300,
    class_weight="balanced"
)

lr_model.fit(X_train_tfidf, y_train)

lr_preds = lr_model.predict(X_test_tfidf)

print("\n===== Logistic Regression =====\n")
print(classification_report(y_test, lr_preds))

# ============================
# CONFUSION MATRIX (PHASE 2)
# ============================

cm = confusion_matrix(y_test, lr_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=lr_model.classes_,
    yticklabels=lr_model.classes_
)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.close()

print("\nConfusion matrix saved to models/confusion_matrix.png")

# ============================
# MODEL 2 — DECISION TREE
# ============================

dt_model = DecisionTreeClassifier(max_depth=20)

dt_model.fit(X_train_tfidf, y_train)

dt_preds = dt_model.predict(X_test_tfidf)

print("\n===== Decision Tree =====\n")
print(classification_report(y_test, dt_preds))

# ============================
# FEATURE IMPORTANCE (PHASE 3)
# ============================

feature_names = vectorizer.get_feature_names_out()
coefficients = lr_model.coef_

print("\n===== Top Features Per Class =====\n")

for i, class_label in enumerate(lr_model.classes_):
    top_indices = coefficients[i].argsort()[-10:]
    print(f"\nTop features for {class_label}:")
    for idx in reversed(top_indices):
        print(feature_names[idx])

# ============================
# SAVE FINAL MODEL
# ============================

joblib.dump(lr_model, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")