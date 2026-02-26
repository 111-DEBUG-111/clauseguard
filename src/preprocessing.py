import pandas as pd
import re


def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset(input_path):
    df = pd.read_csv(input_path)

    print("Original dataset size:", len(df))

    df["cleaned_text"] = df["clause_text"].apply(clean_text)

    print("Preprocessing complete.")

    return df