import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from model_utils import save_model

def train_and_evaluate():
    # Verify file exists
    if not os.path.exists("combined_data.csv"):
        raise FileNotFoundError("Run preprocessing.py first!")
    
    # Load data with additional validation
    data = pd.read_csv("D:/Soft Dev LAB/fake_rev_det/combined_data.csv")
    
    # Final data sanitization
    data = data.dropna(subset=["cleaned_review"])
    data = data[data["cleaned_review"].str.strip() != ""]
    data["cleaned_review"] = data["cleaned_review"].astype(str)
    
    # Handle empty dataset edge case
    if len(data) == 0:
        raise ValueError("No valid data remaining after preprocessing!")
    
    # TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    
    try:
        X = tfidf.fit_transform(data["cleaned_review"])
    except ValueError as e:
        print("Error in vectorization:")
        print("Sample problematic reviews:")
        print(data[data["cleaned_review"].str.len() < 3][["cleaned_review"]].head())
        raise
    
    y = data["label"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Train model
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver='liblinear'  # Better for small-medium datasets
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    save_model(model, tfidf, "fake_review_model.pkl", "tfidf_vectorizer.pkl")

if __name__ == "__main__":
    train_and_evaluate()