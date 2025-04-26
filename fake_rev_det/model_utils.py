# model_utils.py
import joblib

def save_model(model, vectorizer, model_path, vectorizer_path):
    """Save model and vectorizer to disk."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path} and vectorizer to {vectorizer_path}.")

def load_model(model_path, vectorizer_path):
    """Load model and vectorizer from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer