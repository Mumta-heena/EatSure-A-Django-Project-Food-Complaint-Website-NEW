import os
from ..fake_rev_det import model_utils
from ..fake_rev_det import utils

# Build correct paths to the model and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'D:\Soft Dev LAB\safe_eatML/fake_review_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

# Load model and vectorizer ONCE
model, tfidf = model_utils.load_model(model_path, vectorizer_path)

def predict_review(text):
    """Predict if a review is fake or real."""
    cleaned_text = utils.clean_text(text)
    features = tfidf.transform([cleaned_text])
    proba = model.predict_proba(features)[0]
    return {
        "prediction": "Fake" if model.predict(features)[0] == 1 else "Real",
        "confidence": {"real": round(proba[0], 2), "fake": round(proba[1], 2)}
    }
