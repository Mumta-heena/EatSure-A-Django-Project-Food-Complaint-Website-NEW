# predict.py
from model_utils import load_model
from utils import clean_text

def predict_review(text, model_path="fake_review_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
    """Predict if a review is fake."""
    model, tfidf = load_model(model_path, vectorizer_path)
    cleaned_text = clean_text(text)
    features = tfidf.transform([cleaned_text])
    proba = model.predict_proba(features)[0]
    return {
        "prediction": "Fake" if model.predict(features)[0] == 1 else "Real",
        "confidence": {"real": round(proba[0], 2), "fake": round(proba[1], 2)}
    }

if __name__ == "__main__":
    test_review = "This place is terrible! The food was cold and the staff was rude."
    print(predict_review(test_review))