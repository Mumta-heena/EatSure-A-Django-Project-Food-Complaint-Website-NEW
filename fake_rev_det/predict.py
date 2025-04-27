# predict.py
from model_utils import load_model
from utils import clean_text

def predict_review(text, model_path="D:/Soft Dev LAB/safe_eat/ML/fake_review_model.pkl", vectorizer_path="D:/Soft Dev LAB/safe_eat/ML/tfidf_vectorizer.pkl"):
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
    while True:
        review_text = input("Enter a review (or type 'exit' to quit): ")
        if review_text.lower() == "exit":
            print("Exiting...")
            break

        result = predict_review(review_text)
        print("\nPrediction:", result["prediction"])
        print("Confidence:")
        print(f"  Real: {result['confidence']['real']}")
        print(f"  Fake: {result['confidence']['fake']}")
