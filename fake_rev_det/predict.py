# predict.py
import joblib
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK resources (if missing)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load the trained model and preprocessing pipeline
MODEL_PATH = 'optimized_model.pkl'

# Metadata features from your dataset (same order as training)
META_FEATURES = [
    'F1-AWL', 'F2-PAU', 'F3-ANP', 'F4-ASL', 'F5-NCL', 
    'F6-NWO', 'F7-NVB', 'F8-NAJ', 'F9-NPV', 'F10-EMO',
    'F11-CDV', 'F12-RED', 'F13-LXD', 'F14-NMV', 'F15-NTY'
]

def clean_text(text):
    """Identical cleaning function used during training"""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs and special characters
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text, language='english')
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def predict_review(review_text, metadata):
    """
    Predict if a review is fake (1) or real (0).
    
    Args:
        review_text (str): Raw review text
        metadata (dict): Dictionary containing values for all 15 metadata features
    
    Returns:
        dict: Prediction and confidence scores
    """
    try:
        # Load the trained pipeline
        model = joblib.load(MODEL_PATH)
        
        # Clean the text
        cleaned_text = clean_text(review_text)
        
        # Create input DataFrame with same structure as training data
        input_data = pd.DataFrame([{
            'Review': cleaned_text,
            **{feature: metadata[feature] for feature in META_FEATURES}
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        return {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': {
                'real': round(probabilities[0], 3),
                'fake': round(probabilities[1], 3)
            }
        }
        
    except Exception as e:
        return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    # Sample metadata (replace with actual values)
    sample_metadata = {
        'F1-AWL': 4.236363636, 'F2-PAU': 1.666666667, 'F3-ANP': 14.86666667,
        'F4-ASL': 12.22222222, 'F5-NCL': 9, 'F6-NWO': 110,
        'F7-NVB': 11, 'F8-NAJ': 11, 'F9-NPV': 1,
        'F10-EMO': 0.394736842, 'F11-CDV': 0.836363636, 'F12-RED': 4.444444444,
        'F13-LXD': 0.726315789, 'F14-NMV': 2, 'F15-NTY': 1
    }
    
    sample_review = "This restaurant was absolutely terrible! The food tasted like cardboard."
    
    result = predict_review(sample_review, sample_metadata)
    print("Prediction Result:")
    print(f"Classification: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")