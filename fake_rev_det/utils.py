import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_text(text):
    """
    Cleans text by removing URLs, special characters, and stopwords.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs and non-alphabetic characters
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text, language='english')
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)