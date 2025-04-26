# utils.py
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

def clean_text(text):
    """Clean and tokenize text."""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)