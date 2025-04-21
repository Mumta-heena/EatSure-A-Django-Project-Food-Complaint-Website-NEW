import pandas as pd
import re
import numpy as np
import nltk
nltk.data.path.append('D:/Soft Dev LAB/.venv/nltk_data')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
df_fake = pd.read_csv("restaurant_reviews_anonymized.csv", encoding='latin1')

# Check columns and missing values
print(df_fake.columns)
print(df_fake.isnull().sum())

# Subset relevant columns
df_fake = df_fake[['Review', 'Real']]

# Fill missing reviews BEFORE cleaning
df_fake['Review'] = df_fake['Review'].fillna('')

def clean_text(text):
    # Remove URLs, special characters, etc.
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    # Tokenize with explicit language
    tokens = word_tokenize(text, language='english')
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Create cleaned_review column
df_fake['cleaned_review'] = df_fake['Review'].apply(clean_text)

# Define features (x) and labels (y)
x = df_fake['cleaned_review']  # This is a Series, not a DataFrame
y = df_fake['Real']

# Save preprocessed data
df_fake.to_csv("preprocessed_fake_reviews.csv", index=False)

# TF-IDF transformation (use x directly)
tfidf = TfidfVectorizer(max_features=1000)
x_text = tfidf.fit_transform(x)  # No need for x['cleaned_review']