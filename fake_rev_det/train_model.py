# train_model_optimized.py
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
from utils import clean_text  # Import the shared function
import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ===========================================
# 1. Define Global Variables (FIXED)
# ===========================================
text_feature = 'Review'
meta_features = [
    'F1-AWL', 'F2-PAU', 'F3-ANP', 'F4-ASL', 'F5-NCL', 
    'F6-NWO', 'F7-NVB', 'F8-NAJ', 'F9-NPV', 'F10-EMO',
    'F11-CDV', 'F12-RED', 'F13-LXD', 'F14-NMV', 'F15-NTY'
]

# ===========================================
# 2. Load Data (FIXED: Use Global meta_features)
# ===========================================
def load_data():
    df = pd.read_csv("restaurant_reviews_anonymized.csv", encoding='latin1')
    
    # Select relevant columns using global variables
    df = df[[text_feature, 'Real'] + meta_features].copy()
    df[text_feature] = df[text_feature].fillna('')  # Handle missing text
    
    return df

# ===========================================
# 3. Text Cleaning
# ===========================================
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    tokens = word_tokenize(text, language='english')
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# ===========================================
# 4. Feature Pipeline (FIXED: Use Global meta_features)
# ===========================================
def create_feature_pipeline():
    # Text processing pipeline
    text_transformer = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english'
        ))
    ])
    
    # Numerical features pipeline
    meta_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('text', text_transformer, text_feature),  # Use text_feature
        ('meta', meta_transformer, meta_features)  # Use global meta_features
    ])
    
    return preprocessor

# ===========================================
# 5. Model Training
# ===========================================
def train_model(X, y):
    pipeline = Pipeline([
        ('preprocessor', create_feature_pipeline()),
        ('classifier', RandomForestClassifier(
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    params = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    return grid_search

# ===========================================
# 6. Main Workflow (FIXED: Properly format X)
# ===========================================
if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Clean text and retain original features
    df['cleaned_review'] = df[text_feature].apply(clean_text)
    
    # Prepare features and target
    X = df[[text_feature] + meta_features]  # Include original text + meta features
    y = df['Real']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Train model
    best_model = train_model(X_train, y_train)
    
    # Evaluate
    print(f"Best Parameters: {best_model.best_params_}")
    print(f"Best CV Accuracy: {best_model.best_score_:.2f}")
    
    test_acc = best_model.score(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.2f}")
    
    # Save model
    joblib.dump(best_model, 'optimized_model.pkl')
    print("\nOptimized model saved!")