import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Download stopwords
nltk.download('stopwords')

# Load stopwords and stemmer
stopword = set(stopwords.words('english'))
stemmer = PorterStemmer()

# File names for model and vectorizer
MODEL_FILE = "rf_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# List of Explicit Hate Speech Words
HATE_WORDS = {"hate", "kill", "violence", "racist", "terrorist", "disgusting", "destroy"}

# List of Positive Words
POSITIVE_WORDS = {"good", "great", "excellent", "fantastic", "amazing", "love", "wonderful", "nice"}

# Function to clean text
def clean(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove new lines
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopword]  # Remove stopwords + stemming
    return ' '.join(words)

# Train model if not found
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    st.info("🔄 Training model, please wait...")

    # Load dataset
    df = pd.read_csv("labeled_data.csv")  # Ensure this file is in your repo

    # Label mapping
    df['labels'] = df['class'].map({
        0: 'Hate Speech Detected',
        1: 'Offensive Language Detected',
        2: 'No Hate or Offensive Speech Detected'
    })
    df = df.dropna(subset=['labels'])

    # Apply text cleaning
    df['tweet'] = df['tweet'].apply(clean)

    # Features and Labels
    x = np.array(df['tweet'])
    y = np.array(df['labels'])

    # Vectorization using TF-IDF (with bigrams for better phrase recognition)
    tfidf = TfidfVectorizer(ngram_range=(1,2))  # Captures both single words and two-word phrases
    X = tfidf.fit_transform(x)

    # Handle Class Imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.33, random_state=42)

    # Train Random Forest Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Save the trained model & TF-IDF vectorizer
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(tfidf, VECTORIZER_FILE)
    st.success("✅ Model trained and saved!")

# Load trained model & vectorizer
clf = joblib.load(MODEL_FILE)
tfidf = joblib.load(VECTORIZER_FILE)

# Function to predict with confidence threshold
def predict_text(text, confidence_threshold=0.4):  # Lowered threshold for better predictions
    text_cleaned = clean(text)  # Ensure test data is preprocessed
    test_vector = tfidf.transform([text_cleaned]).toarray()
    
    # Get predicted class and probability
    probabilities = clf.predict_proba(test_vector)[0]  # Get class probabilities
    max_prob = max(probabilities)
    prediction = clf.predict(test_vector)[0]

    # **Force Prediction for Specific Words**
    words = set(text_cleaned.split())
    if words & HATE_WORDS:
        return "Hate Speech Detected"
    if words & POSITIVE_WORDS:
        return "No Hate or Offensive Speech Detected"

    # Debugging Output
    print(f"Text: {text}, Probabilities: {probabilities}, Max Prob: {max_prob}, Prediction: {prediction}")

    # If the probability is too low, return "Neutral"
    if max_prob < confidence_threshold:
        return "Neutral"
    
    return prediction

# Streamlit UI
st.title("AI-Based Identification of Inappropriate Language")
st.write("Enter a text message below to detect hate speech or offensive content.")

# User Input
user_input = st.text_area("Enter text here:")

# Predict Button
if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter a valid text input!")
    else:
        result = predict_text(user_input)
        st.success(f"Prediction: **{result}**")
