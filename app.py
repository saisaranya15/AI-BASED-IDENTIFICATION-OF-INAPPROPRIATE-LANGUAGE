import streamlit as st
import numpy as np
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Load stopwords and stemmer
stopword = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load trained model & vectorizer
clf = joblib.load("rf_model.pkl")  # Load pre-trained Random Forest model
tfidf = joblib.load("tfidf_vectorizer.pkl")  # Load pre-trained TF-IDF vectorizer

# Text Cleaning Function
def clean(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove new lines
    text = [word for word in text.split() if word not in stopword]  # Remove stopwords
    text = [stemmer.stem(word) for word in text]  # Apply stemming
    return ' '.join(text)

# Function to predict with confidence threshold
def predict_text(text, confidence_threshold=0.4):  
    text_cleaned = clean(text)  # Ensure test data is preprocessed
    test_vector = tfidf.transform([text_cleaned]).toarray()
    
    # Get predicted class and probability
    prediction = clf.predict(test_vector)[0]
    probabilities = clf.predict_proba(test_vector)[0]  # Get class probabilities

    # Get max probability
    max_prob = max(probabilities)
    
    # If the probability is too low, return "None or Undefined"
    if max_prob < confidence_threshold:
        return "None or Undefined"
    
    return prediction

# Streamlit UI
st.title("AI based identification of inappropriate language")
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

# Footer
#st.markdown("---")
#st.markdown("💡 **Developed by Saranya| Powered by Machine Learning 🚀")
