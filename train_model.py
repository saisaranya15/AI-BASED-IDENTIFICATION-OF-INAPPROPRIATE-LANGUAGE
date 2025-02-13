import pandas as pd
import numpy as np
import re
import string
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

# Load dataset
df = pd.read_csv("labeled_data.csv")  # Replace with your dataset

# Label mapping
df['labels'] = df['class'].map({
    0: 'Hate Speech Detected',
    1: 'Offensive Language detected',
    2: 'No hate or Offensive speech detected'
})
df = df.dropna(subset=['labels'])

# Text Cleaning Function
def clean(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = [stemmer.stem(word) for word in text]
    return ' '.join(text)

# Apply text cleaning
df['tweet'] = df['tweet'].apply(clean)

# Features and Labels
x = np.array(df['tweet'])
y = np.array(df['labels'])

# Vectorization using TF-IDF
tfidf = TfidfVectorizer()
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
joblib.dump(clf, 'rf_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("✅ Model & Vectorizer Saved Successfully!")
