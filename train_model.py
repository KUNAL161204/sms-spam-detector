# train_model.py
# Author: Khushi

import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    """
    Preprocesses message text:
    1. Lowercases text
    2. Tokenizes into words
    3. Keeps only alphanumeric tokens
    4. Removes stopwords and punctuation
    5. Applies Porter Stemming
    """
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    # Keep only alphanumeric words
    clean_tokens = [word for word in tokens if word.isalnum()]
    
    # Remove stopwords and punctuation, then stem
    stemmed_tokens = [
        ps.stem(word) for word in clean_tokens 
        if word not in stop_words and word not in string.punctuation
    ]
    
    return " ".join(stemmed_tokens)

# 1. Load the Kaggle dataset (latin-1 encoding handles special SMS characters)
print("Loading dataset...")
df = pd.read_csv('spam.csv', encoding='latin-1')

# 2. Data Cleaning: Drop unnecessary columns and rename v1 & v2
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode target labels: ham -> 0, spam -> 1
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Drop missing and duplicate rows
df.dropna(inplace=True)
df.drop_duplicates(keep='first', inplace=True)
print(f"Dataset cleaned. Total unique messages: {len(df)}")

# 3. Text Preprocessing
print("Preprocessing text messages...")
df['transformed_text'] = df['text'].apply(transform_text)

# 4. Feature Extraction using TF-IDF (top 3000 features)
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# 5. Split into Training (80%) and Testing (20%) sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train the Multinomial Naive Bayes Model
print("Training Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("\n--- Model Evaluation Results (Khushi's Spam Classifier) ---")
print(f"Accuracy Score  : {accuracy * 100:.2f}%")
print(f"Precision Score : {precision * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))

# 8. Save the Vectorizer and Model for Streamlit Deployment
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
print("Successfully saved 'vectorizer.pkl' and 'model.pkl'!")