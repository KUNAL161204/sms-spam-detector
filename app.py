# app.py (Upgraded: Random Sample Loader + Modern Link/Money Spam Detection)
# Developed by: Khushi

import os
import pickle
import random
import re
import string
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Page Configuration
st.set_page_config(
    page_title="Khushi's SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# 2. Download Required NLTK Data
@st.cache_resource
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    # Keep action words often used in phishing links
    for word in ['below', 'above', 'here', 'now', 'only']:
        stop_words.discard(word)
    return ps, stop_words

ps, stop_words = setup_nltk()

# 3. Text Preprocessing Function
def transform_text(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    clean_tokens = [word for word in tokens if word.isalnum()]
    stemmed_tokens = [
        ps.stem(word) for word in clean_tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(stemmed_tokens)

# 4. Load Dataset, Augment with Modern Link Spam, Split & Train
@st.cache_resource
def train_model_on_dataset():
    if not os.path.exists('spam.csv'):
        return None

    # Load Kaggle dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    df.dropna(inplace=True)
    df.drop_duplicates(keep='first', inplace=True)

    # Split Kaggle data into 80% Training and 20% Testing
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['target']
    )

    # Augment ONLY the Training Set with modern link/money/phishing spam patterns
    # so the model recognizes modern scams like "click on the below link to get money"
    modern_spam_samples = [
        "Click on the below link to get money instantly in your bank account",
        "Click the link below to claim your free cash reward and money prize",
        "Tap on the link below to get free money and cashback offer now",
        "Earn money fast! Click on the link below to receive your instant payout",
        "Congratulations! Click below link to get money transferred to your wallet",
        "Urgent: Your bank account KYC is expiring. Click the link below to verify now",
        "Click here to get free money and win a guaranteed cash bonus today",
        "Follow the link below to claim money and lottery winnings immediately",
        "Get instant money by clicking on the below link right now",
        "Verify your account by clicking the link below to unlock your money"
    ] * 8  # Weighted so TF-IDF and Naive Bayes learn modern link+money patterns

    modern_spam_df = pd.DataFrame({
        'target': [1] * len(modern_spam_samples),
        'text': modern_spam_samples
    })

    augmented_train_df = pd.concat([train_df, modern_spam_df], ignore_index=True)

    # Preprocess Training and Testing sets
    augmented_train_df['transformed_text'] = augmented_train_df['text'].apply(transform_text)
    test_df = test_df.copy()
    test_df['transformed_text'] = test_df['text'].apply(transform_text)

    # Feature Extraction using Unigrams + Bigrams (1, 2) to capture phrases like "click link" & "get money"
    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(augmented_train_df['transformed_text']).toarray()
    X_test = tfidf.transform(test_df['transformed_text']).toarray()

    y_train = augmented_train_df['target'].values
    y_test = test_df['target'].values

    # Train Multinomial Naive Bayes with balanced priors so 87% Ham doesn't overpower short spam
    model = MultinomialNB(alpha=0.3, class_prior=[0.55, 0.45])
    model.fit(X_train, y_train)

    # Evaluate on the pure Kaggle 20% Test Set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    # Prepare pools of diverse Spam and Ham messages from the unseen Test Set for the UI buttons
    spam_pool = test_df[test_df['target'] == 1]['text'].tolist() + [
        "Click on the below link to get money directly in your account: http://bit.ly/free-cash",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt CLAIM to 81010",
        "Congratulations! Click the link below to claim your ₹50,000 lottery reward now."
    ]
    ham_pool = [
        "Hey Khushi, are we still meeting at the library at 4 PM today to review the final presentation slides?",
        "Khushi, please send me the Python project report once you reach home.",
        "Can you call me when you are free? We need to discuss tomorrow's exam schedule.",
        "Mom said dinner will be ready by 8 PM, don't forget to bring milk on your way back.",
        "Good morning Khushi! Best of luck for your final project presentation today."
    ] + test_df[test_df['target'] == 0]['text'].head(30).tolist()

    return {
        "tfidf": tfidf,
        "model": model,
        "total_rows": len(df),
        "train_df": train_df,
        "test_df": test_df,
        "accuracy": acc,
        "precision": prec,
        "spam_pool": spam_pool,
        "ham_pool": ham_pool
    }

with st.spinner("Training model on Kaggle SMS Spam Dataset (spam.csv)..."):
    pipeline_data = train_model_on_dataset()

if pipeline_data is None:
    st.error("⚠️ `spam.csv` was not found! Please make sure `spam.csv` is uploaded to your GitHub repository alongside `app.py`.")
    st.stop()

tfidf = pipeline_data["tfidf"]
model = pipeline_data["model"]

# Sidebar Info
with st.sidebar:
    st.header("👩‍💻 Project Info")
    st.markdown("**Developer:** Khushi")
    st.markdown("**Algorithm:** Multinomial Naive Bayes (Bigrams)")
    st.markdown("**Vectorizer:** TF-IDF (Unigrams + Bigrams)")
    st.markdown("**Dataset:** Kaggle SMS Spam Collection")
    st.divider()
    st.success("✅ Model Trained on `spam.csv`")
    st.metric("Test Set Accuracy", f"{pipeline_data['accuracy'] * 100:.2f}%")
    st.metric("Test Set Precision", f"{pipeline_data['precision'] * 100:.2f}%")

# Main Heading
st.title("📩 Message & SMS Spam Classifier")
st.markdown(
    "Welcome to **Khushi's Spam Detection System**. This application trains on the "
    "Kaggle SMS Spam Dataset (`spam.csv`) using an **80% Training / 20% Testing split** and Bigram NLP features."
)

# Expandable Proof of Training & Testing for Presentation
with st.expander("📊 View Dataset Training & Testing Split Details (Presentation Proof)"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Unique Messages", pipeline_data["total_rows"])
    m2.metric("Training Set (80%)", len(pipeline_data["train_df"]))
    m3.metric("Testing Set (20%)", len(pipeline_data["test_df"]))

    st.markdown("**1. Sample of Training Data (80% used by `model.fit()`):**")
    st.dataframe(pipeline_data["train_df"][['target', 'text']].head(5), use_container_width=True)

    st.markdown("**2. Sample of Unseen Testing Data (20% used by `model.predict()`):**")
    st.dataframe(pipeline_data["test_df"][['target', 'text']].head(5), use_container_width=True)

st.divider()

# Initialize session state for text box
if "input_sms" not in st.session_state:
    st.session_state.input_sms = ""

# Dynamic Random Sample Buttons (Loads a DIFFERENT message on every click)
st.subheader("Try a Sample Message (Click multiple times for new messages):")
col1, col2 = st.columns(2)

with col1:
    if st.button("🚨 Load Random Spam Message", use_container_width=True):
        st.session_state.input_sms = random.choice(pipeline_data["spam_pool"])

with col2:
    if st.button("✅ Load Random Normal Message", use_container_width=True):
        st.session_state.input_sms = random.choice(pipeline_data["ham_pool"])

# User Input Box (Bound via key="input_sms" so button clicks always update the box)
input_sms = st.text_area(
    "Enter the message to classify:",
    key="input_sms",
    height=130,
    placeholder="Type or paste any SMS/email message here..."
)

# Predict Button
if st.button("🔍 Predict Message Status", type="primary", use_container_width=True):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before clicking Predict.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        
        # Get Naive Bayes probabilities [Ham_Prob, Spam_Prob]
        probabilities = model.predict_proba(vector_input)[0]
        spam_prob = probabilities[1]

        # Check for high-risk modern phishing combinations (e.g., link + money/claim/verify)
        lower_raw = input_sms.lower()
        has_link_trigger = any(w in lower_raw for w in ["link", "click", "http", "www.", "bit.ly"])
        has_reward_trigger = any(w in lower_raw for w in ["money", "cash", "claim", "prize", "won", "urgent", "kyc", "verify", "bank", "free"])
        if has_link_trigger and has_reward_trigger:
            spam_prob = max(spam_prob, 0.92)

        result = 1 if spam_prob >= 0.50 else 0
        confidence = (spam_prob if result == 1 else (1 - spam_prob)) * 100

        st.divider()
        if result == 1:
            st.error(f"🚨 **SPAM DETECTED!** (Confidence: {confidence:.2f}%)")
            st.markdown("This message exhibits patterns of promotional, phishing, or fraudulent link spam.")
        else:
            st.success(f"✅ **NOT SPAM (Legitimate / Ham)** (Confidence: {confidence:.2f}%)")
            st.markdown("This message looks safe and conversational.")

        with st.expander("🛠️ View Internal NLP Preprocessing Steps"):
            st.write("**Original Message:**", input_sms)
            st.write("**Cleaned & Stemmed Tokens Sent to Model:**")
            st.code(transformed_sms if transformed_sms else "[No tokens remaining]")
