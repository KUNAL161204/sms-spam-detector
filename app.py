# app.py
# Developed by: Khushi

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK packages are available on Streamlit Cloud
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    clean_tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    stemmed_tokens = [
        ps.stem(word) for word in clean_tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(stemmed_tokens)

# Load saved TF-IDF Vectorizer and Naive Bayes Model
@st.cache_resource
def load_artifacts():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_artifacts()

# Page Configuration
st.set_page_config(
    page_title="Khushi's SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Sidebar for Presentation Details
with st.sidebar:
    st.header("👩‍💻 Project Info")
    st.markdown("**Developer:** Khushi")
    st.markdown("**Model:** Multinomial Naive Bayes")
    st.markdown("**Feature Extraction:** TF-IDF (3,000 features)")
    st.markdown("**Dataset:** Kaggle UCI SMS Spam Collection")
    st.divider()
    st.caption("Built with Python, Scikit-Learn, NLTK & Streamlit")

# Main UI Header
st.title("📩 Message & SMS Spam Classifier")
st.markdown(
    "Welcome to **Khushi's Spam Detection System**. Enter an SMS or email message below "
    "to analyze whether it is legitimate (**Ham**) or unsolicited (**Spam**)."
)

# Quick Sample Buttons for Live Presentation Demo
st.subheader("Try a Sample Message:")
col1, col2 = st.columns(2)
if "input_sms" not in st.session_state:
    st.session_state.input_sms = ""

with col1:
    if st.button("🚨 Load Sample Spam"):
        st.session_state.input_sms = (
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! "
            "Txt the word: CLAIM to No: 81010 T&C www.dbuk.net"
        )
with col2:
    if st.button("✅ Load Sample Normal Message"):
        st.session_state.input_sms = (
            "Hey Khushi, are we still meeting at the library at 4 PM today to review the final presentation slides?"
        )

# Input Text Area
input_sms = st.text_area(
    "Enter the message to classify:",
    value=st.session_state.input_sms,
    height=140,
    placeholder="Type or paste your SMS/email message here..."
)

# Prediction Button
if st.button("🔍 Predict Message Status", type="primary", use_container_width=True):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before clicking Predict.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict class & probability
        result = model.predict(vector_input)[0]
        probabilities = model.predict_proba(vector_input)[0]
        confidence = probabilities[result] * 100

        st.divider()
        if result == 1:
            st.error(f"🚨 **SPAM DETECTED!** (Confidence: {confidence:.2f}%)")
            st.markdown("This message exhibits patterns typical of promotional, phishing, or fraudulent spam.")
        else:
            st.success(f"✅ **NOT SPAM (Legitimate / Ham)** (Confidence: {confidence:.2f}%)")
            st.markdown("This message looks safe and conversational.")

        # Expandable section to show evaluators how NLP processed the text
        with st.expander("🛠️ View Internal NLP Preprocessing Details"):
            st.write("**Original Length:**", len(input_sms), "characters")
            st.write("**Processed & Stemmed Tokens:**")
            st.code(transformed_sms if transformed_sms else "[No significant tokens remaining]")