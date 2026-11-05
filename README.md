# 📩 SMS & Message Spam Detection System

**Author:** Khushi  
**Framework:** Python, Scikit-Learn, NLTK, Streamlit  
**Dataset:** [UCI SMS Spam Collection Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 📌 Project Overview
Unsolicited spam messages pose both a security risk (phishing, financial scams) and a daily nuisance. This project implements an end-to-end **Natural Language Processing (NLP)** and **Machine Learning** pipeline that classifies incoming SMS or text messages as either **Legitimate (Ham)** or **Spam** in real time.

The trained model achieves **~97.5% accuracy** and **~100% precision** on unseen test data and is deployed as an interactive web application using **Streamlit**.

---

## ⚙️ Workflow & Architecture
1. **Data Collection:** Loaded 5,572 labeled SMS messages from the Kaggle SMS Spam Collection Dataset.
2. **Data Cleaning:** Removed empty metadata columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`), encoded labels (`ham` = 0, `spam` = 1), and eliminated 403 duplicate records.
3. **NLP Text Preprocessing:**
   - Converted all text to lowercase.
   - Tokenized sentences into individual words using `nltk.word_tokenize()`.
   - Filtered out special characters, punctuation, and English stopwords.
   - Reduced words to their root stems using `PorterStemmer()` (e.g., *"winning"*, *"winner"* → *"win"*).
4. **Feature Engineering:** Converted cleaned text into numerical vectors using **TF-IDF Vectorizer** (`max_features=3000`).
5. **Model Training & Evaluation:** Trained a **Multinomial Naive Bayes** classifier using an 80/20 stratified train-test split, prioritizing high **Precision** so legitimate messages are not falsely flagged as spam.
6. **Web Deployment:** Serialized the model and vectorizer with `pickle` and built a responsive frontend using **Streamlit**.

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/khushi/khushi-sms-spam-detector.git](https://github.com/khushi/khushi-sms-spam-detector.git)
cd khushi-sms-spam-detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional if `.pkl` files are already present)
```bash
python train_model.py
```

### 4. Launch the Streamlit Web App
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Model Performance
| Metric | Score | Why It Matters |
| :--- | :--- | :--- |
| **Accuracy** | ~97.5% | Overall percentage of correct predictions across both classes |
| **Precision** | ~99.0% - 100% | Ensures normal ("Ham") messages are almost never mistakenly marked as Spam |

---

## 👩‍💻 Developed By
**Khushi**
*Machine Learning & NLP Final Presentation Project*