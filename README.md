# 📱 Sentiment Analysis of M-Pajak Mobile Application Reviews

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)

## 📌 Project Overview

This project analyzes user sentiment toward the **M-Pajak mobile application** using Natural Language Processing (NLP) and Machine Learning.

The objective is to classify user reviews into:

- 🔴 Negative
- ⚪ Neutral
- 🟢 Positive

The project follows an end-to-end NLP workflow, starting from data collection and preprocessing, continuing through feature extraction and model development, and ending with deployment using **Streamlit**.

---

## 🎯 Objectives

- Analyze public sentiment toward the M-Pajak application.
- Identify dominant user complaints and satisfaction.
- Build an automatic sentiment classification model.
- Deploy the model into an interactive web application.

---

# 📊 Dataset

- Source : Google Play Store Reviews
- Total Reviews : ±5,500
- Language : Indonesian
- Domain : Government Mobile Application

Example review:

> "Kode OTP selalu telat masuk jadi tidak bisa login."

Expected Output:

> **Negative**

---

# 🛠️ NLP Pipeline

The preprocessing pipeline includes:

```
Raw Text
      │
      ▼
Cleaning
      │
      ▼
Case Folding
      │
      ▼
Tokenization
      │
      ▼
Slang Word Normalization
      │
      ▼
Stopword Removal
      │
      ▼
Stemming (Sastrawi)
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Machine Learning Model
```

---

# 🔍 Text Preprocessing

The preprocessing stage consists of:

- Remove URL
- Remove Emoji
- Remove Number
- Remove Punctuation
- Lowercase Conversion
- Tokenization
- Slang Normalization
- Stopword Removal
- Indonesian Stemming using **Sastrawi**

---

# 🤖 Machine Learning

Feature Extraction

- TF-IDF Vectorizer

Algorithms Tested

- Naive Bayes
- Logistic Regression
- Linear Support Classification (LinearSVC)
- Random Forest
- Decision Tree
- ADA Boost


Final Model

✅ Linear Support Classification (LinearSVC)

Saved Model

```
svm_calibrated_model.pkl
```

Saved TF-IDF

```
tfidf.pkl
```

---

# 📈 Model Evaluation

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The calibrated SVM model achieved the best overall performance for Indonesian sentiment classification.

---

# 🚀 Streamlit Deployment

The deployed application allows users to:

- Enter Indonesian review text
- Predict sentiment automatically
- Display prediction confidence score
- Show probability for each sentiment class

Example

Input

```
Kode OTP selalu telat masuk sehingga tidak bisa login.
```

Output

```
Sentiment : Negative

Confidence : 98.7%
```

---

# 📂 Project Structure

```
M-Pajak-Sentiment-Analysis/
│
├── app.py
├── Sentiment_Analysis_Aplikasi_M_Pajak.ipynb
├── svm_calibrated_model.pkl
├── tfidf.pkl
├── requirements.txt
├── README.md
└── assets/
```

---

# ▶️ Installation

Clone repository

```bash
git clone https://github.com/yourusername/m-pajak-sentiment-analysis.git
```

Move into project

```bash
cd m-pajak-sentiment-analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

# 📦 Libraries

- pandas
- numpy
- scikit-learn
- nltk
- Sastrawi
- joblib
- streamlit
- matplotlib
- seaborn

---

# 💡 Future Improvements

- Fine-tuning IndoBERT for sentiment classification.
- Explainable AI using SHAP/LIME.
- Automatic visualization dashboard.
- Topic Modeling to identify major user complaints.
- Real-time sentiment monitoring from Google Play reviews.

---

# 👩‍💻 Author

**Hasti Sri Fatmawati**

Chemical Engineering Graduate transitioning into Data Analytics & Data Science.

**Skills**

- Python
- Machine Learning
- NLP
- Streamlit
- Scikit-Learn

LinkedIn:
> *https://www.linkedin.com/in/hasti-sri-fatmawati-361b49417/*

GitHub:
> *https://github.com/hastisf/*

---

⭐ If you find this project useful, consider giving it a **Star** on GitHub.
