import streamlit as st
import joblib
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. Setup Preprocessing (HARUS SAMA DENGAN SAAT TRAINING)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    text = str(text).lower()
    # Tambahkan custom stopword jika ada saat training
    tokens = [w for w in text.split() if w not in stop_words]
    return stemmer.stem(' '.join(tokens))

# 2. Load Model & Vectorizer
model = joblib.load('svm_calibrated_model.pkl')
tfidf = joblib.load('tfidf.pkl')

# 3. UI Streamlit
st.title("Analisis Sentimen M-Pajak")
st.write("Masukkan ulasan aplikasi untuk mengetahui sentimennya.")

user_input = st.text_input("Tulis ulasan Anda di sini:")

if st.button("Analisis Sentimen"):
    if user_input:
        # Preprocessing
        clean_text = preprocess_text(user_input)
        # Vectorizing
        text_vec = tfidf.transform([clean_text])
        # Prediction
        prediction = model.predict(text_vec)[0]
        confidence = model.predict_proba(text_vec).max() * 100
        
        # Display
        st.subheader("Hasil Prediksi:")
        st.success(f"Sentimen: {prediction}")
        st.info(f"Confidence Score: {confidence:.2f}%")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu!")