import streamlit as st

import joblib

import nltk

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import re



# Konfigurasi Page

st.set_page_config(page_title="Analisis Sentimen M-Pajak", page_icon="logo.png", layout="wide")



# Setup Sastrawi & Stopwords

nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()

stemmer = factory.create_stemmer()



# --- FUNGSI PREPROCESSING & TRANSPARANSI ---

def clean_text(text):

    text = re.sub(r'[^\w\s]', '', str(text).lower()) # Hapus simbol

    return text



def remove_stopwords_stem(text):

    tokens = [w for w in text.split() if w not in stop_words]

    return stemmer.stem(' '.join(tokens))



def get_processing_steps(text):

    # Tahap 1: Case Folding & Cleaning

    cleaned = clean_text(text)

    # Tahap 2: Stopwords & Stemming

    final = remove_stopwords_stem(cleaned)

    return cleaned, final



# --- LOAD MODEL ---

@st.cache_resource

def load_assets():

    model = joblib.load('svm_calibrated_model.pkl')

    tfidf = joblib.load('tfidf.pkl')

    return model, tfidf



model, tfidf = load_assets()



# --- SIDEBAR (INFORMASI MODEL) ---

with st.sidebar:

    st.header("📌 Informasi Model")

    st.info("Model ini digunakan untuk menganalisis sentimen pengguna aplikasi M-Pajak secara real-time.")

    

    st.write("**Detail Teknis:**")

    st.markdown("- **Algoritma:** LinearSVC (Calibrated)")

    st.markdown("- **Feature Extraction:** TF-IDF Vectorizer")

    st.markdown("- **Dataset:** 5,500 Ulasan (Scraping Google PlayStore)")

    st.markdown("- **Akurasi:** 81.03%")

    

    st.divider()

    st.caption("Dibuat untuk kebutuhan analisis sentimen aplikasi perpajakan.")

col1, col2 = st.columns([1, 8]) # Angka 1 dan 8 menentukan lebar kolom (logo lebih kecil dari judul)

with col1:
    # Memanggil logo dengan ukuran yang pas
    st.image("logo.png", width=70) 

with col2:
    # Menggeser judul agar sejajar dengan logo
    st.title("Analisis Sentimen M-Pajak")

# --- MAIN UI ---

st.write("Dashboard ini memprediksi apakah ulasan pengguna bersifat **Positif, Negatif, atau Netral**.")



user_input = st.text_area("Tulis ulasan aplikasi di sini:", placeholder="Contoh: Loginnya susah banget, loading terus...")



if st.button("Analisis Sentimen"):

    if user_input:

        # Pipeline Data

        clean_txt, final_txt = get_processing_steps(user_input)

        vec = tfidf.transform([final_txt])

        

        # Prediksi

        prediction = model.predict(vec)[0]

        confidence = model.predict_proba(vec).max() * 100

        

        # Display Metric

        col1, col2 = st.columns(2)

        warna = {"positive": "green", "negative": "red", "neutral": "orange"}

        

        col1.metric("Hasil Sentimen", prediction.upper())

        col2.metric("Confidence", f"{confidence:.2f}%")

        

        # Transparansi Data Science

        with st.expander("🔍 Lihat Detail Proses (Data Science Pipeline)"):

            st.write(f"**1. Input Asli:** `{user_input}`")

            st.write(f"**2. Case Folding & Cleaning:** `{clean_txt}`")

            st.write(f"**3. Stopwords Removal & Stemming:** `{final_txt}`")

            st.success("Teks berhasil diproses dan dikonversi ke vektor numerik.")

            

    else:

        st.warning("Silakan masukkan teks ulasan terlebih dahulu!")