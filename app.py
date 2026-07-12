import streamlit as st

import joblib

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory



# --- Setup ---

st.set_page_config(page_title="Analisis Sentimen M-Pajak", page_icon="logo.png", layout="wide")

factory = StemmerFactory()

stemmer = factory.create_stemmer()



# Load Model

model = joblib.load('svm_calibrated_model.pkl')

tfidf = joblib.load('tfidf.pkl')



# --- Sidebar (Input Area) ---

with st.sidebar:

    st.header("Input Data")

    user_input = st.text_area("Masukkan ulasan di sini:", height=150)

    tombol_analisis = st.button("Analisis Sentimen")

    st.markdown("---")

    st.caption("Proyek Analisis Sentimen M-Pajak")



# --- Main Area ---

st.title("📊 Analisis Sentimen M-Pajak")

st.write("Pantau sentimen pengguna secara real-time melalui dashboard cerdas.")



if tombol_analisis:

    if user_input:

        # 1. Prediksi

        clean_text = user_input.lower() # Sederhanakan preprocessing untuk contoh

        text_vec = tfidf.transform([clean_text])

        prediction = model.predict(text_vec)[0]

        confidence = model.predict_proba(text_vec).max() * 100

        

        # 2. Visualisasi Dashboard

        col1, col2 = st.columns(2)

        

        # Tampilkan status warna

        warna = "red" if prediction == "negative" else "green"

        col1.metric("Sentimen", prediction.upper())

        col2.metric("Confidence Score", f"{confidence:.2f}%")

        

        # Pesan status

        if prediction == "negative":

            st.error("⚠️ Model mendeteksi keluhan. Perlu perhatian khusus.")

        else:

            st.success("✅ Model mendeteksi sentimen positif.")

            

        # 3. Transparansi (Traceability)

        with st.expander("Lihat Detail Pemrosesan"):

            st.write("Input: ", user_input)

            st.write("Stemmed: ", stemmer.stem(user_input))

            st.info("Model menggunakan SVM + TF-IDF Vectorizer")

            

    else:

        st.warning("Silakan masukkan teks ulasan terlebih dahulu!")