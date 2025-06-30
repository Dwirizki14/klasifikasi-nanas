# app.py
import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load model & scaler
model = joblib.load('model_knn.pkl')
scaler = joblib.load('scaler_knn.pkl')

st.title('🍍 Prediksi Tingkat Kematangan Buah Nanas')
st.write('Upload gambar nanas, lalu sistem akan memprediksi tingkat kematangannya.')

# Upload file gambar
uploaded_file = st.file_uploader("Pilih file gambar...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar nanas yang diupload', use_column_width=True)

    # Konversi ke array & ekstraksi mean RGB
    img_array = np.array(image)
    mean_rgb = img_array.mean(axis=(0,1))
    r, g, b = mean_rgb
    st.write(f'📊 Nilai rata-rata RGB: R={r:.0f}, G={g:.0f}, B={b:.0f}')

    # Normalisasi input
    rgb_input = np.array([[r, g, b]])
    rgb_input_scaled = scaler.transform(rgb_input)

    # Prediksi
    prediction = model.predict(rgb_input_scaled)
    st.subheader(f'✅ Prediksi Tingkat Kematangan: **{prediction[0]}**')
