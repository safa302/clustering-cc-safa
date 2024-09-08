import streamlit as st
import pickle
import numpy as np

# Load model dan scaler dari file pickle
with open('kmeans_model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Judul aplikasi
st.title("Aplikasi Prediksi Klaster Kartu Kredit")

# Input data dari pengguna
st.header("Input Data Nasabah")
balance = st.number_input("Balance", min_value=0.0, max_value=1000000.0, step=1000.0)
purchases = st.number_input("Purchases", min_value=0.0, max_value=1000000.0, step=1000.0)
cash_advance = st.number_input("Cash Advance", min_value=0.0, max_value=1000000.0, step=1000.0)
credit_limit = st.number_input("Credit Limit", min_value=0.0, max_value=100000.0, step=1000.0)
payments = st.number_input("Payments", min_value=0.0, max_value=100000.0, step=1000.0)
tenure = st.number_input("Tenure", min_value=0, max_value=12, step=1)  # Misalnya fitur tambahan

# Debugging: Periksa bentuk input
st.write(f"Scaler expects {scaler.n_features_in_} features")
user_data = np.array([[balance, purchases, cash_advance, credit_limit, payments, tenure]])
st.write(f"User input shape: {user_data.shape}")

# Jika tombol prediksi ditekan
if st.button("Prediksi Klaster"):
    # Normalisasi data input menggunakan scaler yang sama dengan model
    try:
        scaled_data = scaler.transform(user_data)

        # Prediksi klaster menggunakan model KMeans
        cluster = kmeans.predict(scaled_data)

        # Menampilkan hasil prediksi
        st.success(f"Nasabah ini diprediksi berada di klaster: {cluster[0]}")
    except ValueError as e:
        st.error(f"Error: {e}")


