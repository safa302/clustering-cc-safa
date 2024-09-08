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
balance_frequency = st.number_input("Balance Frequency", min_value=0.0, max_value=1.0, step=0.01)
purchases = st.number_input("Purchases", min_value=0.0, max_value=1000000.0, step=1000.0)
oneoff_purchases = st.number_input("One-off Purchases", min_value=0.0, max_value=1000000.0, step=1000.0)
installments_purchases = st.number_input("Installments Purchases", min_value=0.0, max_value=1000000.0, step=1000.0)
cash_advance = st.number_input("Cash Advance", min_value=0.0, max_value=1000000.0, step=1000.0)
purchases_frequency = st.number_input("Purchases Frequency", min_value=0.0, max_value=1.0, step=0.01)
oneoff_purchases_frequency = st.number_input("One-off Purchases Frequency", min_value=0.0, max_value=1.0, step=0.01)
purchases_installments_frequency = st.number_input("Purchases Installments Frequency", min_value=0.0, max_value=1.0, step=0.01)
cash_advance_frequency = st.number_input("Cash Advance Frequency", min_value=0.0, max_value=1.0, step=0.01)
cash_advance_trx = st.number_input("Cash Advance TRX", min_value=0, max_value=100, step=1)
purchases_trx = st.number_input("Purchases TRX", min_value=0, max_value=100, step=1)
credit_limit = st.number_input("Credit Limit", min_value=0.0, max_value=100000.0, step=1000.0)
payments = st.number_input("Payments", min_value=0.0, max_value=100000.0, step=1000.0)
minimum_payments = st.number_input("Minimum Payments", min_value=0.0, max_value=100000.0, step=1000.0)
prc_full_payment = st.number_input("Percentage Full Payment", min_value=0.0, max_value=1.0, step=0.01)
tenure = st.number_input("Tenure", min_value=0, max_value=12, step=1)

# Jika tombol prediksi ditekan
if st.button("Prediksi Klaster"):
    # Data input dalam bentuk array
    user_data = np.array([[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
                           purchases_frequency, oneoff_purchases_frequency, purchases_installments_frequency,
                           cash_advance_frequency, cash_advance_trx, purchases_trx, credit_limit, payments,
                           minimum_payments, prc_full_payment, tenure]])

    # Normalisasi data input menggunakan scaler yang sama dengan model
    scaled_data = scaler.transform(user_data)

    # Prediksi klaster menggunakan model KMeans
    cluster = kmeans.predict(scaled_data)

    # Menampilkan hasil prediksi
    st.success(f"Nasabah ini diprediksi berada di klaster: {cluster[0]}")
