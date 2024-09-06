import streamlit as st
import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Judul aplikasi
st.title("Clustering Credit Card Data")

# Instruksi
st.write("Unggah file dataset CSV untuk melakukan klasterisasi menggunakan model KMeans yang sudah dilatih.")

# Fungsi untuk load model dan scaler dari file pickle
@st.cache
def load_model():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler_model = pickle.load(f)
    return kmeans_model, scaler_model

# Load model KMeans dan scaler
kmeans_model, scaler_model = load_model()

# Fungsi untuk menampilkan hasil evaluasi klasterisasi
def evaluate_clustering(data_scaled, labels):
    silhouette_avg = silhouette_score(data_scaled, labels)
    db_index = davies_bouldin_score(data_scaled, labels)
    ch_index = calinski_harabasz_score(data_scaled, labels)
    return silhouette_avg, db_index, ch_index

# Upload dataset
uploaded_file = st.file_uploader("Upload your cleaned dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Baca dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset yang diunggah:")
    st.write(df.head())

    # Proses data (cleaning dan scaling)
    df_cleaned = df.drop(columns=['CUST_ID'], errors='ignore')  # Hapus CUST_ID jika ada
    df_cleaned = df_cleaned.fillna(df_cleaned.median())  # Isi nilai kosong dengan median
    data_scaled = scaler_model.transform(df_cleaned)  # Lakukan scaling pada data

    # Prediksi klaster menggunakan model yang telah diload
    predicted_clusters = kmeans_model.predict(data_scaled)
    df['Cluster'] = predicted_clusters

    # Tampilkan hasil klasterisasi
    st.write("Hasil klasterisasi:")
    st.write(df)

    # Evaluasi klasterisasi
    silhouette_avg, db_index, ch_index = evaluate_clustering(data_scaled, predicted_clusters)
    st.write(f"Silhouette Score: {silhouette_avg:.4f}")
    st.write(f"Davies-Bouldin Index: {db_index:.4f}")
    st.write(f"Calinski-Harabasz Index: {ch_index:.4f}")
