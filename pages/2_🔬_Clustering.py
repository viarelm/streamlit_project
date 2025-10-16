import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

# =================================================================================
# FUNGSI-FUNGSI BANTUAN (Sama seperti sebelumnya)
# =================================================================================
@st.cache_data
def load_data(source):
    """Memuat data dari file yang diunggah atau path file."""
    try:
        if isinstance(source, str): # Jika sumber adalah path file
            if source.endswith('.xlsx'):
                return pd.read_excel(source)
            else:
                return pd.read_csv(source)
        else: # Jika sumber adalah file yang diunggah
            if source.name.endswith('.xlsx'):
                return pd.read_excel(source)
            else:
                return pd.read_csv(source)
    except Exception as e:
        st.error(f"Error memuat data: {e}")
        return None

# ... (Salin semua fungsi bantuan lainnya dari respons sebelumnya) ...
def run_dbscan(data_scaled, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_scaled)
    return dbscan, clusters

def run_kmeans(data_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(data_scaled)
    return kmeans, clusters

def plot_dbscan_results(data, dbscan_model, clusters, feature_x, feature_y):
    core_sample_indices = dbscan_model.core_sample_indices_
    is_core = np.zeros(len(data), dtype=bool)
    if len(core_sample_indices) > 0:
        is_core[core_sample_indices] = True
    is_noise = (clusters == -1)
    is_border = ~(is_core | is_noise)
    fig, ax = plt.subplots(figsize=(10, 7))
    if np.any(is_core):
        ax.scatter(data.loc[is_core, feature_x], data.loc[is_core, feature_y], s=100, c=clusters[is_core], cmap='viridis', edgecolor='k', label='Core Points')
    if np.any(is_border):
        ax.scatter(data.loc[is_border, feature_x], data.loc[is_border, feature_y], s=40, c=clusters[is_border], cmap='viridis', edgecolor='k', label='Border Points')
    if np.any(is_noise):
        ax.scatter(data.loc[is_noise, feature_x], data.loc[is_noise, feature_y], s=50, c='black', marker='x', label='Noise')
    ax.set_title(f'Visualisasi DBSCAN ({feature_x} vs {feature_y})', fontsize=16)
    ax.set_xlabel(feature_x); ax.set_ylabel(feature_y)
    ax.legend(); ax.grid(True, alpha=0.5)
    st.pyplot(fig)

def plot_kmeans_results(data, kmeans_model, clusters, feature_x, feature_y):
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(data[feature_x], data[feature_y], c=clusters, cmap='viridis', alpha=0.8, label='Data Points')
    ax.set_title(f'Visualisasi K-Means ({feature_x} vs {feature_y})', fontsize=16)
    ax.set_xlabel(feature_x); ax.set_ylabel(feature_y)
    ax.grid(True, alpha=0.5)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)

def plot_elbow_method(data_scaled, max_k=10):
    inertias = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(data_scaled)
        inertias.append(kmeans.inertia_)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertias, 'bo-')
    ax.set_xlabel('Jumlah Cluster (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method untuk Menentukan k Optimal')
    ax.grid(True)
    st.pyplot(fig)

# =================================================================================
# TAMPILAN ANTARMUKA STREAMLIT
# =================================================================================

st.set_page_config(page_title="Clustering", page_icon="🔬", layout="wide")
st.title("🔬 Laman Analisis Clustering")

# --- SIDEBAR: KONTROL PENGGUNA ---
with st.sidebar:
    st.header("⚙️ Panel Kontrol")
    selected_method = st.selectbox("Pilih Metode Clustering", ["DBSCAN", "Intelligent K-Means"])
    data_source_option = st.radio("Pilih Sumber Data", ["Gunakan Contoh Dataset", "Unggah File Sendiri"])
    
    data = None
    if data_source_option == "Gunakan Contoh Dataset":
        try:
            example_files = [f for f in os.listdir("contoh_dataset") if f.endswith(('.xlsx', '.csv'))]
            if not example_files:
                st.warning("Folder 'contoh_dataset' kosong.")
            else:
                selected_file = st.selectbox("Pilih Contoh Dataset", example_files)
                if selected_file:
                    data = load_data(os.path.join("contoh_dataset", selected_file))
        except FileNotFoundError:
            st.error("Folder 'contoh_dataset' tidak ditemukan.")
    else:
        uploaded_file = st.file_uploader("Unggah file Anda (.xlsx atau .csv)", type=['xlsx', 'csv'])
        if uploaded_file:
            data = load_data(uploaded_file)
    
    # --- Tombol Download Template (Diperbarui) ---
    try:
        with open("template_dataset.xlsx", "rb") as file:
            st.download_button(
                label="📥 Download Template Dataset (.xlsx)",
                data=file,
                file_name="template_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" # Tipe MIME untuk .xlsx
            )
    except FileNotFoundError:
        st.error("File 'template_dataset.xlsx' tidak ditemukan.")

    # --- Parameter Spesifik Algoritma ---
    if selected_method == "DBSCAN":
        st.subheader("Parameter DBSCAN")
        eps_value = st.slider("Epsilon (eps)", 0.01, 2.0, 0.5, 0.01)
        min_samples_value = st.slider("Minimum Samples (min_pts)", 2, 50, 5, 1)
    elif selected_method == "Intelligent K-Means":
        st.subheader("Parameter K-Means")

# --- AREA UTAMA: HASIL ANALISIS (Sama seperti sebelumnya) ---
if data is not None:
    st.header("1. Pratinjau Dataset")
    st.dataframe(data.head())
    # ... (Salin sisa kode area utama dari respons sebelumnya) ...
    features = data.iloc[:, 1:]
    if not all(features.dtypes.apply(pd.api.types.is_numeric_dtype)):
        st.error("Error: Pastikan semua kolom fitur (selain kolom pertama) adalah numerik.")
    else:
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        st.markdown("---")
        st.header("2. Hasil Analisis Clustering")

        if selected_method == "DBSCAN":
            dbscan, clusters = run_dbscan(features_scaled, eps_value, min_samples_value)
            data['Cluster'] = clusters
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Jumlah Klaster", n_clusters)
            col2.metric("Jumlah Noise/Outlier", n_noise)
            if n_clusters > 1:
                score = silhouette_score(features_scaled, clusters)
                col3.metric("Silhouette Score", f"{score:.3f}")
            else:
                col3.metric("Silhouette Score", "N/A")

            st.subheader("Visualisasi Hasil")
            feature_cols = features.columns.tolist()
            feat_x = st.selectbox("Pilih Fitur Sumbu X", feature_cols, index=0)
            feat_y = st.selectbox("Pilih Fitur Sumbu Y", feature_cols, index=1 if len(feature_cols)>1 else 0)
            plot_dbscan_results(data, dbscan, clusters, feat_x, feat_y)

            st.subheader("Data dengan Hasil Cluster")
            st.dataframe(data)

        elif selected_method == "Intelligent K-Means":
            st.subheader("Langkah 1: Menemukan 'K' Optimal dengan Elbow Method")
            st.write("Grafik di bawah menunjukkan 'inertia' untuk berbagai nilai K. 'Siku' (elbow) pada grafik biasanya merupakan indikasi jumlah cluster yang baik.")
            plot_elbow_method(features_scaled)
            
            st.subheader("Langkah 2: Jalankan K-Means dengan K Pilihan Anda")
            k_value = st.number_input("Masukkan jumlah cluster (K) pilihan Anda:", min_value=2, max_value=15, value=3, step=1)

            if st.button("Jalankan K-Means Clustering"):
                kmeans, clusters = run_kmeans(features_scaled, k_value)
                data['Cluster'] = clusters
                
                col1, col2 = st.columns(2)
                col1.metric("Jumlah Klaster (K)", k_value)
                score = silhouette_score(features_scaled, clusters)
                col2.metric("Silhouette Score", f"{score:.3f}")

                st.subheader("Visualisasi Hasil")
                feature_cols = features.columns.tolist()
                feat_x = st.selectbox("Pilih Fitur Sumbu X", feature_cols, index=0, key="km_x")
                feat_y = st.selectbox("Pilih Fitur Sumbu Y", feature_cols, index=1 if len(feature_cols)>1 else 0, key="km_y")
                plot_kmeans_results(data, kmeans, clusters, feat_x, feat_y)

                st.subheader("Data dengan Hasil Cluster")
                st.dataframe(data)
else:
    st.info("👈 Silakan pilih atau unggah dataset Anda dari sidebar untuk memulai analisis.")