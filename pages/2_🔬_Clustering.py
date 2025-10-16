import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances, silhouette_score
import matplotlib.pyplot as plt
import os
import time

# =================================================================================
# FUNGSI-FUNGSI BANTUAN
# =================================================================================

@st.cache_data
def load_data(source):
    """Memuat data dari file yang diunggah atau path file."""
    try:
        if isinstance(source, str):
            if source.endswith('.xlsx'):
                return pd.read_excel(source)
            else:
                return pd.read_csv(source)
        else:
            if source.name.endswith('.xlsx'):
                return pd.read_excel(source)
            else:
                return pd.read_csv(source)
    except Exception as e:
        st.error(f"Error memuat data: {e}")
        return None

def run_dbscan(data_scaled, eps, min_samples):
    """Menjalankan DBSCAN."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_scaled)
    return dbscan, clusters

def run_intelligent_kmeans(normalized_data, feature_names):
    """
    Menjalankan algoritma Intelligent K-Means lengkap berdasarkan skrip yang diberikan.
    Fungsi ini akan mengembalikan hasil akhir dan log proses yang detail.
    """
    log_area = st.empty()
    logs = ["**Memulai Proses Intelligent K-Means...**"]
    
    # --- FASE 1: PENCARIAN INITIAL CENTROID OPTIMAL ---
    logs.append("\n--- FASE 1: Pencarian Initial Centroid ---")
    log_area.info("\n".join(logs))
    time.sleep(1)

    historical_centroids = []
    
    # Centroid 1: Titik terjauh dari pusat massa
    center_of_mass = normalized_data.mean(axis=0)
    distances_to_com = np.linalg.norm(normalized_data - center_of_mass, axis=1)
    c1_idx = np.argmax(distances_to_com)
    centroid1 = normalized_data[c1_idx]
    historical_centroids.append(centroid1)
    logs.append(f"➡️ Centroid 1 ditemukan (indeks data: {c1_idx}).")
    log_area.info("\n".join(logs))
    time.sleep(1)

    # Centroid 2: Titik terjauh dari Centroid 1
    distances_to_c1 = np.linalg.norm(normalized_data - centroid1, axis=1)
    c2_idx = np.argmax(distances_to_c1)
    centroid2 = normalized_data[c2_idx]
    historical_centroids.append(centroid2)
    logs.append(f"➡️ Centroid 2 ditemukan (indeks data: {c2_idx}).")
    log_area.info("\n".join(logs))
    time.sleep(1)

    k = 2
    while True:
        logs.append(f"\n**Mencari kandidat untuk Centroid ke-{k+1}...**")
        log_area.info("\n".join(logs))
        time.sleep(1)

        reference_centroids = np.array(historical_centroids)
        point_to_centroid_distances = pairwise_distances(normalized_data, reference_centroids)
        avg_of_distances = point_to_centroid_distances.mean(axis=1)
        next_centroid_idx = np.argmax(avg_of_distances)
        next_centroid_candidate = normalized_data[next_centroid_idx]
        is_duplicate = any(np.allclose(next_centroid_candidate, old_c) for old_c in historical_centroids)

        if is_duplicate:
            logs.append(f"🛑 Kandidat (indeks {next_centroid_idx}) adalah duplikat. Pencarian dihentikan.")
            log_area.info("\n".join(logs))
            time.sleep(1)
            break
        else:
            logs.append(f"✅ Kandidat unik ditemukan (indeks {next_centroid_idx}). Ditambahkan sebagai Centroid ke-{k+1}.")
            historical_centroids.append(next_centroid_candidate)
            k += 1
            log_area.info("\n".join(logs))
            time.sleep(1)

    final_k = len(historical_centroids)
    final_initial_centroids = np.array(historical_centroids)
    logs.append(f"\n**✅ FASE 1 SELESAI: Ditemukan K optimal = {final_k}**")
    log_area.info("\n".join(logs))
    time.sleep(1)
    
    # --- FASE 2: MENJALANKAN K-MEANS FINAL HINGGA KONVERGEN ---
    logs.append("\n--- FASE 2: Clustering Final dengan K-Means ---")
    log_area.info("\n".join(logs))
    time.sleep(1)

    current_centroids = final_initial_centroids
    labels_sebelumnya = np.full(shape=len(normalized_data), fill_value=-1)
    iter_count = 0

    while True:
        iter_count += 1
        logs.append(f"\n**Iterasi Konvergensi Final ke-{iter_count}...**")
        log_area.info("\n".join(logs))
        time.sleep(1)
        
        # Tentukan klaster baru
        distances = pairwise_distances(normalized_data, current_centroids)
        labels_sekarang = np.argmin(distances, axis=1)

        # Cek konvergensi
        if np.array_equal(labels_sekarang, labels_sebelumnya):
            logs.append(f"✅ KONVERGENSI TERCAPAI dalam {iter_count-1} langkah.")
            log_area.info("\n".join(logs))
            time.sleep(1)
            break

        perpindahan = labels_sekarang != labels_sebelumnya
        if not np.any(perpindahan):
             logs.append("   - Tidak ada titik data yang berpindah klaster.")
        else:
            pindah_count = np.sum(perpindahan)
            logs.append(f"   - {pindah_count} titik data berpindah klaster.")
        log_area.info("\n".join(logs))
        time.sleep(0.5)

        labels_sebelumnya = labels_sekarang
        
        # Update posisi centroid
        logs.append("   - Memperbarui posisi centroid...")
        current_centroids = np.array([normalized_data[labels_sekarang == i].mean(axis=0) for i in range(final_k)])
        log_area.info("\n".join(logs))
        time.sleep(0.5)
        
        if iter_count > 100:
            logs.append("⚠️ Peringatan: Iterasi melebihi 100, berhenti paksa.")
            log_area.warning("\n".join(logs))
            break
            
    # Kembalikan hasil
    return labels_sekarang, current_centroids, final_k

def plot_dbscan_results(data, dbscan_model, clusters, feature_x, feature_y):
    # (Fungsi ini tetap sama)
    core_sample_indices = dbscan_model.core_sample_indices_
    is_core = np.zeros(len(data), dtype=bool)
    if len(core_sample_indices) > 0:
        is_core[core_sample_indices] = True
    is_noise = (clusters == -1)
    is_border = ~(is_core | is_noise)
    fig, ax = plt.subplots(figsize=(10, 7))
    if np.any(is_core): ax.scatter(data.loc[is_core, feature_x], data.loc[is_core, feature_y], s=100, c=clusters[is_core], cmap='viridis', edgecolor='k', label='Core Points')
    if np.any(is_border): ax.scatter(data.loc[is_border, feature_x], data.loc[is_border, feature_y], s=40, c=clusters[is_border], cmap='viridis', edgecolor='k', label='Border Points')
    if np.any(is_noise): ax.scatter(data.loc[is_noise, feature_x], data.loc[is_noise, feature_y], s=50, c='black', marker='x', label='Noise')
    ax.set_title(f'Visualisasi DBSCAN ({feature_x} vs {feature_y})', fontsize=16)
    ax.set_xlabel(feature_x); ax.set_ylabel(feature_y)
    ax.legend(); ax.grid(True, alpha=0.5)
    st.pyplot(fig)

def plot_kmeans_results(data, clusters, final_centroids_scaled, feature_x, feature_y, scaler):
    # (Fungsi ini disesuaikan untuk Intelligent K-Means)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot data points
    scatter = ax.scatter(data[feature_x], data[feature_y], c=clusters, cmap='viridis', alpha=0.8, label='Data Points')
    
    # Buat dataframe dari centroid yang ternormalisasi untuk inverse transform
    centroids_df_scaled = pd.DataFrame(final_centroids_scaled, columns=data.columns[1:-1]) # <-- BARIS YANG DIPERBAIKI    # Inverse transform untuk mendapatkan posisi centroid dalam skala data asli
    centroids_original_scale = scaler.inverse_transform(centroids_df_scaled)
    centroids_original_df = pd.DataFrame(centroids_original_scale, columns=data.columns[1:-1]) # <-- CORRECTED LINE
    # Plot centroids
    ax.scatter(centroids_original_df[feature_x], centroids_original_df[feature_y], s=250, c='red', marker='P', label='Centroids', edgecolor='k')

    ax.set_title(f'Visualisasi Intelligent K-Means ({feature_x} vs {feature_y})', fontsize=16)
    ax.set_xlabel(feature_x); ax.set_ylabel(feature_y)
    ax.grid(True, alpha=0.5)
    legend1 = ax.legend(*scatter.legend_elements(), title="Klaster")
    ax.add_artist(legend1)
    ax.legend(loc='upper right')
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
            if not example_files: st.warning("Folder 'contoh_dataset' kosong.")
            else:
                selected_file = st.selectbox("Pilih Contoh Dataset", example_files)
                if selected_file: data = load_data(os.path.join("contoh_dataset", selected_file))
        except FileNotFoundError: st.error("Folder 'contoh_dataset' tidak ditemukan.")
    else:
        uploaded_file = st.file_uploader("Unggah file Anda (.xlsx atau .csv)", type=['xlsx', 'csv'])
        if uploaded_file: data = load_data(uploaded_file)
    
    try:
        with open("template_dataset.xlsx", "rb") as file:
            st.download_button(label="📥 Download Template Dataset (.xlsx)", data=file, file_name="template_dataset.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except FileNotFoundError: st.error("File 'template_dataset.xlsx' tidak ditemukan.")

    if selected_method == "DBSCAN":
        st.subheader("Parameter DBSCAN")
        eps_value = st.slider("Epsilon (eps)", 0.01, 2.0, 0.5, 0.01)
        min_samples_value = st.slider("Minimum Samples (min_pts)", 2, 50, 5, 1)

# --- AREA UTAMA: HASIL ANALISIS ---
if data is not None:
    st.header("1. Pratinjau Dataset")
    st.dataframe(data.head())
    features = data.iloc[:, 1:]
    if not all(features.dtypes.apply(pd.api.types.is_numeric_dtype)):
        st.error("Error: Pastikan semua kolom fitur adalah numerik.")
    else:
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        st.markdown("---")
        st.header("2. Hasil Analisis Clustering")

        if selected_method == "DBSCAN":
            # (Logika DBSCAN tidak berubah)
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
            st.info("Metode ini secara otomatis menentukan jumlah klaster (K) yang optimal dan menjalankannya hingga konvergen.")
            
            if st.button("🚀 Jalankan Analisis Intelligent K-Means", type="primary"):
                st.subheader("Log Proses Real-time")
                
                # Menjalankan algoritma dan mendapatkan hasilnya
                final_labels, final_centroids_scaled, final_k = run_intelligent_kmeans(features_scaled, features.columns)
                
                st.success("🎉 Analisis Selesai!")
                st.markdown("---")
                st.header("3. Ringkasan Hasil")
                
                data['Cluster'] = final_labels + 1 # Tambah 1 agar klaster dimulai dari 1
                
                # Tampilkan Metrik
                col1, col2 = st.columns(2)
                col1.metric("Jumlah Klaster Optimal (K)", final_k)
                score = silhouette_score(features_scaled, final_labels)
                col2.metric("Silhouette Score Final", f"{score:.3f}")

                # Visualisasi
                st.subheader("Visualisasi Hasil Clustering")
                feature_cols = features.columns.tolist()
                feat_x = st.selectbox("Pilih Fitur Sumbu X", feature_cols, index=0, key="ikm_x")
                feat_y = st.selectbox("Pilih Fitur Sumbu Y", feature_cols, index=1 if len(feature_cols)>1 else 0, key="ikm_y")
                plot_kmeans_results(data, final_labels, final_centroids_scaled, feat_x, feat_y, scaler)

                # Tampilkan data hasil
                st.subheader("Data dengan Hasil Cluster")
                st.dataframe(data)
else:
    st.info("👈 Silakan pilih atau unggah dataset Anda dari sidebar untuk memulai analisis.")