# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import os

# Impor fungsi-fungsi yang telah dipisah
# get_example_data_options sudah tidak diperlukan lagi
from utils import load_data, get_template_file 
from clustering_algorithms import run_dbscan, run_intelligent_kmeans
from clustering_visuals import plot_dbscan_results, plot_kmeans_results

# =================================================================================
# TAMPILAN ANTARMUKA STREAMLIT
# =================================================================================

st.set_page_config(page_title="Clustering", page_icon="🔬", layout="wide")
st.title("🔬 Laman Analisis Clustering")

# --- 1. Konfigurasi Analisis ---
st.header("1. Konfigurasi Analisis")
col1, col2 = st.columns([1, 2])

with col1:
    selected_method = st.selectbox(
        "Pilih Metode Clustering", 
        ["Intelligent K-Means", "DBSCAN"]
    )

with col2:
    data_source_option = st.radio(
        "Pilih Sumber Data", 
        ["Gunakan Contoh Dataset", "Unggah File Sendiri"], 
        horizontal=True,
        label_visibility="collapsed" # Sembunyikan label "Pilih Sumber Data"
    )

# --- 2. Input Data ---
st.header("2. Input Data & Fitur")
data = None

# --- PERUBAHAN DI SINI ---
# Mapping Fitur (sesuai permintaan baru)
mapping_fitur = {
    "Indeks Pembangunan Manusia Laki-Laki": "IPM_L",
    "Indeks Pembangunan Manusia Perempuan": "IPM_P",
    "Angka Harapan Hidup Laki-Laki": "AHH_L",
    "Angka Harapan Hidup Perempuan": "AHH_P",
    "Pengeluaran Per Kapita Laki-Laki": "PKP_L",
    "Pengeluaran Per Kapita Perempuan": "PKP_P",

}

# Logika untuk memuat data (sesuai permintaan baru)
if data_source_option == "Gunakan Contoh Dataset":
    dataset_path = "contoh_dataset/dataset.xlsx"
    if os.path.exists(dataset_path):
        data = load_data(dataset_path)
    else:
        st.error(f"Error: File dataset contoh '{dataset_path}' tidak ditemukan.")
else:
    uploaded_file = st.file_uploader("Unggah file Anda (.xlsx atau .csv)", type=['xlsx', 'csv'])
    if uploaded_file:
        data = load_data(uploaded_file)
# --- AKHIR PERUBAHAN ---

# Tombol download template
template_data = get_template_file()
if template_data:
    st.download_button(
        label="📥 Download Template Dataset (.xlsx)",
        data=template_data,
        file_name="template_dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- PERUBAHAN DI SINI ---
# Standardisasi kolom identifier
# Ini akan mengubah 'label' dari dataset.xlsx menjadi 'Nama Wilayah'
if data is not None:
    if "Label" in data.columns:
        data = data.rename(columns={"Label": "Nama Wilayah"})
# --- AKHIR PERUBAHAN ---

# --- 3. Filter & Persiapan Data ---
if data is not None:
    st.subheader("Filter Data dan Fitur")

    # Validasi kolom wajib
    if "Tahun" not in data.columns:
        st.error("Dataset Error: Kolom 'Tahun' tidak ditemukan. Silakan periksa template.")
        st.stop()
    if "Nama Wilayah" not in data.columns:
        # Peringatan ini akan muncul jika 'label' atau 'Nama Wilayah' tidak ada
        st.warning("Dataset Peringatan: Kolom 'Nama Wilayah' (atau 'label') tidak ditemukan. Ini disarankan sebagai identifier.")

    # 1. Multiselect Tahun
    available_years = sorted(data["Tahun"].unique())
    selected_years = st.multiselect("Pilih Tahun", available_years, default=available_years)
    
    # 2. Multiselect Fitur
    # Filter mapping agar hanya menampilkan fitur yang ada di data
    available_features_map = {key: val for key, val in mapping_fitur.items() if val in data.columns}
    if not available_features_map:
         st.error(f"Dataset Error: Tidak ditemukan satupun fitur yang dikenali dari daftar: {list(mapping_fitur.values())}")
         st.stop()

    selected_feature_keys = st.multiselect(
        "Pilih Fitur untuk Clustering", 
        available_features_map.keys(), 
        default=list(available_features_map.keys())
    )
    
    # Ambil nama kolom aktual dari fitur yang dipilih
    fitur_terpilih = [available_features_map[key] for key in selected_feature_keys]

    if not selected_years or not fitur_terpilih:
        st.warning("Silakan pilih minimal satu tahun dan satu fitur.")
        st.stop()
    
    # Filter data berdasarkan pilihan
    data_filtered = data[data["Tahun"].isin(selected_years)].copy()
    
    # Pisahkan identifier dan fitur
    identifier_cols = [col for col in ["Nama Wilayah", "Tahun"] if col in data_filtered.columns]
    
    # Siapkan data untuk clustering (handle NaN)
    data_clustering = data_filtered[identifier_cols + fitur_terpilih]
    if data_clustering[fitur_terpilih].isnull().values.any():
        st.warning(f"Data mengandung nilai NaN. {data_clustering[fitur_terpilih].isnull().values.sum()} baris dengan NaN akan dihapus sebelum clustering.")
        data_clustering = data_clustering.dropna(subset=fitur_terpilih).reset_index(drop=True)
    
    if data_clustering.empty:
        st.error("Tidak ada data tersisa setelah filtering. Silakan ubah pilihan Anda.")
        st.stop()
        
    data_clustering.index = data_clustering.index + 1
    st.dataframe(data_clustering, use_container_width=True)
    
    # Normalisasi data
    features_to_scale = data_clustering[fitur_terpilih]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_to_scale)

    # --- 4. Parameter & Eksekusi ---
    st.header("3. Parameter & Eksekusi")
    
    if selected_method == "DBSCAN":
        st.subheader("Parameter DBSCAN")
        col1_param, col2_param = st.columns(2)
        eps_value = col1_param.slider("Epsilon (eps)", 0.01, 2.0, 0.5, 0.01)
        min_samples_value = col2_param.slider("Minimum Samples (min_pts)", 2, 50, 5, 1)
        
        if st.button("🚀 Jalankan Analisis DBSCAN", type="primary"):
            st.header("4. Hasil Analisis DBSCAN")
            dbscan, clusters = run_dbscan(features_scaled, eps_value, min_samples_value)
            
            # Tambahkan hasil cluster ke dataframe
            data_clustering['Cluster'] = clusters
            
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            # Tampilkan Metrik
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Jumlah Klaster", n_clusters)
            res_col2.metric("Jumlah Noise/Outlier", n_noise)
            
            if n_clusters > 1:
                score = silhouette_score(features_scaled, clusters)
                res_col3.metric("Silhouette Score", f"{score:.3f}")
            else:
                res_col3.metric("Silhouette Score", "N/A")
            
            # Visualisasi
            st.subheader("Visualisasi Hasil")
            plot_col1, plot_col2 = st.columns(2)
            feat_x = plot_col1.selectbox("Pilih Fitur Sumbu X", fitur_terpilih, index=0, key="dbscan_x")
            feat_y_index = 1 if len(fitur_terpilih) > 1 else 0
            feat_y = plot_col2.selectbox("Pilih Fitur Sumbu Y", fitur_terpilih, index=feat_y_index, key="dbscan_y")
            
            plot_dbscan_results(data_clustering, dbscan, clusters, feat_x, feat_y)
            
            # Tampilkan data hasil
            st.subheader("Data dengan Hasil Cluster")
            st.dataframe(data_clustering, use_container_width=True)

    elif selected_method == "Intelligent K-Means":
        st.info("Metode ini secara otomatis menentukan jumlah klaster (K) yang optimal. Tidak ada parameter tambahan yang diperlukan.")
        
        if st.button("🚀 Jalankan Analisis Intelligent K-Means", type="primary"):
            st.header("4. Hasil Analisis Intelligent K-Means")
            st.subheader("Log Proses Real-time")
            
            # Jalankan algoritma
            final_labels, final_centroids_scaled, final_k = run_intelligent_kmeans(features_scaled, fitur_terpilih)
            
            st.success("🎉 Analisis Selesai!")
            st.markdown("---")
            
            # Tambahkan hasil cluster ke dataframe (dimulai dari 1)
            data_clustering['Cluster'] = final_labels + 1
            
            # Tampilkan Metrik
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Jumlah Klaster Optimal (K)", final_k)
            score = silhouette_score(features_scaled, final_labels)
            res_col2.metric("Silhouette Score Final", f"{score:.3f}")

            # Visualisasi
            st.subheader("Visualisasi Hasil Clustering")
            plot_col1, plot_col2 = st.columns(2)
            feat_x = plot_col1.selectbox("Pilih Fitur Sumbu X", fitur_terpilih, index=0, key="ikm_x")
            feat_y_index = 1 if len(fitur_terpilih) > 1 else 0
            feat_y = plot_col2.selectbox("Pilih Fitur Sumbu Y", fitur_terpilih, index=feat_y_index, key="ikm_y")
            
            plot_kmeans_results(data_clustering, final_labels, final_centroids_scaled, feat_x, feat_y, scaler, fitur_terpilih)

            # Tampilkan data hasil
            st.subheader("Data dengan Hasil Cluster")
            st.dataframe(data_clustering, use_container_width=True)

else:
    st.info("☝️ Silakan pilih atau unggah dataset Anda untuk memulai analisis.")