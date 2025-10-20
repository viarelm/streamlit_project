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

# Mapping Fitur
mapping_fitur = {
    "Indeks Pembangunan Manusia Laki-Laki": "IPM_L",
    "Indeks Pembangunan Manusia Perempuan": "IPM_P",
    "Angka Harapan Hidup Laki-Laki": "AHH_L",
    "Angka Harapan Hidup Perempuan": "AHH_P",
    "Pengeluaran Per Kapita Laki-Laki": "PKP_L",
    "Pengeluaran Per Kapita Perempuan": "PKP_P",

}

# Logika untuk memuat data
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

# Tombol download template
template_data = get_template_file()
if template_data:
    st.download_button(
        label="📥 Download Template Dataset (.xlsx)",
        data=template_data,
        file_name="template_dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Standardisasi kolom identifier
if data is not None:
    if "Label" in data.columns:
        data = data.rename(columns={"Label": "Nama Wilayah"})

# --- 3. Filter & Persiapan Data ---
if data is not None:
    st.subheader("Filter Data dan Fitur")

    # Validasi kolom wajib
    if "Tahun" not in data.columns:
        st.error("Dataset Error: Kolom 'Tahun' tidak ditemukan. Silakan periksa template.")
        st.stop()
    if "Nama Wilayah" not in data.columns:
        st.warning("Dataset Peringatan: Kolom 'Nama Wilayah' (atau 'label') tidak ditemukan. Ini disarankan sebagai identifier.")

    # 1. Multiselect Tahun
    available_years = sorted(data["Tahun"].unique())
    selected_years = st.multiselect("Pilih Tahun", available_years, default=available_years)
    
    # 2. Multiselect Fitur
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
    fitur_terpilih_base = [available_features_map[key] for key in selected_feature_keys]

    if not selected_years or not fitur_terpilih_base:
        st.warning("Silakan pilih minimal satu tahun dan satu fitur.")
        st.stop()
    
    # 1. Filter data HANYA berdasarkan tahun terpilih dan kolom yang relevan
    data_to_pivot = data[data["Tahun"].isin(selected_years)].copy()
    data_to_pivot = data_to_pivot[["Nama Wilayah", "Tahun"] + fitur_terpilih_base]

    # 2. Lakukan Pivot
    st.subheader("Data yang Dipersiapkan untuk Clustering (Setelah Pivot)")
    try:
        data_pivoted = data_to_pivot.pivot(
            index="Nama Wilayah", 
            columns="Tahun", 
            values=fitur_terpilih_base
        )
    except ValueError as e:
        st.error(f"Error saat mem-pivot data: {e}")
        st.error("Ini biasanya terjadi jika ada duplikat 'Nama Wilayah' untuk 'Tahun' yang sama. Harap periksa data Anda.")
        st.stop()
    
    # 3. Ratakan MultiIndex kolom (misal: ('IPM_L', 2020) -> 'IPM_L_2020')
    data_pivoted.columns = [f"{val}_{year}" for val, year in data_pivoted.columns]
    
    # 4. Reset index untuk menjadikan 'Nama Wilayah' sebagai kolom
    data_clustering = data_pivoted.reset_index().copy()

    # 5. Tentukan ulang mana kolom identifier dan mana kolom fitur
    identifier_cols = ["Nama Wilayah"]
    # Fitur baru adalah semua kolom KECUALI identifier
    features_for_scaling = [col for col in data_clustering.columns if col not in identifier_cols]
    
    st.info(f"Data berhasil di-pivot. Jumlah wilayah: **{len(data_clustering)}**. Jumlah fitur: **{len(features_for_scaling)}** ({len(fitur_terpilih_base)} fitur x {len(selected_years)} tahun).")

    # 6. Siapkan data untuk clustering (handle NaN)
    if data_clustering[features_for_scaling].isnull().values.any():
        nan_count = data_clustering[features_for_scaling].isnull().values.sum()
        st.warning(f"Data mengandung **{nan_count}** nilai NaN setelah pivot (kemungkinan karena data wilayah tidak lengkap di semua tahun terpilih). Baris yang mengandung NaN akan dihapus sebelum clustering.")
        data_clustering = data_clustering.dropna(subset=features_for_scaling).reset_index(drop=True)
    
    if data_clustering.empty:
        st.error("Tidak ada data tersisa setelah filtering/pivot. Silakan ubah pilihan Anda.")
        st.stop()
        
    data_clustering.index = data_clustering.index + 1
    
    # 7. Normalisasi data
    features_to_scale_df = data_clustering[features_for_scaling]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_to_scale_df)
    
    # --- PERUBAHAN 1 DI SINI: Terapkan nilai ternormalisasi kembali ke DataFrame ---
    # Ini memastikan bahwa DataFrame yang ditampilkan di bawah
    # dan yang digunakan untuk plotting adalah data yang sudah ternormalisasi.
    data_clustering[features_for_scaling] = features_scaled
    st.success("Data telah dinormalisasi menggunakan MinMaxScaler (skala 0-1).")
    # --- AKHIR PERUBAHAN 1 ---

    st.dataframe(data_clustering, use_container_width=True)
    
    # (Array 'features_scaled' sudah siap untuk digunakan oleh fungsi clustering)
    

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
            
            data_clustering['Cluster'] = clusters
            
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Jumlah Klaster", n_clusters)
            res_col2.metric("Jumlah Noise/Outlier", n_noise)
            
            if n_clusters > 1:
                score = silhouette_score(features_scaled, clusters)
                res_col3.metric("Silhouette Score", f"{score:.3f}")
            else:
                res_col3.metric("Silhouette Score", "N/A")
            
            st.subheader("Visualisasi Hasil")
            plot_col1, plot_col2 = st.columns(2)
            
            feat_x = plot_col1.selectbox("Pilih Fitur Sumbu X", features_for_scaling, index=0, key="dbscan_x")
            feat_y_index = 1 if len(features_for_scaling) > 1 else 0
            feat_y = plot_col2.selectbox("Pilih Fitur Sumbu Y", features_for_scaling, index=feat_y_index, key="dbscan_y")
            
            # data_clustering yang dikirim ke plot sekarang sudah berisi nilai ternormalisasi
            plot_dbscan_results(data_clustering, dbscan, clusters, feat_x, feat_y)
            
            st.subheader("Data dengan Hasil Cluster (Nilai Ternormalisasi)")
            
            # --- PERUBAHAN 2 DI SINI: Mengatur urutan kolom ---
            column_order = ["Nama Wilayah", "Cluster"] + features_for_scaling
            st.dataframe(data_clustering[column_order], use_container_width=True)
            # --- AKHIR PERUBAHAN 2 ---

    elif selected_method == "Intelligent K-Means":
        st.info("Metode ini secara otomatis menentukan jumlah klaster (K) yang optimal. Tidak ada parameter tambahan yang diperlukan.")
        
        if st.button("🚀 Jalankan Analisis Intelligent K-Means", type="primary"):
            st.header("4. Hasil Analisis Intelligent K-Means")
            st.subheader("Log Proses Real-time")
            
            final_labels, final_centroids_scaled, final_k = run_intelligent_kmeans(features_scaled, features_for_scaling)
            
            st.success("🎉 Analisis Selesai!")
            st.markdown("---")
            
            data_clustering['Cluster'] = final_labels + 1
            
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Jumlah Klaster Optimal (K)", final_k)
            score = silhouette_score(features_scaled, final_labels)
            res_col2.metric("Silhouette Score Final", f"{score:.3f}")

            st.subheader("Visualisasi Hasil Clustering")
            plot_col1, plot_col2 = st.columns(2)
            
            feat_x = plot_col1.selectbox("Pilih Fitur Sumbu X", features_for_scaling, index=0, key="ikm_x")
            feat_y_index = 1 if len(features_for_scaling) > 1 else 0
            feat_y = plot_col2.selectbox("Pilih Fitur Sumbu Y", features_for_scaling, index=feat_y_index, key="ikm_y")
            
            # data_clustering yang dikirim ke plot sekarang sudah berisi nilai ternormalisasi
            plot_kmeans_results(data_clustering, final_labels, final_centroids_scaled, feat_x, feat_y, scaler, features_for_scaling)

            st.subheader("Data dengan Hasil Cluster (Nilai Ternormalisasi)")
            
            # --- PERUBAHAN 3 DI SINI: Mengatur urutan kolom ---
            column_order = ["Nama Wilayah", "Cluster"] + features_for_scaling
            st.dataframe(data_clustering[column_order], use_container_width=True)
            # --- AKHIR PERUBAHAN 3 ---

else:
    st.info("☝️ Silakan pilih atau unggah dataset Anda untuk memulai analisis.")