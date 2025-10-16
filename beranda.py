import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =================================================================================
st.set_page_config(
    page_title="Analisis Klaster Interaktif dengan DBSCAN",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set opsi untuk menonaktifkan peringatan Pyplot global
# st.set_option('deprecation.showPyplotGlobalUse', False)

# =================================================================================
# FUNGSI-FUNGSI UTAMA
# =================================================================================

def run_dbscan(data_scaled, eps, min_samples):
    """Menjalankan algoritma DBSCAN dan mengembalikan model dan label cluster."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_scaled)
    return dbscan, clusters

def calculate_silhouette(data_scaled, clusters):
    """Menghitung Silhouette Score jika memungkinkan."""
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    if n_clusters > 1:
        avg_score = silhouette_score(data_scaled, clusters)
        sample_scores = silhouette_samples(data_scaled, clusters)
        return avg_score, sample_scores, n_clusters
    else:
        return None, None, n_clusters

def plot_dbscan_results(data, features_scaled, dbscan_model, clusters, feature_x, feature_y):
    """Membuat dan menampilkan plot hasil clustering DBSCAN yang detail."""
    core_sample_indices = dbscan_model.core_sample_indices_
    is_core = np.zeros(len(features_scaled), dtype=bool)
    is_core[core_sample_indices] = True
    is_noise = (clusters == -1)
    is_border = ~(is_core | is_noise)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot Core Points
    if np.any(is_core):
        ax.scatter(data.loc[is_core, feature_x], data.loc[is_core, feature_y],
                   s=100, c=clusters[is_core], cmap='viridis', edgecolor='k', label='Core Points', alpha=0.8)

    # Plot Border Points
    if np.any(is_border):
        ax.scatter(data.loc[is_border, feature_x], data.loc[is_border, feature_y],
                   s=40, c=clusters[is_border], cmap='viridis', edgecolor='k', label='Border Points', alpha=0.8)

    # Plot Noise Points
    if np.any(is_noise):
        ax.scatter(data.loc[is_noise, feature_x], data.loc[is_noise, feature_y],
                   s=50, c='black', marker='x', label='Noise')

    ax.set_title('Visualisasi Hasil Clustering DBSCAN', fontsize=16)
    ax.set_xlabel(feature_x, fontsize=12)
    ax.set_ylabel(feature_y, fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)


# =================================================================================
# TAMPILAN ANTARMUKA STREAMLIT
# =================================================================================

# --- HEADER ---
st.title("🔬 Analisis Klaster Interaktif dengan DBSCAN")
st.write("""
Aplikasi ini memungkinkan Anda untuk melakukan clustering pada dataset menggunakan algoritma **DBSCAN**. 
Unggah file Excel Anda, atur parameter `epsilon` dan `min_samples`, dan lihat hasilnya secara visual.
""")

# --- SIDEBAR: INPUT PENGGUNA ---
with st.sidebar:
    st.header("⚙️ Pengaturan Analisis")

    # 1. File Uploader
    uploaded_file = st.file_uploader(
        "Pilih file Excel dari folder Anda",
        type=['xlsx'],
        help="Pastikan file Anda memiliki format yang benar. Kolom pertama akan diabaikan (dianggap label/ID) dan sisanya dianggap sebagai fitur numerik."
    )

    # 2. Input Parameter DBSCAN
    st.subheader("Parameter DBSCAN")
    eps_value = st.slider(
        "Epsilon (eps): Jarak maksimum antar titik",
        min_value=0.01, max_value=2.0, value=0.1, step=0.01,
        help="Menentukan seberapa dekat titik harus berada untuk dianggap sebagai tetangga. Nilai kecil akan menciptakan cluster yang lebih padat."
    )

    min_samples_value = st.slider(
        "Minimum Samples (min_pts): Jumlah minimum titik dalam lingkungan",
        min_value=2, max_value=50, value=8, step=1,
        help="Jumlah minimum titik yang dibutuhkan dalam radius `eps` untuk membentuk sebuah *core point*."
    )

# --- AREA UTAMA: PEMROSESAN DAN TAMPILAN HASIL ---
if uploaded_file is not None:
    try:
        # Memuat dan memproses data
        data = pd.read_excel(uploaded_file)
        st.header("1. Pratinjau Dataset")
        st.write(f"Dataset berhasil dimuat. **Shape: {data.shape}**")
        st.dataframe(data.head())

        # Pisahkan fitur dan normalisasi
        if data.shape[1] < 2:
            st.error("Error: Dataset harus memiliki setidaknya dua kolom (ID/Label dan setidaknya satu fitur).")
        else:
            features = data.iloc[:, 1:]
            
            # Cek jika semua fitur adalah numerik
            if not all(features.dtypes.apply(pd.api.types.is_numeric_dtype)):
                st.error("Error: Semua kolom fitur harus bertipe numerik. Harap bersihkan data Anda terlebih dahulu.")
            else:
                scaler = MinMaxScaler()
                features_scaled = scaler.fit_transform(features)

                # Menjalankan Analisis
                dbscan_model, clusters = run_dbscan(features_scaled, eps_value, min_samples_value)
                avg_score, sample_scores, n_clusters = calculate_silhouette(features_scaled, clusters)

                data['Cluster'] = clusters
                if sample_scores is not None:
                    data['Silhouette_Score'] = sample_scores

                st.markdown("---")
                st.header("2. Ringkasan Hasil Clustering")

                # Tampilkan metrik utama
                col1, col2, col3 = st.columns(3)
                col1.metric("Jumlah Klaster Ditemukan", n_clusters)
                col2.metric("Jumlah Titik Noise/Outlier", list(clusters).count(-1))
                if avg_score is not None:
                    col3.metric("Silhouette Score Rata-rata", f"{avg_score:.4f}")
                else:
                    col3.metric("Silhouette Score Rata-rata", "N/A (butuh > 1 klaster)")

                st.markdown("---")
                st.header("3. Visualisasi Hasil")
                
                # Pilihan fitur untuk plot
                feature_options = features.columns.tolist()
                col_x, col_y = st.columns(2)
                with col_x:
                    feature_x = st.selectbox("Pilih Fitur untuk Sumbu X:", options=feature_options, index=0)
                with col_y:
                    # Pastikan index default tidak sama jika memungkinkan
                    default_y_index = 1 if len(feature_options) > 1 else 0
                    feature_y = st.selectbox("Pilih Fitur untuk Sumbu Y:", options=feature_options, index=default_y_index)

                if feature_x == feature_y:
                    st.warning("Peringatan: Anda memilih fitur yang sama untuk sumbu X dan Y.")
                
                # Tampilkan plot utama
                plot_dbscan_results(data, features_scaled, dbscan_model, clusters, feature_x, feature_y)
                
                # Visualisasi Tambahan
                st.subheader("Visualisasi Tambahan")
                tab1, tab2 = st.tabs(["Distribusi Klaster", "Distribusi Silhouette Score"])

                with tab1:
                    fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
                    cluster_counts = data['Cluster'].value_counts().sort_index()
                    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax_dist, palette="viridis")
                    ax_dist.set_title("Jumlah Titik Data per Klaster")
                    ax_dist.set_xlabel("ID Klaster (-1 adalah Noise)")
                    ax_dist.set_ylabel("Jumlah Titik")
                    st.pyplot(fig_dist)

                with tab2:
                    if avg_score is not None:
                        fig_sil, ax_sil = plt.subplots(figsize=(10, 5))
                        sns.boxplot(x='Cluster', y='Silhouette_Score', data=data[data['Cluster']!=-1], ax=ax_sil, palette="viridis")
                        ax_sil.set_title("Distribusi Silhouette Score per Klaster")
                        ax_sil.set_xlabel("ID Klaster")
                        ax_sil.set_ylabel("Silhouette Score")
                        st.pyplot(fig_sil)
                    else:
                        st.info("Silhouette Score tidak dapat divisualisasikan karena hanya ditemukan 1 klaster atau kurang.")


                st.markdown("---")
                st.header("4. Detail Data Hasil Clustering")
                st.write("Tabel di bawah ini berisi data asli ditambah dengan kolom 'Cluster' dan 'Silhouette_Score' yang dihasilkan.")
                st.dataframe(data)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.warning("Pastikan file Excel Anda tidak kosong dan formatnya benar.")

else:
    st.info("Silakan unggah file Excel untuk memulai analisis.")