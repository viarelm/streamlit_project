# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import os
import json
import folium
from streamlit_folium import st_folium
import io
# Impor fungsi-fungsi yang telah dipisah
from utils import load_data, get_template_file 
from clustering_algorithms import run_dbscan, run_intelligent_kmeans

# --- TAMBAHAN: IMPOR UNTUK VISUALISASI TOOLKIT ---
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# --------------------------------------------------


# =================================================================================
# --- FUNGSI TOOLKIT ANALISIS & VISUALISASI ---
# =================================================================================

# Skema Label Cluster & Deskripsi
skema_label = {
    2:  ["Sejahtera", "Tertinggal"],
    3:  ["Sejahtera", "Menengah", "Tertinggal"],
    4:  ["Sejahtera", "Menengah Atas", "Menengah Bawah", "Tertinggal"],
    5:  ["Sejahtera", "Menengah Atas", "Menengah", "Menengah Bawah", "Tertinggal"],
    6:  ["Sejahtera", "Cukup Sejahtera", "Menengah Atas", "Menengah Bawah", "Rentan", "Tertinggal"],
    7:  ["Sejahtera", "Cukup Sejahtera", "Menengah Atas", "Menengah", "Menengah Bawah", "Rentan", "Tertinggal"],
    8:  ["Sejahtera", "Cukup Sejahtera", "Menengah Atas", "Menengah", "Menengah Bawah", "Cukup Rentan", "Rentan", "Tertinggal"],
    9:  ["Sejahtera", "Cukup Sejahtera", "Menengah Atas", "Menengah", "Menengah Bawah", "Cukup Rentan", "Rentan", "Sangat Rentan", "Tertinggal"],
    10: ["Sejahtera", "Cukup Sejahtera", "Menengah Atas", "Menengah", "Menengah Bawah", "Cukup Rentan", "Rentan", "Sangat Rentan", "Tertinggal", "Tertinggal Berat"],
    11: ["Sejahtera Tinggi", "Sejahtera", "Cukup Sejahtera", "Menengah Atas", "Menengah", "Menengah Bawah", "Cukup Rentan", "Rentan", "Sangat Rentan", "Tertinggal", "Tertinggal Berat"],
}

# ... (setelah blok `skema_label` Anda) ...

@st.cache_data
def load_geojson(path):
    """Memuat file GeoJSON dari path yang diberikan."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File GeoJSON '{path}' tidak ditemukan. Pastikan file ini ada di folder yang sama dengan aplikasi Anda.")
        return None

def create_cluster_map(df_clustered: pd.DataFrame, geojson_data: dict, base_features: list):
    """
    Membuat peta Folium berdasarkan hasil clustering dengan tooltip rata-rata.
    """
    map_center = [-2.5, 118.0]
    m = folium.Map(location=map_center, zoom_start=5, tiles="cartodbpositron")

    # Palet warna dinamis (Anda bisa ganti "viridis" dengan palet lain)
    unique_labels = sorted([label for label in df_clustered['Cluster_Label'].unique() if 'Noise' not in str(label)])
    colors = sns.color_palette("viridis", len(unique_labels)).as_hex()
    color_map = dict(zip(unique_labels, colors))

    # --- LOGIKA BARU: Hitung rata-rata dan gabungkan data untuk tooltip ---
    geojson_data_copy = geojson_data.copy()
    data_dict = df_clustered.set_index('Nama Wilayah').to_dict('index')

    for feature in geojson_data_copy['features']:
        nama_wilayah = feature['properties']['NAME_2']
        if nama_wilayah in data_dict:
            wilayah_data = data_dict[nama_wilayah]
            feature['properties']['Cluster_Label'] = wilayah_data.get('Cluster_Label', 'N/A')

            # Iterasi melalui fitur dasar yang dipilih (misal: 'AHH', 'RLS')
            for base_feat in base_features:
                # Temukan semua kolom yang berhubungan (misal: 'AHH_L_2024', 'AHH_P_2024')
                related_cols = [col for col in df_clustered.columns if col.startswith(base_feat)]
                
                # Ambil semua nilai numerik dari kolom-kolom tersebut untuk wilayah ini
                values = [
                    wilayah_data.get(col) for col in related_cols 
                    if pd.notna(wilayah_data.get(col)) and isinstance(wilayah_data.get(col), (int, float))
                ]
                
                # Hitung rata-ratanya jika ada data
                if values:
                    average_value = sum(values) / len(values)
                    # Tambahkan properti baru untuk rata-rata
                    feature['properties'][f'Rata-rata {base_feat}'] = f"{average_value:.2f}"
    # --- AKHIR LOGIKA BARU ---

    # ---------------------------------------------------

    # Fungsi untuk mewarnai setiap wilayah pada peta
    def style_function(feature):
        cluster_label = feature['properties'].get('Cluster_Label', None)
        
        fill_color = '#D3D3D3' # Default untuk data yang tidak ada/noise
        fill_opacity = 0.3
        
        if cluster_label and 'Noise' not in str(cluster_label):
            fill_color = color_map.get(cluster_label, '#808080')
            fill_opacity = 0.7
            
        return {
            'fillColor': fill_color,
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': fill_opacity
        }

    # --- PERUBAHAN 3: PERBARUI FIELDS DAN ALIASES DI TOOLTIP ---
    # Dapatkan nama kolom fitur terbaru untuk ditampilkan
    fields_to_show = ['NAME_2', 'Cluster_Label'] + [f'Rata-rata {feat}' for feat in base_features]
    aliases_to_show = ['Wilayah:', 'Cluster:'] + [f'Rata-rata {feat}:' for feat in base_features]
    
    # Pastikan hanya menampilkan field yang ada
    available_fields = [f for f in fields_to_show if f in geojson_data_copy['features'][0]['properties']]
    available_aliases = [aliases_to_show[fields_to_show.index(f)] for f in available_fields]

    folium.GeoJson(
        geojson_data_copy,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=available_fields,
            aliases=available_aliases,
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)
    # --------------------------------------------------------------

    # Buat legenda HTML kustom (tidak perlu diubah)
    legend_html = '''
        <div style="position: fixed; 
        bottom: 50px; left: 50px; width: auto; height: auto; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:white; color: black; padding: 10px; border-radius: 5px;">
        <b>Legenda Cluster</b><br>
        '''
    for label, color in color_map.items():
        if 'Noise' not in str(label):
            legend_html += f'<i style="background:{color}; width:20px; height:20px; float:left; margin-right:8px; border:1px solid grey;"></i> {label}<br>'
    
    legend_html += '<i style="background:#D3D3D3; width:20px; height:20px; float:left; margin-right:8px; border:1px solid grey;"></i> Noise / Outlier<br>'
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def ambil_skema_label(jumlah_k: int):
    """Ambil daftar label cluster berdasarkan jumlah K."""
    return skema_label.get(jumlah_k, [f"Cluster {i+1}" for i in range(jumlah_k)])


# =================================================================================
# --- FUNGSI TOOLKIT ANALISIS & VISUALISASI ---
# =================================================================================

# Ringkasan Cluster (FUNGSI YANG DIPERBAIKI)
# =================================================================================
# --- FUNGSI TOOLKIT ANALISIS & VISUALISASI ---
# =================================================================================

# ... (Fungsi skema_label, ambil_skema_label, analisis_cluster Anda tetap di sini) ...


# Ringkasan Cluster (FUNGSI YANG DIPERBAIKI)
# Ganti fungsi ringkasan_cluster Anda dengan yang ini

def ringkasan_cluster(df: pd.DataFrame, judul: str = "Ringkasan Cluster"):
    s = df["Cluster"]
    hitung = s.value_counts().sort_index()
    k = len(hitung)
    total = hitung.sum()

    ringkasan = pd.DataFrame({
        "Cluster": hitung.index,
        "Jumlah": hitung.values,
        "Persen": (hitung.values / total * 100).round(1)
    })

    fig, ax = plt.subplots(figsize=(4, 3)) # Sedikit menambah tinggi figure secara keseluruhan
    warna = plt.cm.Blues(np.linspace(0.4, 0.8, k))
    
    ringkasan = ringkasan.sort_values(by="Cluster")
    
    bars = ax.bar(ringkasan["Cluster"].astype(str), ringkasan["Jumlah"], color=warna)

    # --- PERBAIKAN UTAMA ADA DI SINI ---
    # Tambahkan margin atas sebesar 20% (sebelumnya 10%)
    # Ini memberikan ruang lebih bagi teks di atas bar tertinggi.
    ax.margins(y=0.2) 
    # ------------------------------------

    for bar, v, p in zip(bars, ringkasan["Jumlah"], ringkasan["Persen"]):
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height(), # Letakkan teks tepat di atas bar
            f"{v}\n({p}%)", 
            ha="center", 
            va="bottom", # Mulai teks dari atas bar ke atas
            fontsize=8
        )

    ax.set_title(f"{judul} (K={k})", fontsize=12, pad=15)
    ax.set_xlabel("Cluster", fontsize=10)
    ax.set_ylabel("Jumlah Wilayah", fontsize=10)
    
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.85, bottom=0.2) 
    return ringkasan, fig


# ... (Fungsi visualisasi_silhouette_full dan display_clustering_results Anda tetap di sini) ...
# --- TAMBAHAN BARU: FUNGSI PLOT SILHOUETTE ---
def visualisasi_silhouette_full(data_matriks: np.ndarray, label_cluster: np.ndarray, label_map: dict, algo: str = ""):
    """
    Membuat plot silhouette untuk semua sampel.
    data_matriks: Data yang sudah ternormalisasi (array numpy).
    label_cluster: Array label numerik hasil cluster.
    label_map: Kamus untuk memetakan label numerik ke label deskriptif.
    """
    nilai_sample = silhouette_samples(data_matriks, label_cluster)
    nilai_rata   = silhouette_score(data_matriks, label_cluster)

    n_clusters = len(np.unique(label_cluster))
    y_bawah = 5
    
    fig, ax1 = plt.subplots(figsize=(4, 3))

    for i in sorted(np.unique(label_cluster)):
        nilai_i = nilai_sample[label_cluster == i]
        nilai_i.sort()

        ukuran_i = nilai_i.shape[0]
        y_atas = y_bawah + ukuran_i

        # Dapatkan label deskriptif dari map, gunakan label numerik jika tidak ada
        nama_label = label_map.get(i, str(i))

        warna = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_bawah, y_atas),
            0,
            nilai_i,
            facecolor=warna,
            edgecolor=warna,
            alpha=0.7
        )

        # --- PERUBAHAN UTAMA: Gunakan nama_label untuk teks ---
        ax1.text(1.02, y_bawah + 0.5 * ukuran_i, nama_label, fontsize=8, ha='left', va='center')        # ----------------------------------------------------
        y_bawah = y_atas + 10

    ax1.set_title(f"Plot Silhouette ({algo})", fontsize=11, pad=10) 
    ax1.set_xlabel("Nilai Silhouette Coefficient", fontsize=10)
    ax1.set_ylabel("Cluster", fontsize=10)

    ax1.axvline(x=nilai_rata, color="red", linestyle="--", linewidth=1.5)
    ax1.text(
        nilai_rata + 0.01, 
        y_bawah * 0.01,
        f"Rata-rata: {nilai_rata:.2f}", 
        color="red", 
        fontsize=9, 
        ha="left", 
        va="bottom"
    )

    ax1.set_yticks([])
    ax1.set_xticks(np.linspace(-0.1, 1.0, 6))
    ax1.tick_params(axis="both", labelsize=9)

    plt.tight_layout(pad=1)
    return fig
# Analisis Cluster
# Analisis Cluster (DIPERBARUI UNTUK DBSCAN)
def analisis_cluster(df: pd.DataFrame, fitur_digunakan, algoritma: str = ""):
    fitur_semua = fitur_digunakan
    fitur_negatif_heuristik = [f for f in fitur_semua if any(kata in f.upper() for kata in ["MISKIN", "P0", "P1", "P2"])]
    fitur_positif_heuristik = [f for f in fitur_semua if f not in fitur_negatif_heuristik]

    st.info(f"""
    **Heuristik Fitur untuk Penentuan Skor:**
    - **Positif (Lebih tinggi lebih baik):** {fitur_positif_heuristik or ['Tidak ada']}
    - **Negatif (Lebih rendah lebih baik):** {fitur_negatif_heuristik or ['Tidak ada']}
    """)

    rata_c = df.groupby("Cluster")[fitur_semua].mean().round(3)

    # --- PERUBAHAN UNTUK DBSCAN ---
    # Jangan buat label deskriptif jika ini DBSCAN
    if algoritma.upper() == "DBSCAN":
        label_cluster = {i: f"Cluster {i}" for i in rata_c.index}
        skor = None # Skor tidak relevan untuk DBSCAN
    
    # Logika K-Means tetap sama
    else:
        skor = pd.Series(index=rata_c.index, dtype=float, data=0.0)
        if fitur_positif_heuristik:
            skor += rata_c[fitur_positif_heuristik].mean(axis=1)
        if fitur_negatif_heuristik:
            skor -= rata_c[fitur_negatif_heuristik].mean(axis=1)
        
        if not fitur_positif_heuristik and not fitur_negatif_heuristik and not skor.any():
            skor = rata_c[fitur_semua].mean(axis=1) 

        ranking = skor.sort_values(ascending=False)
        urutan  = ranking.index.tolist()
        k       = len(ranking)

        skema          = ambil_skema_label(k)
        label_cluster  = {urutan[i]: skema[i] for i in range(k)}
    # --- AKHIR PERUBAHAN ---
    
    return rata_c, label_cluster, skor

# Ringkasan Cluster
# --- FUNGSI TAMPILAN HASIL (PERBAIKAN + SILHOUETTE LANGSUNG) ---
# --- FUNGSI TAMPILAN HASIL (PERBAIKAN PLOT SEBARAN + SILHOUETTE LANGSUNG) ---
def display_clustering_results(
    data_original,
    data_clustering,
    features_for_scaling,
    fitur_terpilih_base,
    selected_years,
    method_name,
    plot_args
):
    """
    Fungsi terpusat untuk menampilkan SEMUA visualisasi hasil, dengan peta di bagian bawah.
    """

    # --- Visualisasi Sebaran Data Ternormalisasi ---
    st.subheader("Visualisasi Sebaran (Data Ternormalisasi)")
    plot_col1, plot_col2 = st.columns(2)
    feat_x = plot_col1.selectbox("Pilih Fitur Sumbu X", features_for_scaling, index=0, key=f"{method_name}_x")
    feat_y_index = 1 if len(features_for_scaling) > 1 else 0
    feat_y = plot_col2.selectbox("Pilih Fitur Sumbu Y", features_for_scaling, index=feat_y_index, key=f"{method_name}_y")

    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 5))
    plot_data = data_clustering.copy()
    plot_data['Cluster_Label'] = data_original['Cluster_Label'] # Pastikan kolom ini ada

    sns.scatterplot(data=plot_data, x=feat_x, y=feat_y, hue='Cluster_Label', palette="Set1", s=50, alpha=0.7, ax=ax_scatter)

    if method_name == "Intelligent K-Means" and 'centroids' in plot_args:
        centroids = plot_args['centroids']
        try:
            x_index = features_for_scaling.index(feat_x)
            y_index = features_for_scaling.index(feat_y)
            ax_scatter.scatter(centroids[:, x_index], centroids[:, y_index], marker='X', s=200, c='black', edgecolor='white', label='Centroids')
        except (ValueError, IndexError):
            st.warning("Gagal memplot centroid karena ketidaksesuaian fitur.")

    ax_scatter.set_title(f"Visualisasi Sebaran Cluster ({method_name})")
    ax_scatter.set_xlabel(f"{feat_x} (Ternormalisasi)")
    ax_scatter.set_ylabel(f"{feat_y} (Ternormalisasi)")
    ax_scatter.set_xlim(-0.1, 1.1)
    ax_scatter.set_ylim(-0.1, 1.1)
    ax_scatter.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    col_kiri, col_plot, col_kanan = st.columns([1, 3, 1])

    # Gunakan 'with' untuk menempatkan plot di kolom tengah (col_plot)
    with col_plot:
        st.pyplot(fig_scatter, use_container_width=True)

    # 1. Siapkan buffer di memori
    buf_png = io.BytesIO()
    # Simpan gambar ke buffer (gunakan bbox_inches='tight' agar legenda tidak terpotong)
    fig_scatter.savefig(buf_png, format="png", bbox_inches='tight')

    buf_pdf = io.BytesIO()
    fig_scatter.savefig(buf_pdf, format="pdf", bbox_inches='tight')


    st.markdown("""
    <style>
        [data-testid="stHorizontalBlock"] {
            gap: 0.25rem; 
        }
    </style>
    """, unsafe_allow_html=True)
    # 2. Buat kolom untuk tombol download
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        st.download_button(
            label="📥 Download as PNG",
            data=buf_png.getvalue(),  # Ambil data bytes dari buffer
            file_name=f"scatter_plot_{feat_x}_vs_{feat_y}.png",
            mime="image/png",
            use_container_width=True # Tambahkan ini agar tombol mengisi kolom
        )

    with dl_col2:
        st.download_button(
            label="📥 Download as PDF",
            data=buf_pdf.getvalue(), # Ambil data bytes dari buffer
            file_name=f"scatter_plot_{feat_x}_vs_{feat_y}.pdf",
            mime="application/pdf",
            use_container_width=True # Tambahkan ini agar tombol mengisi kolom
        )
    # --- AKHIR KODE TAMBAHAN ---
    
    # --- Visualisasi Silhouette Score ---
    st.markdown("---")
    st.subheader("Analisis Plot Silhouette")
    data_matriks = data_clustering[features_for_scaling].values
    labels = plot_args['labels']
    label_map = plot_args.get('label_map', {})
    n_clusters_unique = len(np.unique(labels))

    # --- PERUBAHAN DIMULAI DI SINI ---
    # 1. Inisialisasi fig_sil sebagai None
    fig_sil = None
    plot_title = "silhouette_score" # Untuk nama file

    if method_name == "DBSCAN":
        mask_non_noise = (labels != -1)
        if np.sum(mask_non_noise) > 0:
            labels_filtered = labels[mask_non_noise]
            n_clusters_filtered = len(np.unique(labels_filtered))
            if n_clusters_filtered > 1:
                st.write("Plot Silhouette (Tanpa Noise):")
                plot_title = f"{method_name} (K={n_clusters_filtered})"
                # 2. Buat figure, tapi JANGAN panggil st.pyplot dulu
                fig_sil = visualisasi_silhouette_full(data_matriks[mask_non_noise], labels_filtered, label_map, plot_title)
            else:
                st.info("Silhouette tidak ditampilkan (kurang dari 2 cluster tanpa noise).")
        else:
            st.info("Semua data adalah noise, Silhouette tidak dapat dibuat.")
    elif method_name == "Intelligent K-Means" and n_clusters_unique > 1:
        plot_title = f"{method_name} (K={n_clusters_unique})"
        # 2. Buat figure, tapi JANGAN panggil st.pyplot dulu
        fig_sil = visualisasi_silhouette_full(
                data_matriks, labels, label_map, plot_title
            )  
    else:
        st.info("Plot Silhouette tidak dapat ditampilkan (K=1).")


    # 3. Sekarang, cek apakah fig_sil BERHASIL dibuat
    if fig_sil:
        # 4. Jika berhasil, tampilkan plotnya
        col_kiri, col_plot, col_kanan = st.columns([1, 3, 1])
    
        with col_plot:
            # Tampilkan plot di dalam kolom tengah
            st.pyplot(fig_sil, use_container_width=True)

        # 5. Siapkan buffer untuk download
        buf_png = io.BytesIO()
        fig_sil.savefig(buf_png, format="png", bbox_inches='tight')

        buf_pdf = io.BytesIO()
        fig_sil.savefig(buf_pdf, format="pdf", bbox_inches='tight')

        # 6. Suntikkan CSS untuk merapatkan tombol
        st.markdown("""
        <style>
            [data-testid="stHorizontalBlock"] {
                gap: 0.25rem; 
            }
        </style>
        """, unsafe_allow_html=True)

        # 7. Buat tombol download
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📥 Download as PNG",
                data=buf_png.getvalue(),
                file_name=f"silhouette_plot_{method_name}.png",
                mime="image/png",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                label="📥 Download as PDF",
                data=buf_pdf.getvalue(),
                file_name=f"silhouette_plot_{method_name}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # --- Analisis Karakteristik Cluster ---
    st.subheader("Analisis Karakteristik Cluster")

    # --- Bar Chart Distribusi Anggota ---
    st.markdown("---")
    st.subheader("1. Distribusi Anggota Cluster (Bar Chart)")
    df_ringkasan_label = data_original[['Cluster_Label']].rename(columns={"Cluster_Label": "Cluster"})
    ringkasan_df, fig_ringkasan = ringkasan_cluster(df_ringkasan_label, f"Ringkasan Anggota Cluster ({method_name})")
    col_kiri, col_plot, col_kanan = st.columns([1, 3, 1])
    
    with col_plot:
        st.pyplot(fig_ringkasan) # ringkasan_cluster sudah memanggil st.pyplot

    buf_png = io.BytesIO()
    fig_ringkasan.savefig(buf_png, format="png", bbox_inches='tight')

    buf_pdf = io.BytesIO()
    fig_ringkasan.savefig(buf_pdf, format="pdf", bbox_inches='tight')

    # 4. Suntikkan CSS untuk merapatkan tombol
    st.markdown("""
    <style>
        [data-testid="stHorizontalBlock"] {
            gap: 0.25rem; 
        }
    </style>
    """, unsafe_allow_html=True)

    # 5. Buat tombol download
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="📥 Download as PNG",
            data=buf_png.getvalue(),
            file_name=f"ringkasan_cluster_{method_name}.png",
            mime="image/png",
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            label="📥 Download as PDF",
            data=buf_pdf.getvalue(),
            file_name=f"ringkasan_cluster_{method_name}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
    # --- Persiapan Data Long ---
    id_vars = ["Nama Wilayah", "Cluster_Label"]
    if 'Cluster_Num' in data_original.columns: id_vars.append('Cluster_Num')
    data_long = data_original.melt(id_vars=id_vars, value_vars=features_for_scaling, var_name="Fitur_Tahun", value_name="Nilai")
    split_data = data_long['Fitur_Tahun'].str.rsplit('_', n=1, expand=True)
    data_long['Fitur'] = split_data[0]
    data_long['Tahun'] = split_data[1].astype(int)

    # --- Boxplot per Fitur ---
    st.markdown("---")
    st.subheader("2. Distribusi Fitur per Cluster (Boxplot per Tahun)")
    base_feature_to_plot = st.selectbox("Pilih Fitur untuk Boxplot", fitur_terpilih_base, key=f"{method_name}_boxplot_feat")
    df_boxplot = data_long[data_long['Fitur'] == base_feature_to_plot]
    sorted_years = sorted(selected_years)
    cluster_order = sorted(df_boxplot['Cluster_Label'].unique())
    n_years, n_cols = len(sorted_years), min(len(sorted_years), 3)
    n_rows = (n_years + n_cols - 1) // n_cols

    # 1. Figure Anda dibuat di sini
    fig_box, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()
    for i, year in enumerate(sorted_years):
        ax = axes[i]
        sns.boxplot(data=df_boxplot[df_boxplot['Tahun'] == year], x='Cluster_Label', y='Nilai', hue='Cluster_Label', order=cluster_order, ax=ax, legend=False)
        ax.set_title(f"Tahun {year}"); ax.set_xlabel(None); ax.tick_params(axis='x', rotation=30)
    for j in range(n_years, len(axes)): fig_box.delaxes(axes[j])
    fig_box.suptitle(f"Distribusi {base_feature_to_plot} per Cluster", fontsize=16)
    fig_box.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 2. Tampilkan plot di Streamlit
  
    st.pyplot(fig_box)

    # --- MULAI KODE TAMBAHAN UNTUK DOWNLOAD ---

    # 3. Siapkan buffer untuk download
    buf_png = io.BytesIO()
    fig_box.savefig(buf_png, format="png", bbox_inches='tight')

    buf_pdf = io.BytesIO()
    fig_box.savefig(buf_pdf, format="pdf", bbox_inches='tight')

    # 4. Suntikkan CSS untuk merapatkan tombol
    st.markdown("""
    <style>
        [data-testid="stHorizontalBlock"] {
            gap: 0.25rem; 
        }
    </style>
    """, unsafe_allow_html=True)

    # 5. Buat tombol download
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="📥 Download as PNG",
            data=buf_png.getvalue(),
            file_name=f"boxplot_{base_feature_to_plot}_{method_name}.png",
            mime="image/png",
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            label="📥 Download as PDF",
            data=buf_pdf.getvalue(),
            file_name=f"boxplot_{base_feature_to_plot}_{method_name}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # --- Scatter Plot Perbandingan Tahun ---
    if len(selected_years) >= 2:
        st.markdown("---")
        st.subheader("3. Perbandingan Antar Tahun (per Fitur)")
        base_feature_scatter = st.selectbox("Pilih Fitur untuk Scatter Plot", fitur_terpilih_base, key=f"{method_name}_scatter_feat")
        col_scat1, col_scat2 = st.columns(2)
        year_x = col_scat1.selectbox("Pilih Tahun Sumbu X", selected_years, index=0, key=f"{method_name}_scat_x")
        year_y = col_scat2.selectbox("Pilih Tahun Sumbu Y", selected_years, index=len(selected_years)-1, key=f"{method_name}_scat_y")
        feat_x_scat, feat_y_scat = f"{base_feature_scatter}_{year_x}", f"{base_feature_scatter}_{year_y}"

        # 1. Figure Anda dibuat di sini
        fig_scatter_comp, ax_scatter_comp = plt.subplots(figsize=(4, 3))
        sns.scatterplot(data=data_original, x=feat_x_scat, y=feat_y_scat, hue='Cluster_Label', ax=ax_scatter_comp, palette="Set1", s=50, alpha=0.7)
        ax_scatter_comp.set_title(f"Perbandingan {base_feature_scatter}: {year_x} vs {year_y}")
        ax_scatter_comp.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. Tampilkan plot di Streamlit
        col_kiri, col_plot, col_kanan = st.columns([1, 3, 1])
    
        with col_plot:
            st.pyplot(fig_scatter_comp)

        # --- MULAI KODE TAMBAHAN UNTUK DOWNLOAD ---

        # 3. Siapkan buffer untuk download
        buf_png = io.BytesIO()
        # Gunakan bbox_inches='tight' agar legenda tidak terpotong
        fig_scatter_comp.savefig(buf_png, format="png", bbox_inches='tight')

        buf_pdf = io.BytesIO()
        fig_scatter_comp.savefig(buf_pdf, format="pdf", bbox_inches='tight')

        # 4. Suntikkan CSS untuk merapatkan tombol
        st.markdown("""
        <style>
            [data-testid="stHorizontalBlock"] {
                gap: 0.25rem; 
            }
        </style>
        """, unsafe_allow_html=True)

        # 5. Buat tombol download
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📥 Download as PNG",
                data=buf_png.getvalue(),
                file_name=f"perbandingan_{base_feature_scatter}_{year_x}_vs_{year_y}.png",
                mime="image/png",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                label="📥 Download as PDF",
                data=buf_pdf.getvalue(),
                file_name=f"perbandingan_{base_feature_scatter}_{year_x}_vs_{year_y}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # --- Tabel Data Hasil ---
    st.subheader("Data Asli dengan Hasil Cluster")
    column_order = ["Nama Wilayah", "Cluster_Label"] + features_for_scaling
    st.dataframe(data_original[column_order], use_container_width=True)

    # --- VISUALISASI PETA CLUSTERING (PINDAH KE SINI) ---
    st.markdown("---") # Tambahkan pemisah
    st.subheader("Peta Sebaran Cluster")
    geojson_data = load_geojson('indonesia_kabupaten.geojson')

    if geojson_data:
        # Panggil fungsi independen untuk membuat objek peta
        map_object = create_cluster_map(data_original, geojson_data, fitur_terpilih_base)        
        # Tampilkan peta di Streamlit
        st_folium(map_object, use_container_width=True, height=600)
        st.caption("Arahkan kursor ke wilayah untuk melihat namanya. Wilayah berwarna abu-abu tidak termasuk dalam analisis atau merupakan noise.")
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
        label_visibility="collapsed"
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
    "Pengeluaran Per Kapita Perempuan": "PKP_P"
}

# ... (Logika load_data dan download template sama) ...
if data_source_option == "Gunakan Contoh Dataset":
    dataset_path = "contoh_dataset/dataset.xlsx"
    if os.path.exists(dataset_path):
        data = load_data(dataset_path)
    else:
        st.error(f"Error: File dataset contoh '{dataset_path}' tidak ditemukan.")
else:
    template_data = get_template_file()
    if template_data:
        st.download_button(
        label="📥 Download Template Dataset (.xlsx)",
        data=template_data,
        file_name="assets/template_dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    uploaded_file = st.file_uploader("Unggah file Anda (.xlsx atau .csv)", type=['xlsx', 'csv'])
    if uploaded_file:
        data = load_data(uploaded_file)
       


if data is not None:
    if "Label" in data.columns:
        data = data.rename(columns={"Label": "Nama Wilayah"})

# --- 3. Filter & Persiapan Data ---
if data is not None:
    st.subheader("Filter Data dan Fitur")

    # ... (Validasi kolom sama) ...
    if "Tahun" not in data.columns:
        st.error("Dataset Error: Kolom 'Tahun' tidak ditemukan. Silakan periksa template.")
        st.stop()
    if "Nama Wilayah" not in data.columns:
        st.warning("Dataset Peringatan: Kolom 'Nama Wilayah' (atau 'label') tidak ditemukan.")

    available_years = sorted(data["Tahun"].unique())
    selected_years = st.multiselect("Pilih Tahun", available_years, default=available_years)
    
    available_features_map = {key: val for key, val in mapping_fitur.items() if val in data.columns}
    if not available_features_map:
         st.error(f"Dataset Error: Tidak ditemukan satupun fitur yang dikenali dari daftar: {list(mapping_fitur.values())}")
         st.stop()

    selected_feature_keys = st.multiselect(
        "Pilih Fitur untuk Clustering", 
        available_features_map.keys(), 
        default=list(available_features_map.keys())
    )
    
    fitur_terpilih_base = [available_features_map[key] for key in selected_feature_keys]

    if not selected_years or not fitur_terpilih_base:
        st.warning("Silakan pilih minimal satu tahun dan satu fitur.")
        st.stop()
    
    # ... (Logika pivot data sama) ...
    data_to_pivot = data[data["Tahun"].isin(selected_years)].copy()
    data_to_pivot = data_to_pivot[["Nama Wilayah", "Tahun"] + fitur_terpilih_base]

    st.subheader("Data yang Dipersiapkan untuk Clustering (Setelah Pivot)")
    try:
        data_pivoted = data_to_pivot.pivot(
            index="Nama Wilayah", 
            columns="Tahun", 
            values=fitur_terpilih_base
        )
    except ValueError as e:
        st.error(f"Error saat mem-pivot data: {e}")
        st.error("Ini biasanya terjadi jika ada duplikat 'Nama Wilayah' untuk 'Tahun' yang sama.")
        st.stop()
    
    data_pivoted.columns = [f"{val}_{year}" for val, year in data_pivoted.columns]
    data_clustering = data_pivoted.reset_index().copy()
    data_original = data_clustering.copy()

    identifier_cols = ["Nama Wilayah"]
    features_for_scaling = [col for col in data_clustering.columns if col not in identifier_cols]
    
    st.info(f"Data berhasil di-pivot. Jumlah wilayah: **{len(data_clustering)}**. Jumlah fitur: **{len(features_for_scaling)}** ({len(fitur_terpilih_base)} fitur x {len(selected_years)} tahun).")

    if data_clustering[features_for_scaling].isnull().values.any():
        nan_count = data_clustering[features_for_scaling].isnull().values.sum()
        st.warning(f"Data mengandung **{nan_count}** nilai NaN. Baris yang mengandung NaN akan dihapus.")
        data_clustering = data_clustering.dropna(subset=features_for_scaling).reset_index(drop=True)
        data_original = data_original.dropna(subset=features_for_scaling).reset_index(drop=True)
    
    if data_clustering.empty:
        st.error("Tidak ada data tersisa setelah filtering/pivot.")
        st.stop()
        
    data_clustering.index = data_clustering.index + 1
    data_original.index = data_original.index + 1
    
    features_to_scale_df = data_clustering[features_for_scaling]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_to_scale_df)
    
    data_clustering[features_for_scaling] = features_scaled
    st.success("Data telah dinormalisasi menggunakan MinMaxScaler (skala 0-1).")
    
    st.write("Data ternormalisasi yang siap di-cluster (Pratinjau):")
    st.dataframe(data_clustering.head(), use_container_width=True)

    # --- 4. Parameter & Eksekusi (HANYA KOMPUTASI & PENYIMPANAN STATE) ---
    st.header("3. Parameter & Eksekusi")
    
    if selected_method == "DBSCAN":
        st.subheader("Parameter DBSCAN")
        col1_param, col2_param = st.columns(2)
        eps_value = col1_param.slider("Epsilon (eps)", 0.01, 2.0, 0.5, 0.01)
        min_samples_value = col2_param.slider("Minimum Samples (min_pts)", 2, 50, 5, 1)
        
        if st.button("🚀 Jalankan Analisis DBSCAN", type="primary"):
            st.header("4. Hasil Analisis DBSCAN")
            with st.spinner("Menjalankan DBSCAN..."):
                dbscan, clusters = run_dbscan(features_scaled, eps_value, min_samples_value)
                
                data_clustering['Cluster_Num'] = clusters
                data_original['Cluster_Num'] = clusters
                
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                n_noise = list(clusters).count(-1)
                
                score = "N/A"
                if n_clusters > 1:
                    mask_non_noise = clusters != -1
                    score = silhouette_score(features_scaled[mask_non_noise], clusters[mask_non_noise])
                
                data_original_no_noise = data_original[data_original['Cluster_Num'] != -1].copy()
                # Ganti nama 'Cluster_Num' -> 'Cluster' untuk groupby
                data_original_no_noise = data_original_no_noise.rename(columns={'Cluster_Num': 'Cluster'})

                if n_clusters > 0:
                    # Panggil analisis_cluster HANYA untuk mendapatkan rata_c
                    # Kita akan mengabaikan 'label_cluster' yang dihasilkannya
                    rata_c, _, _ = analisis_cluster(
                        data_original_no_noise, 
                        features_for_scaling, 
                        "DBSCAN" # Kirim nama algoritma
                    )
                    st.write("Rata-rata Indikator per Cluster (Tanpa Noise):")
                    st.dataframe(rata_c, use_container_width=True)
                
                # --- PERUBAHAN LABEL DBSCAN DI SINI ---
                # Buat 'Cluster_Label' secara manual, bukan dari analisis_cluster
                data_original['Cluster_Label'] = data_original['Cluster_Num'].apply(
                    lambda x: "Noise (-1)" if x == -1 else f"Cluster {x+1}"
                )
                
                unique_labels_dbscan = np.unique(clusters)
                    # Balikkan proses lambda untuk membuat kamus: {nomor: "Label Deskriptif"}
                label_map_dbscan = {
                    label: "Noise / Outlier" if label == -1 else f"Cluster {label + 1}"
                    for label in unique_labels_dbscan
                }
                # --- AKHIR PERUBAHAN ---

                # --- SIMPAN HASIL KE SESSION STATE ---
                st.session_state.clustering_complete = True
                st.session_state.method_name = "DBSCAN"
                st.session_state.data_original = data_original.copy()
                st.session_state.data_clustering = data_clustering.copy()
                st.session_state.features_for_scaling = features_for_scaling
                st.session_state.fitur_terpilih_base = fitur_terpilih_base
                st.session_state.selected_years = selected_years
                st.session_state.metrics = {'n_clusters': n_clusters, 'n_noise': n_noise, 'score': score}
                st.session_state.plot_args = {'labels': clusters, 'dbscan_model': dbscan, 'label_map': label_map_dbscan}                
                st.rerun()

    elif selected_method == "Intelligent K-Means":
        st.info("Metode ini secara otomatis menentukan jumlah klaster (K) yang optimal.")
        
        if st.button("🚀 Jalankan Analisis Intelligent K-Means", type="primary"):
            st.header("4. Hasil Analisis Intelligent K-Means")
            st.subheader("Log Proses Real-time")
            
            # Ini (run_intelligent_kmeans) sudah ada st.spinner/logging di dalamnya
            final_labels, final_centroids_scaled, final_k = run_intelligent_kmeans(features_scaled, features_for_scaling)
            
            st.success("🎉 Analisis Selesai!")
            st.markdown("---")
            
            data_clustering['Cluster'] = final_labels + 1
            data_original['Cluster'] = final_labels + 1
            
            score = silhouette_score(features_scaled, final_labels)

            rata_c, label_cluster, skor_analisis = analisis_cluster(
                data_original, 
                features_for_scaling, 
                "Intelligent K-Means"
            )
            data_original['Cluster_Label'] = data_original['Cluster'].map(label_cluster)
            label_map_for_silhouette = {k - 1: v for k, v in label_cluster.items()}
            # --- SIMPAN HASIL KE SESSION STATE ---
            st.session_state.clustering_complete = True
            st.session_state.method_name = "Intelligent K-Means"
            st.session_state.data_original = data_original.copy()
            st.session_state.data_clustering = data_clustering.copy()
            st.session_state.features_for_scaling = features_for_scaling
            st.session_state.fitur_terpilih_base = fitur_terpilih_base
            st.session_state.selected_years = selected_years
            st.session_state.metrics = {'k': final_k, 'score': score}
            st.session_state.plot_args = {
                'labels': final_labels, 
                'centroids': final_centroids_scaled, 
                'scaler': scaler,
                'label_map': label_map_for_silhouette
            }
            
            st.rerun() # Paksa script run ulang untuk masuk ke blok display

    # --- 5. BLOK TAMPILAN HASIL (MEMBACA DARI SESSION STATE) ---
    if 'clustering_complete' in st.session_state and st.session_state.clustering_complete:
        
        # Jika user ganti metode, jangan tampilkan hasil lama
        if st.session_state.method_name != selected_method:
            st.warning(f"Anda mengganti metode ke {selected_method}. Harap jalankan ulang analisis.")
            # Hapus state lama
            for key in list(st.session_state.keys()):
                if key.startswith('clustering_') or key.startswith('data_') or key.startswith('metrics_'):
                    del st.session_state[key]
        else:
            # --- Tampilkan Metrik ---
            st.header(f"4. Hasil Analisis {st.session_state.method_name}")
            metrics = st.session_state.metrics
            
            if st.session_state.method_name == "DBSCAN":
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Jumlah Klaster", metrics['n_clusters'])
                res_col2.metric("Jumlah Noise/Outlier", metrics['n_noise'])
                res_col3.metric("Silhouette Score (tanpa noise)", f"{metrics['score']:.3f}" if isinstance(metrics['score'], float) else "N/A")
            else:
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Jumlah Klaster Optimal (K)", metrics['k'])
                res_col2.metric("Silhouette Score Final", f"{metrics['score']:.3f}")

            # --- Panggil Fungsi Display Terpusat ---
            display_clustering_results(
                data_original=st.session_state.data_original,
                data_clustering=st.session_state.data_clustering,
                features_for_scaling=st.session_state.features_for_scaling,
                fitur_terpilih_base=st.session_state.fitur_terpilih_base,
                selected_years=st.session_state.selected_years,
                method_name=st.session_state.method_name,
                plot_args=st.session_state.plot_args
            )

else:
    st.info("☝️ Silakan pilih atau unggah dataset Anda untuk memulai analisis.")