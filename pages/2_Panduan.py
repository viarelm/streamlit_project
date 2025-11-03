import streamlit as st

st.set_page_config(
    page_title="Panduan Aplikasi",
    page_icon="📖",
    layout='wide',
)

st.title("📖 Panduan Aplikasi")
st.write("Selamat datang! Halaman ini akan memandu Anda dalam menggunakan aplikasi pengelompokan komponen IPM.")

st.divider()

st.header("1. Penjelasan Umum Aplikasi")

st.markdown("""
Aplikasi ini dirancang untuk menganalisis dan mengelompokkan data Indeks Pembangunan Manusia (IPM) di seluruh kabupaten dan kota di Indonesia. Tujuannya adalah untuk memberikan visualisasi pengelompokan mengenai pola pembangunan manusia di tingkat daerah.

**Sumber Dataset**
Dataset yang digunakan dalam aplikasi ini bersumber dari [Badan Pusat Statistik Indonesia (BPS)](https://www.bps.go.id/id) dengan data Indeks Pembangunan Manusia (IPM), Angka Harapan Hidup (AHH), Pengeluaran Per Kapita (PKP).

**Metodologi:**
Aplikasi ini menerapkan dua algoritma *clustering* untuk mengelompokkan daerah berdasarkan komponen IPM:
1.  **Intelligent K-Means:** Sebuah variasi dari algoritma K-Means yang bertujuan untuk menemukan pusat cluster awal yang lebih baik, sehingga dapat menghasilkan pengelompokan yang lebih optimal dan menentukan jumlah klaster dari data itu sendiri.
2.  **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Algoritma yang mengelompokkan data berdasarkan kepadatan titik data, yang efektif dalam menemukan cluster dengan bentuk arbitrer dan mengidentifikasi data *outliers*.

Dataset yang diunggah akan melalui proses **standardisasi** menggunakan *MinMaxScaler* sebelum diolah oleh algoritma untuk memastikan setiap komponen memiliki skala yang sebanding.
""")

st.divider()

st.header("2. Format Dataset dan Template")

st.markdown("""
Untuk menggunakan fitur pengelompokan, Anda perlu mengunggah file dataset dengan format `.xlsx`. Pastikan dataset Anda memiliki kolom-kolom berikut dengan nama yang **sesuai persis**:

-   `Label`: Nama Kabupaten atau Kota.
-   `IPM_L`: Indeks Pembangunan Manusia Laki-Laki.
-   `IPM_P`: HIndeks Pembangunan Manusia Perempuan.
-   `AHH_L`: Angka Harapan Hidup Laki-Laki.
-   `AHH_P`: Angka Harapan Hidup Perempuan.
-   `PKP_L`: Pengeluaran Per Kapita Laki-Laki.
-   `PKP_P`: Pengeluaran Per Kapita Perempuan.
-   `Tahun`: Tahun Data.

Anda dapat mengunduh template dataset di bawah ini untuk memastikan format yang benar.
""")

@st.cache_data
def get_file_as_bytes(file_path):
    try:
        with open(file_path, "rb") as file:
            return file.read()
    except FileNotFoundError:
        st.error(f"File tidak ditemukan di path: {file_path}")
        return None

if template_bytes := get_file_as_bytes("assets/template_dataset.xlsx"):
    st.download_button(
        label="Unduh Template Dataset (.xlsx)",
        data=template_bytes,
        file_name='template_dataset.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

st.divider()

st.header("3. Unduh Buku Panduan")

st.markdown("""
Untuk panduan yang lebih detail mengenai setiap fitur dan cara penggunaan, silakan unduh buku panduan lengkap dalam format PDF melalui tautan di bawah ini.
""")

if pdf_bytes := get_file_as_bytes("assets/buku_panduan.pdf"):
    st.download_button(
        label="Unduh Buku Panduan (.pdf)",
        data=pdf_bytes,
        file_name="buku_panduan.pdf",
        mime="application/octet-stream"
    )