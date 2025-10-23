import streamlit as st 
from utils import show_footer

st.set_page_config(
    page_title="Beranda",
    page_icon="🍫",
)

# Judul halaman rata tengah
st.markdown("<h1 style='text-align: center;'>Aplikasi Pengelompokan Komponen Indeks Pembangunan Manusia Indonesia</h1>", unsafe_allow_html=True)

Image_home = "./assets/home_image.jpg"

col1, col2, col3 = st.columns([1, 6, 1])  # Kolom tengah lebih lebar
with col2:
    st.image(Image_home, width=500)  # Sesuaikan width

st.markdown("<br><br>", unsafe_allow_html=True)# Paragraf dengan teks rata kiri-kanan (justify)

st.markdown("""
    <div style='text-align: justify; text-indent: 40px;'>
Selamat datang di aplikasi kami. Aplikasi ini dirancang untuk membantu Anda menganalisis dan mengelompokkan data Indeks Pembangunan Manusia (IPM) di seluruh kabupaten dan kota di Indonesia. Indeks Pembangunan Manusia adalah indikator penting yang mengukur kualitas hidup di suatu daerah berdasarkan tiga dimensi utama: kesehatan, pendidikan, dan standar hidup layak. Dengan menggunakan algoritma pengelompokan (clustering), aplikasi ini dapat mengidentifikasi pola dan kemiripan antar daerah, sehingga memberikan wawasan yang lebih dalam mengenai sebaran tingkat pembangunan manusia di Indonesia.    </div>
    """, unsafe_allow_html=True)

# Spasi antar paragraf
st.markdown("<br>", unsafe_allow_html=True)


# Paragraf dengan teks rata kiri-kanan (justify)
st.markdown("""
    <div style='text-align: justify; text-indent: 40px;'>
Perancangan sistem ini bertujuan untuk melakukan klasifikasi kabupaten/kota di Indonesia berdasarkan komponen-komponen Indeks Pembangunan Manusia (IPM). Proses pengelompokan ini menerapkan algoritma Intelligent K-Means dan DBSCAN untuk menganalisis data, yang hasilnya kemudian disajikan dalam bentuk visualisasi data yang informatif    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Footer
show_footer()
