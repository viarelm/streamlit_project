import streamlit as st
from utils import show_footer

st.set_page_config(
    page_title="Pertanyaan Umum",
    page_icon="🔍",
)

st.markdown("<h1 style='text-align: center;'>Frequently Asked Questions</h1>", unsafe_allow_html=True)

with st.expander("Apa tujuan dari website ini?"):
    st.write("Website ini dirancang untuk melakukan pengelompokan (clustering) dan pemetaan lokasi berdasarkan data komponen Indeks Pembangunan Manusia di Indonesia. Dengan website ini, pengguna dapat melihat hasil pengelompokan dari hasil visual.")

with st.expander("Data apa saja yang digunakan dalam website ini?"):
    st.write("Website ini menggunakan data dari Badan Pusat Statistik (BPS) dengan data Indeks Pembangunan Manusia, Angka Harapan Hidup, dan Pengeluaran Per Kapita dari tingkap daerah.")

with st.expander("Bagaimana cara website ini melakukan pengelompokkan (clustering)?"):
    st.write("Website ini menggunakan beberapa algoritma clustering, yaitu Intelligent K-Means dan DBSCAN untuk mengelompokkan wilayah berdasarkan komponen Indeks Pembangunan Manusia. Pengelompokan dilakukan berdasarkan data yang tersedia dan dilakukan pembersihan data duplikat dan data kosong dengan memasukan data rata-rata per fitur.")

with st.expander("Apa manfaat dari clustering komponen IPM?"):
    st.write("Dengan melakukan clustering, pengguna dapat melihat pola pembangunan manusia per daerah untuk mengetahui seperti apa kelompok daerah yang memiliki kesamaan.")

with st.expander("Apakah bisa mengubah data yang digunakan dalam website?"):
    st.write("Saat ini, program menggunakan data yang telah disediakan, namun tetap dapat mengganti atau memperbarui data tersebut dengan mengikuti format template yang tersedia. Template data dapat diunduh Panduan ataupun Clustering. Pastikan data yang diunggah sesuai dengan struktur yang ditentukan agar dapat diproses dengan benar oleh program.")

with st.expander("Bagaimana cara menghubungi developer jika menemui masalah?"):
    st.write("Jika mengalami masalah atau memiliki pertanyaan lebih lanjut, dapat menghubungi developer melalui pilihan kontak di halaman Tentang.")

# Footer
show_footer()