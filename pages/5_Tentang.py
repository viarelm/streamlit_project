# -*- coding: utf-8 -*-
import streamlit as st
from utils import show_footer

st.set_page_config(
    page_title="Tentang",
    page_icon="👨‍💻",
    layout='wide',
)

st.markdown("""
<style>
    .profile-img {
        border-radius: 50%;
        border: 3px solid #4F8BF9;
        margin-bottom: 0.5rem;
    }
    .section-title {
        color: #4F8BF9;
        border-bottom: 2px solid #4F8BF9;
        margin-top: 1rem !important;
    }
    .compact-text {
        line-height: 1.4;
        margin-bottom: 0.5rem;
        text-align: justify;
    }
    .compact-list {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2], gap="medium")
with col1:
    st.image("./assets/profile.jpeg", width=200, use_container_width=False, 
             caption="Valentino Richardo Lim")

with col2:
    st.title('Valentino Richardo Lim', anchor=False)
    
    st.markdown("""
    <div class="compact-text">
    Halo semua! saya merupakan mahasiswa Teknik Informatika Universitas Tarumanagara angkatan 2022. Aplikasi ini merupakan rancangan tugas akhir yang menjadi syarat kelulusan tingkat sarjana.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: 1rem;">
        <a href="https://github.com/viarelm" target="_blank" style="text-decoration: none; margin-right: 10px;">
            <img src="https://img.icons8.com/plasticine/100/github.png" width="30">
        </a>
        <a href="https://www.linkedin.com/in/valentino-richardo-lim-b67707251" target="_blank" style="text-decoration: none; margin-right: 10px;">
            <img src="https://img.icons8.com/plasticine/100/linkedin.png" width="30">
        </a>
        <a href="https://www.instagram.com/viarelm" target="_blank" style="text-decoration: none;">
            <img src="https://img.icons8.com/plasticine/100/instagram-new.png" width="30">
        </a>
    </div>
    """, unsafe_allow_html=True)


st.divider()  
    
show_footer()