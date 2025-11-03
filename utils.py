# utils.py
import streamlit as st
import pandas as pd
import os

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


@st.cache_data
def get_template_file():
    """Memuat data template untuk di-download."""
    try:
        with open("assets/template_dataset.xlsx", "rb") as file:
            return file.read()
    except FileNotFoundError:
        st.error("File 'template_dataset.xlsx' tidak ditemukan di root folder.")
        return None
    
def show_footer():
    st.markdown("""
    <hr style='margin: 0.5rem 0;'>
    <div style='text-align: center; font-size: 0.9rem; padding: 0.5rem 0;'>
        © 2025 Valentino Richardo Lim
    </div>
    """, unsafe_allow_html=True)

