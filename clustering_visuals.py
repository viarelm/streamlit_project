# clustering_visuals.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_dbscan_results(data, dbscan_model, clusters, feature_x, feature_y):
    """Memvisualisasikan hasil DBSCAN."""
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

def plot_kmeans_results(data, clusters, final_centroids_scaled, feature_x, feature_y, scaler, feature_names):
    """
    Memvisualisasikan hasil K-Means.
    'data' adalah dataframe yang *tidak* diskala (tapi sudah difilter).
    'feature_names' adalah daftar nama kolom yang digunakan untuk scaling.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot data points
    scatter = ax.scatter(data[feature_x], data[feature_y], c=clusters, cmap='viridis', alpha=0.8, label='Data Points')
    
    # Buat dataframe dari centroid yang ternormalisasi untuk inverse transform
    # PERBAIKAN: Gunakan feature_names yang dinamis
    centroids_df_scaled = pd.DataFrame(final_centroids_scaled, columns=feature_names)
    
    # Inverse transform untuk mendapatkan posisi centroid dalam skala data asli
    centroids_original_scale = scaler.inverse_transform(centroids_df_scaled)
    centroids_original_df = pd.DataFrame(centroids_original_scale, columns=feature_names)
    
    # Plot centroids
    ax.scatter(centroids_original_df[feature_x], centroids_original_df[feature_y], s=250, c='red', marker='P', label='Centroids', edgecolor='k')

    ax.set_title(f'Visualisasi Intelligent K-Means ({feature_x} vs {feature_y})', fontsize=16)
    ax.set_xlabel(feature_x); ax.set_ylabel(feature_y)
    ax.grid(True, alpha=0.5)
    
    # Buat legenda
    legend1 = ax.legend(*scatter.legend_elements(), title="Klaster")
    ax.add_artist(legend1)
    
    # Dapatkan handles dan labels untuk legenda centroid
    handles, labels = ax.get_legend_handles_labels()
    centroid_handle = [h for h, l in zip(handles, labels) if l == 'Centroids']
    if centroid_handle:
        ax.legend(handles=centroid_handle, labels=['Centroids'], loc='upper right')

    st.pyplot(fig)