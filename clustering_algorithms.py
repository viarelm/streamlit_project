# clustering_algorithms.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
import time

def run_dbscan(data_scaled, eps, min_samples):
    """Menjalankan DBSCAN."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_scaled)
    return dbscan, clusters

def run_intelligent_kmeans(normalized_data, feature_names):
    """
    Menjalankan algoritma Intelligent K-Means lengkap berdasarkan skrip yang diberikan.
    Fungsi ini akan mengembalikan hasil akhir dan log proses yang detail.
    """
    log_area = st.empty()
    logs = ["**Memulai Proses Intelligent K-Means...**"]
    
    # --- FASE 1: PENCARIAN INITIAL CENTROID OPTIMAL ---
    logs.append("\n--- FASE 1: Pencarian Initial Centroid ---")
    log_area.info("\n".join(logs))
    time.sleep(1)

    historical_centroids = []
    
    # Centroid 1: Titik terjauh dari pusat massa
    center_of_mass = normalized_data.mean(axis=0)
    distances_to_com = np.linalg.norm(normalized_data - center_of_mass, axis=1)
    c1_idx = np.argmax(distances_to_com)
    centroid1 = normalized_data[c1_idx]
    historical_centroids.append(centroid1)
    logs.append(f"➡️ Centroid 1 ditemukan (indeks data: {c1_idx}).")
    log_area.info("\n".join(logs))
    time.sleep(1)

    # Centroid 2: Titik terjauh dari Centroid 1
    distances_to_c1 = np.linalg.norm(normalized_data - centroid1, axis=1)
    c2_idx = np.argmax(distances_to_c1)
    centroid2 = normalized_data[c2_idx]
    historical_centroids.append(centroid2)
    logs.append(f"➡️ Centroid 2 ditemukan (indeks data: {c2_idx}).")
    log_area.info("\n".join(logs))
    time.sleep(1)

    k = 2
    while True:
        logs.append(f"\n**Mencari kandidat untuk Centroid ke-{k+1}...**")
        log_area.info("\n".join(logs))
        time.sleep(1)

        reference_centroids = np.array(historical_centroids)
        point_to_centroid_distances = pairwise_distances(normalized_data, reference_centroids)
        avg_of_distances = point_to_centroid_distances.mean(axis=1)
        next_centroid_idx = np.argmax(avg_of_distances)
        next_centroid_candidate = normalized_data[next_centroid_idx]
        is_duplicate = any(np.allclose(next_centroid_candidate, old_c) for old_c in historical_centroids)

        if is_duplicate:
            logs.append(f"🛑 Kandidat (indeks {next_centroid_idx}) adalah duplikat. Pencarian dihentikan.")
            log_area.info("\n".join(logs))
            time.sleep(1)
            break
        else:
            logs.append(f"✅ Kandidat unik ditemukan (indeks {next_centroid_idx}). Ditambahkan sebagai Centroid ke-{k+1}.")
            historical_centroids.append(next_centroid_candidate)
            k += 1
            log_area.info("\n".join(logs))
            time.sleep(1)

    final_k = len(historical_centroids)
    final_initial_centroids = np.array(historical_centroids)
    logs.append(f"\n**✅ FASE 1 SELESAI: Ditemukan K optimal = {final_k}**")
    log_area.info("\n".join(logs))
    time.sleep(1)
    
    # --- FASE 2: MENJALANKAN K-MEANS FINAL HINGGA KONVERGEN ---
    logs.append("\n--- FASE 2: Clustering Final dengan K-Means ---")
    log_area.info("\n".join(logs))
    time.sleep(1)

    current_centroids = final_initial_centroids
    labels_sebelumnya = np.full(shape=len(normalized_data), fill_value=-1)
    iter_count = 0

    while True:
        iter_count += 1
        logs.append(f"\n**Iterasi Konvergensi Final ke-{iter_count}...**")
        log_area.info("\n".join(logs))
        time.sleep(1)
        
        # Tentukan klaster baru
        distances = pairwise_distances(normalized_data, current_centroids)
        labels_sekarang = np.argmin(distances, axis=1)

        # Cek konvergensi
        if np.array_equal(labels_sekarang, labels_sebelumnya):
            logs.append(f"✅ KONVERGENSI TERCAPAI dalam {iter_count-1} langkah.")
            log_area.info("\n".join(logs))
            time.sleep(1)
            break

        perpindahan = labels_sekarang != labels_sebelumnya
        pindah_count = np.sum(perpindahan)
        logs.append(f" -> {pindah_count} titik data berpindah klaster.")
        log_area.info("\n".join(logs))
        time.sleep(0.5)

        labels_sebelumnya = labels_sekarang
        
        # Update posisi centroid
        logs.append(" -> Memperbarui posisi centroid...")
        current_centroids = np.array([normalized_data[labels_sekarang == i].mean(axis=0) for i in range(final_k)])
        log_area.info("\n".join(logs))
        time.sleep(0.5)
        
        if iter_count > 100:
            logs.append("⚠️ Peringatan: Iterasi melebihi 100, berhenti paksa.")
            log_area.warning("\n".join(logs))
            break
            
    # Kembalikan hasil
    return labels_sekarang, current_centroids, final_k