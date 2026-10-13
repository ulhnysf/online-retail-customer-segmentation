"""RFM + KMeans müşteri segmentasyonu; Elbow, Silhouette ve PCA grafikleri."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from data_load import load_processed
from features import make_rfm


def choose_k(X, k_min=2, k_max=10, random_state=RANDOM_STATE):
    """Her k için inertia (Elbow) ve silhouette skorunu hesaplar."""
    ks = list(range(k_min, k_max + 1))
    inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return ks, inertias, silhouettes


def _line_plot(x, y, xlabel, ylabel, title, path):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="data/processed/clean.csv")
    ap.add_argument("--outdir", required=True, help="outputs")
    ap.add_argument("--k", type=int, default=4, help="nihai KMeans küme sayısı (varsayılan 4)")
    ap.add_argument("--k_max", type=int, default=10, help="Elbow/Silhouette için denenecek en büyük k")
    args = ap.parse_args()

    fig_dir = os.path.join(args.outdir, "figures")
    tab_dir = os.path.join(args.outdir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    rfm = make_rfm(load_processed(args.input))

    # RFM ölçekleri çok farklı olduğu için standartlaştırılır
    X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]].values)

    ks, inertias, silhouettes = choose_k(X, 2, args.k_max)
    _line_plot(ks, inertias, "k", "Inertia (SSE)", "Elbow (KMeans)", os.path.join(fig_dir, "elbow.png"))
    _line_plot(ks, silhouettes, "k", "Silhouette", "Silhouette Scores", os.path.join(fig_dir, "silhouette.png"))

    km = KMeans(n_clusters=args.k, random_state=RANDOM_STATE, n_init="auto")
    rfm["Segment"] = km.fit_predict(X)

    Z = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    rfm["PC1"], rfm["PC2"] = Z[:, 0], Z[:, 1]

    plt.figure()
    for seg in sorted(rfm["Segment"].unique()):
        sub = rfm[rfm["Segment"] == seg]
        plt.scatter(sub["PC1"], sub["PC2"], s=10, label=f"Seg {seg}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Scatter by Segment")
    plt.legend(markerscale=2)
    plt.savefig(os.path.join(fig_dir, "pca_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()

    summary = rfm.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        Recency_mean=("Recency", "mean"),
        Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"),
        Monetary_sum=("Monetary", "sum"),
    ).reset_index()

    rfm.to_csv(os.path.join(tab_dir, "rfm_with_segment.csv"), index=False)
    summary.to_csv(os.path.join(tab_dir, "segment_summary.csv"), index=False)

    print("[OK] Segmentasyon çıktıları kaydedildi:")
    print(" -", os.path.join(tab_dir, "rfm_with_segment.csv"))
    print(" -", os.path.join(tab_dir, "segment_summary.csv"))
    print(" - grafikler:", fig_dir)


if __name__ == "__main__":
    main()
