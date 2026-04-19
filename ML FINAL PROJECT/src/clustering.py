"""
clustering.py — Customer Segmentation for Noon Daily.

Method: K-Means with Elbow + Silhouette analysis, PCA 2D visualisation.
Features: lifetime_spend, purchase_frequency, recency, AOV,
          app_sessions, return_rate.
Assigns descriptive business labels to each cluster.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

from src.preprocessing import merge_customer_data
from src.eda import _apply_theme, NOON_YELLOW, NOON_ACCENT, NOON_TEAL, PALETTE


# ═══════════════════════════════════════════════════════════════
# 1.  FEATURE SELECTION & SCALING
# ═══════════════════════════════════════════════════════════════

CLUSTER_FEATURES = [
    "lifetime_spend_aed",
    "purchase_frequency",
    "days_since_last_purchase",
    "avg_order_value_aed",
    "app_sessions_per_month",
    "return_rate",
]


def prepare_clustering_data():
    """Load, select features, scale. Returns scaled array, scaler, raw df."""
    df = merge_customer_data()
    df_clean = df.dropna(subset=CLUSTER_FEATURES).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[CLUSTER_FEATURES])

    return X_scaled, scaler, df_clean


# ═══════════════════════════════════════════════════════════════
# 2.  ELBOW + SILHOUETTE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def elbow_silhouette(X_scaled, k_range=range(2, 11)):
    """Compute inertia and silhouette scores for each k."""
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
    return list(k_range), inertias, sil_scores


def plot_elbow_silhouette(k_range, inertias, sil_scores) -> go.Figure:
    """Side-by-side elbow curve and silhouette score plot."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Elbow Curve (Inertia)", "Silhouette Score"])

    fig.add_trace(go.Scatter(
        x=k_range, y=inertias, mode="lines+markers",
        line=dict(color=NOON_YELLOW, width=2),
        marker=dict(size=8), name="Inertia",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=k_range, y=sil_scores, mode="lines+markers",
        line=dict(color=NOON_ACCENT, width=2),
        marker=dict(size=8), name="Silhouette",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    fig.update_layout(title="Optimal K Selection", showlegend=False, height=420)
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 3.  FIT FINAL K-MEANS
# ═══════════════════════════════════════════════════════════════

# Business labels applied after profiling
DEFAULT_LABELS = {
    0: "💎 High-Value Loyalists",
    1: "🛒 Regular Essentials Shoppers",
    2: "💤 Dormant At-Risk",
    3: "🏷️ Bargain Hunters",
    4: "🌟 New Enthusiasts",
}


def fit_kmeans(X_scaled, k: int = 4):
    """Fit K-Means and return labels array and model."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def assign_business_labels(df: pd.DataFrame, labels: np.ndarray, k: int) -> pd.DataFrame:
    """
    Heuristic labelling: rank clusters by lifetime spend descending,
    map to meaningful business names.
    """
    df = df.copy()
    df["Cluster"] = labels

    # rank clusters by mean lifetime spend
    cluster_spend = df.groupby("Cluster")["lifetime_spend_aed"].mean().sort_values(ascending=False)
    rank_map = {cluster: rank for rank, cluster in enumerate(cluster_spend.index)}

    label_names = [
        "💎 High-Value Loyalists",
        "🛒 Regular Essentials Shoppers",
        "🏷️ Bargain Hunters",
        "🌟 New Enthusiasts",
        "💤 Dormant At-Risk",
    ]

    # extend if k > 5
    while len(label_names) < k:
        label_names.append(f"Segment {len(label_names)+1}")

    df["Segment"] = df["Cluster"].map(lambda c: label_names[rank_map[c]])
    return df


# ═══════════════════════════════════════════════════════════════
# 4.  CLUSTER PROFILES
# ═══════════════════════════════════════════════════════════════

def cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Mean feature values per segment."""
    numeric_cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    profile = df.groupby("Segment")[numeric_cols].mean().round(2)
    profile["Customer Count"] = df.groupby("Segment").size()
    return profile


# ═══════════════════════════════════════════════════════════════
# 5.  PCA SCATTER
# ═══════════════════════════════════════════════════════════════

def plot_pca_clusters(X_scaled, labels, segment_labels=None) -> go.Figure:
    """2D PCA scatter coloured by cluster."""
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "Segment": segment_labels if segment_labels is not None else labels.astype(str),
    })

    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="Segment",
        color_discrete_sequence=PALETTE,
        title=f"Customer Segments — PCA Projection  (Var explained: "
              f"{pca.explained_variance_ratio_.sum()*100:.1f}%)",
        opacity=0.65,
    )
    fig.update_traces(marker=dict(size=6))
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 6.  RADAR CHART FOR PROFILES
# ═══════════════════════════════════════════════════════════════

def plot_cluster_radar(profile_df: pd.DataFrame) -> go.Figure:
    """Radar / polar chart comparing segment profiles (normalised 0-1)."""
    numeric_cols = [c for c in CLUSTER_FEATURES if c in profile_df.columns]
    normalised = profile_df[numeric_cols].copy()

    for col in numeric_cols:
        rng = normalised[col].max() - normalised[col].min()
        normalised[col] = (normalised[col] - normalised[col].min()) / (rng if rng else 1)

    fig = go.Figure()
    colors = PALETTE
    for i, (seg, row) in enumerate(normalised.iterrows()):
        vals = row.values.tolist()
        vals.append(vals[0])   # close the polygon
        cats = numeric_cols + [numeric_cols[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=seg, line=dict(color=colors[i % len(colors)]),
            opacity=0.6,
        ))

    fig.update_layout(
        title="Segment Profiles — Radar Comparison",
        polar=dict(bgcolor="rgba(0,0,0,0)"),
        height=500,
    )
    return _apply_theme(fig)
