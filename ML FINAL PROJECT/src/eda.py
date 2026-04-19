"""
eda.py — Exploratory Data Analysis visualisations for Noon Daily ML Dashboard.

Produces Plotly figures consumed by the Streamlit dashboard:
  - Distribution plots (univariate)
  - Category frequency charts
  - Correlation heatmap
  - Bivariate scatter / box plots
  - Revenue time-series with Ramadan annotations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────── colour palette ───────────────────
NOON_YELLOW = "#F5C518"
NOON_DARK = "#1A1A2E"
NOON_ACCENT = "#E94560"
NOON_TEAL = "#0F3460"
PALETTE = px.colors.qualitative.Bold


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark-themed styling to all charts."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor="#1A1A2E", font_size=13),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# 1.  UNIVARIATE — NUMERICAL DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════

def plot_distribution(df: pd.DataFrame, col: str, nbins: int = 40) -> go.Figure:
    """Histogram + KDE overlay for a numerical column."""
    fig = px.histogram(
        df, x=col, nbins=nbins, marginal="box",
        color_discrete_sequence=[NOON_YELLOW],
        title=f"Distribution of {col}",
    )
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 2.  UNIVARIATE — CATEGORICAL FREQUENCIES
# ═══════════════════════════════════════════════════════════════

def plot_category_counts(df: pd.DataFrame, col: str, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of value counts for a categorical column."""
    counts = df[col].value_counts().nlargest(top_n).sort_values()
    fig = px.bar(
        x=counts.values, y=counts.index, orientation="h",
        color=counts.values, color_continuous_scale="YlOrRd",
        labels={"x": "Count", "y": col},
        title=f"Top {top_n} — {col}",
    )
    fig.update_coloraxes(showscale=False)
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 3.  CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Annotated correlation heatmap for numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(title="Correlation Heatmap", height=600)
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 4.  BIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def plot_bivariate_scatter(df: pd.DataFrame, x: str, y: str,
                           color: str | None = None) -> go.Figure:
    """Scatter plot with optional colour grouping."""
    fig = px.scatter(
        df, x=x, y=y, color=color, opacity=0.6,
        color_discrete_sequence=PALETTE,
        title=f"{y} vs {x}",
    )
    return _apply_theme(fig)


def plot_box_by_category(df: pd.DataFrame, cat_col: str, num_col: str) -> go.Figure:
    """Box plot of a numerical column grouped by a categorical column."""
    fig = px.box(
        df, x=cat_col, y=num_col, color=cat_col,
        color_discrete_sequence=PALETTE,
        title=f"{num_col} by {cat_col}",
    )
    fig.update_layout(showlegend=False)
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 5.  REVENUE TIME SERIES WITH RAMADAN HIGHLIGHTS
# ═══════════════════════════════════════════════════════════════

def plot_revenue_trend(df: pd.DataFrame) -> go.Figure:
    """
    Weekly revenue line chart.
    Shades Ramadan weeks and marks weekend-heavy weeks.
    Expects sales dataframe with 'order_date', 'revenue_aed', 'is_ramadan'.
    """
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    weekly = df.groupby(pd.Grouper(key="order_date", freq="W")).agg(
        revenue=("revenue_aed", "sum"),
        is_ramadan=("is_ramadan", "max"),
    ).reset_index()

    fig = go.Figure()

    # baseline revenue
    fig.add_trace(go.Scatter(
        x=weekly["order_date"], y=weekly["revenue"],
        mode="lines+markers", name="Weekly Revenue",
        line=dict(color=NOON_YELLOW, width=2),
        marker=dict(size=4),
    ))

    # Ramadan shading
    ramadan_weeks = weekly[weekly["is_ramadan"] == 1]
    if not ramadan_weeks.empty:
        for _, row in ramadan_weeks.iterrows():
            fig.add_vrect(
                x0=row["order_date"] - pd.Timedelta(days=3),
                x1=row["order_date"] + pd.Timedelta(days=3),
                fillcolor=NOON_ACCENT, opacity=0.12,
                line_width=0,
            )
        # legend entry for Ramadan
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=NOON_ACCENT),
            name="Ramadan Period",
        ))

    fig.update_layout(
        title="Weekly Revenue Trend (AED) — Ramadan Highlighted",
        xaxis_title="Week", yaxis_title="Revenue (AED)",
        yaxis_tickformat=",",
    )
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 6.  HELPER – PIE CHART
# ═══════════════════════════════════════════════════════════════

def plot_pie(df: pd.DataFrame, col: str, title: str = "") -> go.Figure:
    """Simple pie/donut chart."""
    counts = df[col].value_counts()
    fig = px.pie(
        names=counts.index, values=counts.values,
        color_discrete_sequence=PALETTE,
        title=title or f"Proportion of {col}",
        hole=0.4,
    )
    return _apply_theme(fig)
