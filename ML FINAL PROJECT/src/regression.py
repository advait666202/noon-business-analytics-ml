"""
regression.py — Weekly Demand Forecasting for Noon Daily.

Models: Linear Regression, Ridge, Lasso (with multiple alphas).
Features: discount_pct, is_ramadan, is_weekend, marketing_spend,
          web_traffic, week_of_year, month, num_categories.
Target:  total weekly revenue (AED).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_sales, aggregate_weekly
from src.eda import _apply_theme, NOON_YELLOW, NOON_ACCENT, NOON_TEAL


# ═══════════════════════════════════════════════════════════════
# 1.  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "avg_discount", "avg_marketing_spend", "avg_web_traffic",
    "is_ramadan", "is_weekend_pct", "week_of_year",
    "month", "num_categories", "avg_unit_price",
]

TARGET = "total_revenue"


def prepare_regression_data():
    """Return X_train, X_test, y_train, y_test, scaler, weekly df."""
    sales = load_sales()
    weekly = aggregate_weekly(sales)
    weekly = weekly.dropna(subset=FEATURE_COLS + [TARGET])

    X = weekly[FEATURE_COLS].values
    y = weekly[TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler, weekly


# ═══════════════════════════════════════════════════════════════
# 2.  TRAINING
# ═══════════════════════════════════════════════════════════════

def train_models(X_train, y_train, X_test, y_test):
    """
    Train OLS, Ridge (α=1), Lasso (α=1) and return a results dict.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (a=1.0)": Ridge(alpha=1.0),
        "Lasso (a=1.0)": Lasso(alpha=1.0),
    }
    results = {}
    n = X_test.shape[0]
    p = X_test.shape[1]

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds) * 100

        results[name] = {
            "model": model,
            "preds": preds,
            "R²": round(r2, 4),
            "Adj R²": round(adj_r2, 4),
            "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
        }
    return results


# ═══════════════════════════════════════════════════════════════
# 3.  METRICS TABLE
# ═══════════════════════════════════════════════════════════════

def metrics_table(results: dict) -> pd.DataFrame:
    """Return a tidy DataFrame of model comparison metrics."""
    rows = []
    for name, vals in results.items():
        rows.append({
            "Model": name,
            "R²": vals["R²"],
            "Adj R²": vals["Adj R²"],
            "RMSE (AED)": vals["RMSE"],
            "MAPE (%)": vals["MAPE (%)"],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 4.  ACTUAL vs PREDICTED PLOT
# ═══════════════════════════════════════════════════════════════

def plot_actual_vs_predicted(y_test, results: dict) -> go.Figure:
    """Scatter of actual vs predicted for each model."""
    fig = go.Figure()
    colors = [NOON_YELLOW, NOON_ACCENT, NOON_TEAL]
    for i, (name, vals) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            x=y_test, y=vals["preds"], mode="markers",
            name=name, marker=dict(color=colors[i % len(colors)], size=7, opacity=0.7),
        ))
    # perfect-prediction line
    mn, mx = min(y_test), max(y_test)
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        name="Perfect Fit", line=dict(dash="dash", color="white", width=1),
    ))
    fig.update_layout(
        title="Actual vs Predicted Weekly Revenue",
        xaxis_title="Actual Revenue (AED)",
        yaxis_title="Predicted Revenue (AED)",
    )
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 5.  COEFFICIENT BAR CHART
# ═══════════════════════════════════════════════════════════════

def plot_coefficients(model, feature_names: list) -> go.Figure:
    """Horizontal bar chart of regression coefficients."""
    coefs = model.coef_
    df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    df = df.sort_values("Coefficient")
    fig = px.bar(
        df, x="Coefficient", y="Feature", orientation="h",
        color="Coefficient", color_continuous_scale="RdYlGn",
        title="Feature Coefficients (Standardised)",
    )
    fig.update_coloraxes(showscale=False)
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 6.  SCENARIO SIMULATOR
# ═══════════════════════════════════════════════════════════════

def simulate_scenario(model, scaler, discount: float, marketing_spend: float,
                      is_ramadan: int, web_traffic: float = 100000,
                      is_weekend_pct: float = 0.3,
                      week_of_year: int = 20, month: int = 5,
                      num_categories: int = 8,
                      avg_unit_price: float = 25.0) -> float:
    """
    Predict total weekly revenue for a hypothetical business scenario.
    Returns predicted revenue in AED.
    """
    row = np.array([[
        discount, marketing_spend, web_traffic,
        is_ramadan, is_weekend_pct, week_of_year,
        month, num_categories, avg_unit_price,
    ]])
    row_scaled = scaler.transform(row)
    return float(model.predict(row_scaled)[0])
