"""
preprocessing.py — Data loading, cleaning, feature engineering for Noon Daily ML Dashboard.

Handles:
  - Loading all five CSVs
  - Missing value treatment (MNAR for NPS & delivery_rating)
  - Feature engineering (time-based, recency, aggregation)
  - Dataset joins (profiles+engagement, transactions+catalogue)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────── paths ───────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ═══════════════════════════════════════════════════════════════
# 1.  RAW DATA LOADERS
# ═══════════════════════════════════════════════════════════════

def load_sales() -> pd.DataFrame:
    """Load noon_sales_transactions.csv with parsed dates."""
    df = pd.read_csv(DATA_DIR / "noon_sales_transactions.csv", parse_dates=["order_date"])
    return df


def load_profiles() -> pd.DataFrame:
    """Load noon_customer_profiles.csv with parsed dates."""
    df = pd.read_csv(DATA_DIR / "noon_customer_profiles.csv",
                     parse_dates=["registration_date", "last_purchase_date"])
    return df


def load_engagement() -> pd.DataFrame:
    """Load noon_customer_engagement.csv."""
    return pd.read_csv(DATA_DIR / "noon_customer_engagement.csv")


def load_catalogue() -> pd.DataFrame:
    """Load noon_product_catalogue.csv with parsed launch_date."""
    return pd.read_csv(DATA_DIR / "noon_product_catalogue.csv", parse_dates=["launch_date"])


def load_basket() -> pd.DataFrame:
    """Load noon_market_basket.csv."""
    return pd.read_csv(DATA_DIR / "noon_market_basket.csv")


# ═══════════════════════════════════════════════════════════════
# 2.  MISSING VALUE HANDLING
# ═══════════════════════════════════════════════════════════════

def handle_engagement_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle MNAR missing values in engagement data.
    - nps_score (~35% missing): impute with median and create
      binary indicator 'nps_missing' (disengaged signal).
    - delivery_rating (~25% missing): impute with median and create
      binary indicator 'delivery_rating_missing'.
    """
    df = df.copy()
    # Create binary indicators BEFORE imputation
    df["nps_missing"] = df["nps_score"].isna().astype(int)
    df["delivery_rating_missing"] = df["delivery_rating"].isna().astype(int)

    # Impute with median (robust to skew)
    df["nps_score"] = df["nps_score"].fillna(df["nps_score"].median())
    df["delivery_rating"] = df["delivery_rating"].fillna(df["delivery_rating"].median())
    return df


# ═══════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def engineer_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to sales data."""
    df = df.copy()
    df["week"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["month"] = df["order_date"].dt.month
    df["year"] = df["order_date"].dt.year
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["year_week"] = df["order_date"].dt.strftime("%Y-%W")
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactions to weekly level for regression.
    Returns one row per week with revenue, quantity, discount,
    marketing spend, web traffic, and Ramadan/weekend flags.
    """
    df = engineer_sales_features(df)
    weekly = df.groupby("year_week").agg(
        total_revenue=("revenue_aed", "sum"),
        total_quantity=("quantity", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_unit_price=("unit_price_aed", "mean"),
        num_orders=("order_id", "nunique"),
        avg_marketing_spend=("weekly_marketing_spend_aed", "mean"),
        avg_web_traffic=("weekly_web_traffic", "mean"),
        is_ramadan=("is_ramadan", "max"),
        is_weekend_pct=("is_weekend", "mean"),
        num_categories=("category", "nunique"),
        week_of_year=("week_of_year", "first"),
        month=("month", "first"),
    ).reset_index()
    return weekly


# ═══════════════════════════════════════════════════════════════
# 4.  DATASET JOINS
# ═══════════════════════════════════════════════════════════════

def merge_customer_data() -> pd.DataFrame:
    """
    Merge customer_profiles + engagement on customer_id.
    Produces the base dataframe for churn classification and clustering.
    """
    profiles = load_profiles()
    eng = handle_engagement_missing(load_engagement())

    # Drop duplicate 'days_since_last_purchase' from engagement before merge
    eng = eng.drop(columns=["days_since_last_purchase"], errors="ignore")

    merged = profiles.merge(eng, on="customer_id", how="left")
    return merged


def merge_transactions_catalogue() -> pd.DataFrame:
    """Enrich transactions with product catalogue metadata."""
    sales = load_sales()
    cat = load_catalogue()
    merged = sales.merge(cat[["product_id", "product_name", "brand",
                              "price_tier", "shelf_life_days", "avg_rating"]],
                         on="product_id", how="left")
    return merged


# ═══════════════════════════════════════════════════════════════
# 5.  DATA-QUALITY REPORT  (used by EDA page)
# ═══════════════════════════════════════════════════════════════

def data_quality_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return a tidy summary of missing values, dtypes, and unique counts."""
    report = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Null": df.isnull().sum().values,
        "Null %": (df.isnull().sum() / len(df) * 100).round(2).values,
        "Unique": df.nunique().values,
    })
    report["Dataset"] = name
    return report


def get_all_quality_reports() -> pd.DataFrame:
    """Compile quality reports for all five datasets."""
    frames = [
        data_quality_report(load_sales(), "Sales Transactions"),
        data_quality_report(load_profiles(), "Customer Profiles"),
        data_quality_report(load_engagement(), "Customer Engagement"),
        data_quality_report(load_catalogue(), "Product Catalogue"),
        data_quality_report(load_basket(), "Market Basket"),
    ]
    return pd.concat(frames, ignore_index=True)
