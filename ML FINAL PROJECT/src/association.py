"""
association.py — Market Basket Analysis for Noon Daily.

Method: Apriori algorithm via mlxtend.
Input:  noon_market_basket.csv (transaction_id, comma-separated product_ids).
Output: Association rules ranked by lift, bundle recommendations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import load_basket, load_catalogue
from src.eda import _apply_theme, NOON_YELLOW, NOON_ACCENT, NOON_TEAL, PALETTE


# ═══════════════════════════════════════════════════════════════
# 1.  TRANSACTION MATRIX
# ═══════════════════════════════════════════════════════════════

def build_transaction_matrix() -> pd.DataFrame:
    """
    Convert market basket CSV to one-hot encoded DataFrame.
    Rows = transactions, columns = product_ids, values = True/False.
    """
    basket = load_basket()
    # explode comma-separated product_ids
    basket["product_list"] = basket["product_ids"].str.split(",")
    exploded = basket.explode("product_list")
    exploded["product_list"] = exploded["product_list"].str.strip()
    exploded["present"] = True

    matrix = exploded.pivot_table(
        index="transaction_id", columns="product_list",
        values="present", aggfunc="max", fill_value=False,
    )
    return matrix.astype(bool)


# ═══════════════════════════════════════════════════════════════
# 2.  APRIORI + RULES
# ═══════════════════════════════════════════════════════════════

def run_apriori(min_support: float = 0.01, min_confidence: float = 0.4,
                metric: str = "lift", min_threshold: float = 1.0):
    """
    Run Apriori and generate association rules.
    Returns frequent_itemsets DataFrame and rules DataFrame.
    """
    from mlxtend.frequent_patterns import apriori, association_rules

    matrix = build_transaction_matrix()
    freq = apriori(matrix, min_support=min_support, use_colnames=True)

    if freq.empty:
        return freq, pd.DataFrame()

    rules = association_rules(freq, metric=metric, min_threshold=min_threshold)
    rules = rules[rules["confidence"] >= min_confidence]

    # Convert frozensets to readable strings
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return freq, rules


# ═══════════════════════════════════════════════════════════════
# 3.  ENRICH WITH PRODUCT NAMES
# ═══════════════════════════════════════════════════════════════

def enrich_rules_with_names(rules: pd.DataFrame) -> pd.DataFrame:
    """Map product_ids in rules to human-readable product names."""
    cat = load_catalogue()
    id_to_name = dict(zip(cat["product_id"], cat["product_name"]))

    def _map_names(frozenset_ids):
        return ", ".join(id_to_name.get(pid, pid) for pid in sorted(frozenset_ids))

    rules = rules.copy()
    rules["Antecedent Products"] = rules["antecedents"].apply(_map_names)
    rules["Consequent Products"] = rules["consequents"].apply(_map_names)
    return rules


# ═══════════════════════════════════════════════════════════════
# 4.  TOP RULES TABLE
# ═══════════════════════════════════════════════════════════════

def top_rules_table(rules: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Tidy table of top rules for dashboard display."""
    if rules.empty:
        return pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift"])
    display = rules.head(top_n)[[
        "antecedents_str", "consequents_str", "support", "confidence", "lift"
    ]].copy()
    display.columns = ["Antecedent", "Consequent", "Support", "Confidence", "Lift"]
    display["Support"] = display["Support"].round(4)
    display["Confidence"] = display["Confidence"].round(4)
    display["Lift"] = display["Lift"].round(2)
    return display.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# 5.  VISUALISATIONS
# ═══════════════════════════════════════════════════════════════

def plot_rules_scatter(rules: pd.DataFrame) -> go.Figure:
    """Support vs Confidence scatter, sized by Lift."""
    if rules.empty:
        return go.Figure()
    fig = px.scatter(
        rules.head(50), x="support", y="confidence",
        size="lift", color="lift",
        hover_data=["antecedents_str", "consequents_str"],
        color_continuous_scale="YlOrRd",
        title="Association Rules — Support vs Confidence (size = Lift)",
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
    )
    return _apply_theme(fig)


def plot_top_rules_bar(rules: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of top rules by Lift."""
    if rules.empty:
        return go.Figure()
    subset = rules.head(top_n).copy()
    subset["rule"] = subset["antecedents_str"] + " → " + subset["consequents_str"]
    subset = subset.sort_values("lift")

    fig = px.bar(
        subset, x="lift", y="rule", orientation="h",
        color="confidence", color_continuous_scale="Teal",
        title=f"Top {top_n} Association Rules by Lift",
        labels={"lift": "Lift", "rule": "Rule", "confidence": "Confidence"},
    )
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 6.  BUNDLE RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

def generate_bundle_recommendations(rules: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Auto-generate bundle proposals from top rules.
    Returns list of dicts with bundle info.
    """
    if rules.empty:
        return []

    enriched = enrich_rules_with_names(rules)
    bundles = []
    for i, row in enriched.head(top_n).iterrows():
        bundle = {
            "Bundle #": i + 1,
            "Products": f'{row["Antecedent Products"]}  +  {row["Consequent Products"]}',
            "Lift": round(row["lift"], 2),
            "Confidence": f'{row["confidence"]*100:.1f}%',
            "Suggested Strategy": (
                "10% combo discount" if row["lift"] > 3
                else "Free delivery on bundle" if row["lift"] > 2
                else "Highlight on product page"
            ),
        }
        bundles.append(bundle)
    return bundles
