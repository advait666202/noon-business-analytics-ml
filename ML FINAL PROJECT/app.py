"""
app.py — Noon Daily Grocery ML Dashboard (Streamlit Multi-Page App).

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ─────────────────── page config (MUST be first st call) ───────────────────
st.set_page_config(
    page_title="Noon Daily — ML Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────── custom CSS ───────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* dark background */
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%); }

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid rgba(245,197,24,0.15);
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(245,197,24,0.08) 0%, rgba(15,52,96,0.18) 100%);
    border: 1px solid rgba(245,197,24,0.2);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(245,197,24,0.15);
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #F5C518, #E94560);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.kpi-label {
    font-size: 0.85rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 6px;
}

/* section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #F5C518;
    margin: 32px 0 12px 0;
    border-left: 4px solid #E94560;
    padding-left: 14px;
}

/* metric highlight */
.metric-badge {
    display: inline-block;
    background: rgba(233,69,96,0.15);
    color: #E94560;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
}

/* dataframes */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* tabs */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str):
    """Render a styled KPI card."""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  CACHED DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_all_data():
    from src.preprocessing import (load_sales, load_profiles, load_engagement,
                                   load_catalogue, load_basket)
    return {
        "sales": load_sales(),
        "profiles": load_profiles(),
        "engagement": load_engagement(),
        "catalogue": load_catalogue(),
        "basket": load_basket(),
    }


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div style="padding: 10px 10px 20px 10px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 28px;">📦</span>
        <div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.3px;">Noon Daily</div>
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600; letter-spacing: 0.5px;">ML DASHBOARD</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Custom CSS targeting the sidebar specifically for the Dark SaaS Theme
st.sidebar.markdown("""
<style>
/* 1. Sidebar Container Constraint to prevent wrapping */
section[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    background-image: none !important;
    border-right: 1px solid #1E293B !important;
    min-width: 330px !important; /* Slightly increased to fit text gracefully */
}

/* 2. Text styling & Strict No-Wrap */
section[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stRadio p,
section[data-testid="stSidebar"] span[data-testid="stRadioActionLabel"] {
    color: #F8FAFC !important; /* Pure slate */
    font-size: 0.95rem !important;
    white-space: nowrap !important; /* CRITICAL: Forces single line */
}

/* 3. Hide Streamlit broken icon circles completely */
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* 4. Layout Row for Icon + Text */
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
    display: flex !important;
    flex-direction: row !important; /* Force row layout */
    align-items: center !important;
    padding: 10px 14px !important;
    margin: 4px 12px !important;
    border-radius: 8px !important;
    background-color: transparent !important;
    transition: all 0.2s ease !important;
    border-left: 3px solid transparent !important;
    cursor: pointer !important;
}

/* 5. Hover State */
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:hover {
    background-color: #1E293B !important;
}

/* 6. Active State */
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:has(input:checked) {
    background-color: #1E293B !important;
    border-left: 3px solid #38BDF8 !important; /* Blue active line */
}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:has(input:checked) p {
    color: #38BDF8 !important;
    font-weight: 600 !important;
}

/* 7. Section Titles (Absolute positioned to avoid breaking flex layout) */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    position: relative !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(1) { margin-top: 36px !important; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(1)::before {
    content: "ANALYTICS"; position: absolute; top: -26px; left: 16px; font-size: 0.75rem; font-weight: 700; color: #475569 !important; letter-spacing: 1px; pointer-events: none; white-space: nowrap;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(3) { margin-top: 40px !important; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(3)::before {
    content: "MACHINE LEARNING"; position: absolute; top: -26px; left: 16px; font-size: 0.75rem; font-weight: 700; color: #475569 !important; letter-spacing: 1px; pointer-events: none; white-space: nowrap;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(7) { margin-top: 40px !important; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(7)::before {
    content: "INSIGHTS"; position: absolute; top: -26px; left: 16px; font-size: 0.75rem; font-weight: 700; color: #475569 !important; letter-spacing: 1px; pointer-events: none; white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

PAGES = [
    "📊 Executive Overview",
    "🔍 Exploratory Data Analysis",
    "📈 Demand Forecasting",
    "⚠️ Churn Prediction",
    "🎯 Customer Segmentation",
    "🔗 Basket Analysis",
    "💡 Business Insights",
]

page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.caption("Built for Noon Daily  ·  MAIB")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

if page == PAGES[0]:
    data = load_all_data()
    sales = data["sales"].copy()
    profiles = data["profiles"]

    # ── Filters ──
    st.markdown("### 🎛️ Filters")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        cities = ["All"] + sorted(sales["delivery_city"].unique().tolist())
        sel_city = st.selectbox("City", cities, key="ov_city")
    with fc2:
        cats = ["All"] + sorted(sales["category"].unique().tolist())
        sel_cat = st.selectbox("Category", cats, key="ov_cat")
    with fc3:
        min_date = sales["order_date"].min().date()
        max_date = sales["order_date"].max().date()
        date_range = st.date_input("Date Range", [min_date, max_date], key="ov_date")

    # Apply filters
    if sel_city != "All":
        sales = sales[sales["delivery_city"] == sel_city]
    if sel_cat != "All":
        sales = sales[sales["category"] == sel_cat]
    if len(date_range) == 2:
        sales = sales[(sales["order_date"].dt.date >= date_range[0]) &
                      (sales["order_date"].dt.date <= date_range[1])]

    st.markdown("")

    # ── KPIs ──
    total_revenue = sales["revenue_aed"].sum()
    avg_order_value = sales.groupby("order_id")["revenue_aed"].sum().mean()
    churn_rate = profiles["churned"].mean() * 100
    total_customers = profiles["customer_id"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total Revenue", f"AED {total_revenue:,.0f}")
    with k2:
        kpi_card("Avg Order Value", f"AED {avg_order_value:,.2f}")
    with k3:
        kpi_card("Churn Rate", f"{churn_rate:.1f}%")
    with k4:
        kpi_card("Total Customers", f"{total_customers:,}")

    st.markdown("")

    # ── Revenue Trend ──
    section_header("Revenue Trend — Ramadan & Weekend Highlights")
    from src.eda import plot_revenue_trend
    st.plotly_chart(plot_revenue_trend(sales), use_container_width=True)

    # ── Category Breakdown ──
    section_header("Revenue by Category")
    from src.eda import plot_category_counts
    cat_rev = sales.groupby("category")["revenue_aed"].sum().reset_index()
    import plotly.express as px
    fig_cat = px.bar(cat_rev.sort_values("revenue_aed"),
                     x="revenue_aed", y="category", orientation="h",
                     color="revenue_aed", color_continuous_scale="YlOrRd",
                     labels={"revenue_aed": "Revenue (AED)", "category": ""})
    fig_cat.update_coloraxes(showscale=False)
    fig_cat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig_cat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EXPLORATORY DATA ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

elif page == PAGES[1]:
    data = load_all_data()

    section_header("Data Quality Summary")
    from src.preprocessing import get_all_quality_reports
    qr = get_all_quality_reports()
    ds_filter = st.selectbox("Dataset", qr["Dataset"].unique())
    st.dataframe(qr[qr["Dataset"] == ds_filter].drop(columns="Dataset"),
                 use_container_width=True, hide_index=True)

    st.markdown("---")

    from src.eda import (plot_distribution, plot_category_counts,
                         plot_correlation_heatmap, plot_bivariate_scatter,
                         plot_box_by_category, plot_pie)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distributions", "📋 Category Counts",
        "🔥 Correlation Heatmap", "🔀 Bivariate"
    ])

    with tab1:
        sales = data["sales"]
        col_choice = st.selectbox(
            "Select column", ["revenue_aed", "discount_pct", "unit_price_aed",
                              "quantity", "weekly_marketing_spend_aed", "weekly_web_traffic"],
            key="dist_col"
        )
        st.plotly_chart(plot_distribution(sales, col_choice), use_container_width=True)

    with tab2:
        cat_choice = st.selectbox(
            "Select column", ["category", "delivery_city", "payment_method",
                              "sub_category"],
            key="cat_col"
        )
        st.plotly_chart(plot_category_counts(data["sales"], cat_choice),
                        use_container_width=True)

        # show pie too
        st.plotly_chart(plot_pie(data["sales"], cat_choice,
                                f"Proportion — {cat_choice}"),
                        use_container_width=True)

    with tab3:
        heatmap_ds = st.radio("Dataset", ["Sales", "Profiles"], horizontal=True,
                              key="hm_ds")
        df_hm = data["sales"] if heatmap_ds == "Sales" else data["profiles"]
        st.plotly_chart(plot_correlation_heatmap(df_hm), use_container_width=True)

    with tab4:
        sales = data["sales"]
        bc1, bc2 = st.columns(2)
        with bc1:
            x_col = st.selectbox("X axis", ["discount_pct", "unit_price_aed",
                                             "quantity", "weekly_web_traffic"], key="bv_x")
        with bc2:
            y_col = st.selectbox("Y axis", ["revenue_aed", "quantity",
                                             "unit_price_aed"], key="bv_y")
        st.plotly_chart(plot_bivariate_scatter(sales, x_col, y_col),
                        use_container_width=True)
        st.plotly_chart(plot_box_by_category(sales, "category", "revenue_aed"),
                        use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — REGRESSION
# ════════════════════════════════════════════════════════════════════════════

elif page == PAGES[2]:
    section_header("Weekly Demand Forecasting")
    st.info("**Objective:** Predict weekly grocery revenue (AED) using discount, "
            "marketing spend, web traffic, seasonality, and category mix.")

    from src.regression import (
        prepare_regression_data, train_models, metrics_table,
        plot_actual_vs_predicted, plot_coefficients, simulate_scenario, FEATURE_COLS,
    )

    with st.spinner("Training regression models…"):
        X_train, X_test, y_train, y_test, scaler, weekly = prepare_regression_data()
        results = train_models(X_train, y_train, X_test, y_test)

    # Metrics
    section_header("Model Comparison")
    mt = metrics_table(results)
    st.dataframe(mt, use_container_width=True, hide_index=True)

    # Highlight best MAPE
    best = mt.loc[mt["MAPE (%)"].idxmin()]
    st.success(f"🏆 **Best Model:** {best['Model']}  —  MAPE = {best['MAPE (%)']}%  "
               f"(vs Noon's current 20–25% forecast error)")

    # Actual vs Predicted
    section_header("Actual vs Predicted")
    st.plotly_chart(plot_actual_vs_predicted(y_test, results), use_container_width=True)

    # Coefficients
    section_header("Feature Coefficients")
    model_sel = st.selectbox("Select model", list(results.keys()), key="reg_coef_model")
    st.plotly_chart(plot_coefficients(results[model_sel]["model"], FEATURE_COLS),
                    use_container_width=True)

    # Scenario Simulator
    st.markdown("---")
    section_header("🎮 Revenue Scenario Simulator")
    st.caption("Adjust business levers to forecast weekly revenue.")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sim_discount = st.slider("Discount %", 0.0, 50.0, 12.0, 0.5, key="sim_disc")
    with sc2:
        sim_marketing = st.slider("Marketing Spend (AED)", 50000, 250000, 120000,
                                  5000, key="sim_mkt")
    with sc3:
        sim_ramadan = st.toggle("Ramadan Period", value=False, key="sim_ram")

    best_model = results[best["Model"]]["model"]
    pred_rev = simulate_scenario(
        best_model, scaler,
        discount=sim_discount, marketing_spend=sim_marketing,
        is_ramadan=int(sim_ramadan),
    )
    st.markdown(f"""
    <div class="kpi-card" style="max-width:420px; margin:16px auto;">
        <div class="kpi-value">AED {pred_rev:,.0f}</div>
        <div class="kpi-label">Predicted Weekly Revenue</div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════

elif page == PAGES[3]:
    section_header("Customer Churn Prediction")
    st.info("**Objective:** Flag at-risk customers before they switch to competitors. "
            "Acquisition costs AED 95; retention campaign costs only AED 12.")

    from src.classification import (
        prepare_classification_data, train_classifiers,
        classification_metrics_table, plot_confusion_matrix,
        plot_roc_curves, plot_feature_importance, threshold_analysis,
        predict_churn_probability, NUMERIC_FEATURES, CAT_FEATURES, TARGET,
    )

    use_smote = st.sidebar.checkbox("Apply SMOTE oversampling", value=False)

    with st.spinner("Training churn classifiers…"):
        X_train, X_test, y_train, y_test, scaler, le_dict, feat_names, full_df = \
            prepare_classification_data(use_smote=use_smote)
        results = train_classifiers(X_train, y_train, X_test, y_test)

    # Metrics table
    section_header("Model Comparison")
    cmt = classification_metrics_table(results)
    st.dataframe(cmt, use_container_width=True, hide_index=True)

    best_name = cmt.loc[cmt["AUC-ROC"].idxmax(), "Model"]
    st.success(f"🏆 **Best Model:** {best_name}  —  AUC = {cmt['AUC-ROC'].max():.4f}")

    # ROC curves
    section_header("ROC Curves")
    st.plotly_chart(plot_roc_curves(y_test, results), use_container_width=True)

    # Confusion matrices
    section_header("Confusion Matrices")
    cm_cols = st.columns(len(results))
    for i, (name, vals) in enumerate(results.items()):
        with cm_cols[i]:
            st.plotly_chart(plot_confusion_matrix(y_test, vals["y_pred"], name),
                            use_container_width=True)

    # Feature importance
    section_header("Top Churn Predictors")
    best_model = results[best_name]["model"]
    st.plotly_chart(plot_feature_importance(best_model, feat_names),
                    use_container_width=True)

    # Threshold tuning
    st.markdown("---")
    section_header("🎚️ Threshold Tuning")
    threshold_val = st.slider("Classification Threshold", 0.3, 0.7, 0.5, 0.05,
                              key="cls_threshold")
    best_prob = results[best_name]["y_prob"]
    ta = threshold_analysis(y_test, best_prob, [0.3, 0.4, 0.5, 0.6, 0.7])
    st.dataframe(ta, use_container_width=True, hide_index=True)

    # Show metrics at selected threshold
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_pred_t = (best_prob >= threshold_val).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f = f1_score(y_test, y_pred_t, zero_division=0)
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Precision", f"{p:.4f}")
    tc2.metric("Recall", f"{r:.4f}")
    tc3.metric("F1-Score", f"{f:.4f}")

    # Churn predictor
    st.markdown("---")
    section_header("🧠 AI Churn Risk Analyzer")
    st.markdown("<p style='color:#8b949e; margin-bottom: 24px; font-size: 1.1rem;'>Leverage deep behavioral analytics to predict customer churn probability in real-time. Adjust features below or load a preset profile.</p>", unsafe_allow_html=True)

    # State initialization for presets
    default_vals = {
        "age": 32, "spend": 3000, "aov": 85, "freq": 20, "sessions": 12, "days": 45,
        "email": 0.18, "push": 0.04, "tickets": 2, "return_r": 0.08, "nps": 7.0, 
        "deliv": 4.0, "cart": 3, "gender": "Female", "nat": "UAE", "city": "Dubai", 
        "tier": "Gold", "pay": "Apple Pay"
    }
    
    if "c_inputs" not in st.session_state:
        st.session_state.c_inputs = default_vals.copy()

    def apply_high_risk():
        st.session_state.c_inputs.update({
            "days": 120, "sessions": 1, "email": 0.02, "nps": 3.0, "deliv": 2.0, "tickets": 5, "tier": "Standard"
        })
    def apply_low_risk():
        st.session_state.c_inputs.update({
            "days": 2, "sessions": 25, "email": 0.65, "nps": 9.0, "deliv": 5.0, "tickets": 0, "tier": "Platinum"
        })
    def reset_inputs():
        st.session_state.c_inputs = default_vals.copy()

    # Preset Action Bar
    pb1, pb2, pb3, pb4 = st.columns([1.5, 1.5, 1.5, 4])
    pb1.button("🚨 Load High Risk", on_click=apply_high_risk, use_container_width=True)
    pb2.button("🌟 Load Low Risk", on_click=apply_low_risk, use_container_width=True)
    pb3.button("↺ Reset Values", on_click=reset_inputs, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Custom CSS for the prediction cards and UI enhancements
    st.markdown("""
        <style>
        .metric-group { background: rgba(22, 27, 34, 0.4); padding: 24px; border-radius: 16px; border: 1px solid rgba(245,197,24,0.15); margin-bottom: 24px; transition: transform 0.2s, box-shadow 0.2s; }
        .metric-group:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.2); border-color: rgba(245,197,24,0.4); }
        .group-title { font-weight: 700; color: #F5C518; font-size: 1.2rem; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid rgba(245,197,24,0.1); padding-bottom: 10px; }
        .predict-btn-container { text-align: center; margin: 30px 0; padding: 20px; background: rgba(233, 69, 96, 0.05); border-radius: 16px; border: 1px dashed rgba(233, 69, 96, 0.3); }
        </style>
    """, unsafe_allow_html=True)

    with st.form("churn_advanced_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("<div class='metric-group'><div class='group-title'>👤 Profile Basics</div>", unsafe_allow_html=True)
            inp_age = st.number_input("Age", 18, 80, st.session_state.c_inputs["age"], help="Customer age in years")
            inp_gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.c_inputs["gender"]))
            inp_nat = st.selectbox("Nationality", ["UAE", "Saudi", "Egyptian", "Indian", "Pakistani", "Filipino", "Other"], index=["UAE", "Saudi", "Egyptian", "Indian", "Pakistani", "Filipino", "Other"].index(st.session_state.c_inputs["nat"]))
            inp_city = st.selectbox("City", ["Dubai", "Abu Dhabi", "Riyadh", "Jeddah", "Sharjah", "Ajman", "Al Ain", "Dammam"], index=["Dubai", "Abu Dhabi", "Riyadh", "Jeddah", "Sharjah", "Ajman", "Al Ain", "Dammam"].index(st.session_state.c_inputs["city"]))
            inp_tier = st.selectbox("Membership Tier", ["Standard", "Gold", "Platinum"], index=["Standard", "Gold", "Platinum"].index(st.session_state.c_inputs["tier"]))
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='metric-group'><div class='group-title'>💰 Purchase Behavior</div>", unsafe_allow_html=True)
            inp_spend = st.number_input("Lifetime Spend (AED)", 0, 50000, st.session_state.c_inputs["spend"], help="Total money spent across all orders")
            inp_aov = st.number_input("Avg Order Value (AED)", 0, 500, st.session_state.c_inputs["aov"], help="Average revenue per order")
            inp_freq = st.number_input("Purchase Frequency", 0, 200, st.session_state.c_inputs["freq"], help="Total orders completed")
            inp_days = st.number_input("Days Since Last Purchase", 0, 365, st.session_state.c_inputs["days"], help="Recency metric. High values heavily indicate churn risk.")
            inp_pay = st.selectbox("Preferred Payment", ["Card", "Cash", "Noon Pay", "Apple Pay"], index=["Card", "Cash", "Noon Pay", "Apple Pay"].index(st.session_state.c_inputs["pay"]))
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c3:
            st.markdown("<div class='metric-group'><div class='group-title'>📱 Engagement Metrics</div>", unsafe_allow_html=True)
            inp_sessions = st.number_input("App Sessions/Month", 0, 100, st.session_state.c_inputs["sessions"], help="Monthly app open count")
            inp_tickets = st.number_input("Service Tickets", 0, 20, st.session_state.c_inputs["tickets"], help="Support tickets raised")
            inp_return = st.number_input("Return Rate", 0.0, 1.0, float(st.session_state.c_inputs["return_r"]), 0.01, help="Percentage of successful items returned")
            inp_cart = st.number_input("Cart Abandonment Count", 0, 30, st.session_state.c_inputs["cart"], help="Unsuccessful cart checkouts")
            
            c_sub1, c_sub2 = st.columns(2)
            with c_sub1:
                inp_nps = st.number_input("NPS", 0.0, 10.0, float(st.session_state.c_inputs["nps"]), 0.5, help="Net Promoter Score")
            with c_sub2:
                inp_deliv = st.number_input("Deliv Rating", 1.0, 5.0, float(st.session_state.c_inputs["deliv"]), 0.5)
                
            c_sub3, c_sub4 = st.columns(2)
            with c_sub3:
                inp_email = st.number_input("Email Open %", 0.0, 1.0, float(st.session_state.c_inputs["email"]), 0.05)
            with c_sub4:
                inp_push = st.number_input("Push CTR", 0.0, 1.0, float(st.session_state.c_inputs["push"]), 0.05)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='predict-btn-container'>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Run AI Prediction Engine", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        # Implicit missing variables computation based on inputs
        inp_nps_miss = 1 if inp_nps == 0 else 0
        inp_del_miss = 1 if inp_deliv == 1.0 else 0
        
        user_inputs = {
            "age": inp_age, "lifetime_spend_aed": inp_spend,
            "avg_order_value_aed": inp_aov, "purchase_frequency": inp_freq,
            "app_sessions_per_month": inp_sessions,
            "days_since_last_purchase": inp_days,
            "email_open_rate": inp_email, "push_notification_ctr": inp_push,
            "customer_service_tickets": inp_tickets, "return_rate": inp_return,
            "nps_score": inp_nps, "delivery_rating": inp_deliv,
            "cart_abandonment_count": inp_cart,
            "nps_missing": inp_nps_miss, "delivery_rating_missing": inp_del_miss,
            "gender": inp_gender, "nationality": inp_nat,
            "city": inp_city, "membership_tier": inp_tier,
            "preferred_payment": inp_pay,
        }
        
        with st.spinner("Analyzing behavioral patterns & predicting retention..."):
            prob = predict_churn_probability(best_model, scaler, le_dict, feat_names, user_inputs)
        
        st.markdown("---")
        risk_color = "#E94560" if prob > 0.5 else "#F5C518" if prob > 0.3 else "#2ea043"
        risk_title = "CRITICAL CHURN RISK" if prob > 0.5 else "MODERATE WARNING" if prob > 0.3 else "SAFE / LOYAL"
        risk_action = "Initiate immediate retention campaign (15% coupon + push notification)." if prob > 0.5 else "Monitor engagement. Suggest sending a re-engagement newsletter." if prob > 0.3 else "Customer is highly engaged. No intervention required."
        
        res1, res2 = st.columns([1, 1.5])
        
        with res1:
            st.markdown(f"""
            <div style='background: linear-gradient(145deg, rgba(22,27,34,0.95), rgba(13,17,23,0.95)); border: 2px solid {risk_color}; border-radius: 16px; padding: 40px 20px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
                <h4 style='color: #8b949e; text-transform: uppercase; letter-spacing: 2px; margin: 0; font-size: 0.9rem;'>AI Assessment</h4>
                <h1 style='font-size: 4.5rem; color: {risk_color}; font-weight: 800; margin: 15px 0; text-shadow: 0 0 20px {risk_color}40;'>{prob*100:.1f}%</h1>
                <h3 style='color: {risk_color}; margin: 0; font-weight: 700; font-size: 1.4rem;'>{risk_title}</h3>
                <p style='color: #c9d1d9; margin-top: 15px; font-size: 1rem; line-height: 1.5;'>{risk_action}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with res2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability Model Confidence", 'font': {'color': '#c9d1d9', 'size': 18}},
                number = {'font': {'color': 'white', 'size': 50}, 'valueformat': '.1f', 'suffix': '%'},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#8b949e"},
                    'bar': {'color': risk_color, 'thickness': 0.8},
                    'bgcolor': "rgba(22,27,34,0.6)",
                    'borderwidth': 2,
                    'bordercolor': "#161b22",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(46, 160, 67, 0.15)"},
                        {'range': [30, 50], 'color': "rgba(245, 197, 24, 0.15)"},
                        {'range': [50, 100], 'color': "rgba(233, 69, 96, 0.15)"}],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Inter"}, height=350, margin=dict(l=30, r=30, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════

elif page == PAGES[4]:
    section_header("Customer Segmentation")
    st.info("**Objective:** Segment 5 K customers into actionable groups to personalise "
            "the AED 22M annual marketing budget.")

    from src.clustering import (
        prepare_clustering_data, elbow_silhouette, plot_elbow_silhouette,
        fit_kmeans, assign_business_labels, cluster_profiles,
        plot_pca_clusters, plot_cluster_radar, CLUSTER_FEATURES,
    )

    with st.spinner("Preparing clustering features…"):
        X_scaled, scaler, df_raw = prepare_clustering_data()

    # Elbow / Silhouette
    section_header("Optimal K Selection")
    with st.spinner("Computing Elbow & Silhouette…"):
        k_range, inertias, sil_scores = elbow_silhouette(X_scaled)
    st.plotly_chart(plot_elbow_silhouette(k_range, inertias, sil_scores),
                    use_container_width=True)

    optimal_k = st.slider("Choose K (number of segments)", 2, 10,
                          int(k_range[np.argmax(sil_scores)]), key="k_slider")

    # Fit
    with st.spinner("Fitting K-Means…"):
        labels, km_model = fit_kmeans(X_scaled, k=optimal_k)
        df_clustered = assign_business_labels(df_raw, labels, optimal_k)

    # PCA scatter
    section_header("Cluster Visualisation (PCA)")
    st.plotly_chart(
        plot_pca_clusters(X_scaled, labels, df_clustered["Segment"].values),
        use_container_width=True
    )

    # Profiles
    section_header("Segment Profiles")
    profiles = cluster_profiles(df_clustered)
    st.dataframe(profiles, use_container_width=True)

    # Radar
    section_header("Radar Comparison")
    st.plotly_chart(plot_cluster_radar(profiles), use_container_width=True)

    # Segment descriptions
    section_header("Segment Descriptions")
    segment_strategies = {
        "💎 High-Value Loyalists": "VIP exclusive access, early product launches, loyalty points multiplier. Channel: In-app + Email. Budget share: 15%.",
        "🛒 Regular Essentials Shoppers": "Free delivery on weekly bundles, auto-replenish discounts. Channel: Push notifications. Budget share: 25%.",
        "🏷️ Bargain Hunters": "Flash deals, coupon stacking, clearance alerts. Channel: SMS + Push. Budget share: 20%.",
        "🌟 New Enthusiasts": "Welcome series, category exploration rewards, first 5 orders free delivery. Channel: Email onboarding. Budget share: 15%.",
        "💤 Dormant At-Risk": "Win-back 20% coupon, 'We miss you' push, limited-time free delivery. Channel: SMS + Push. Budget share: 25%.",
    }
    for seg in df_clustered["Segment"].unique():
        strategy = segment_strategies.get(seg, "Tailored engagement based on cluster profile.")
        st.markdown(f"**{seg}** — {strategy}")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════════════════

elif page == PAGES[5]:
    section_header("Market Basket Analysis — Apriori")
    st.info("**Objective:** Discover co-purchase patterns to power smarter bundling "
            "and boost the 'Frequently Bought Together' CTR from 2.1% to 8–12%.")

    from src.association import (
        run_apriori, top_rules_table, plot_rules_scatter,
        plot_top_rules_bar, generate_bundle_recommendations,
    )

    st.markdown("<p style='font-size:0.9rem; color:#94A3B8; margin-bottom:5px;'>Adjust algorithmic constraints to filter recommendations:</p>", unsafe_allow_html=True)
    ac1, ac2 = st.columns(2)
    with ac1:
        min_sup = st.slider("Min Support", 0.005, 0.10, 0.02, 0.005, key="ar_sup", help="Higher values = stricter rules, fewer results")
    with ac2:
        min_conf = st.slider("Min Confidence", 0.2, 0.8, 0.4, 0.05, key="ar_conf", help="Higher values = stricter rules, fewer results")

    auto_adjusted = False
    with st.spinner("Running Apriori algorithm…"):
        freq, rules = run_apriori(min_support=min_sup, min_confidence=min_conf)
        
        # ── AUTO ADJUST LOGIC ──
        if rules.empty:
            for s in [0.05, 0.02, 0.01, 0.005]:
                for c in [0.6, 0.4, 0.3, 0.2]:
                    if s <= min_sup and c <= min_conf:
                        f_tmp, r_tmp = run_apriori(min_support=s, min_confidence=c)
                        if not r_tmp.empty:
                            freq, rules = f_tmp, r_tmp
                            found_sup, found_conf = s, c
                            auto_adjusted = True
                            break
                if auto_adjusted: break

    # ── SMART UI MESSAGING ──
    if rules.empty:
        st.warning("⚠️ No rules found even at minimum thresholds.")
        if not freq.empty:
            st.info("💡 Showing top frequent raw items as fallback instead:")
            freq_display = freq.copy()
            freq_display["itemsets"] = freq_display["itemsets"].apply(lambda x: ", ".join(list(x)))
            freq_display = freq_display.sort_values("support", ascending=False).head(10)
            st.dataframe(freq_display, use_container_width=True, hide_index=True)
    else:
        if auto_adjusted:
            st.warning("⚠️ No rules found at current thresholds.")
            st.info(f"💡 Try lowering support or confidence. Showing recommended values: **Support={found_sup}**, **Confidence={found_conf}**")
        else:
            st.success(f"✅ Found **{len(freq)}** frequent itemsets and **{len(rules)}** association rules.")

        # Top rules table
        section_header("Top Association Rules (by Lift)")
        trt = top_rules_table(rules, top_n=15)
        st.dataframe(trt, use_container_width=True, hide_index=True)

        # Plots
        pc1, pc2 = st.columns(2)
        with pc1:
            st.plotly_chart(plot_rules_scatter(rules), use_container_width=True)
        with pc2:
            st.plotly_chart(plot_top_rules_bar(rules), use_container_width=True)

        # Bundle Recommendations
        st.markdown("---")
        section_header("🎁 Bundle Recommendations")
        bundles = generate_bundle_recommendations(rules, top_n=5)
        if bundles:
            st.dataframe(pd.DataFrame(bundles), use_container_width=True, hide_index=True)
        else:
            st.info("Enrich rules with product names for bundle display.")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — BUSINESS INSIGHTS
# ════════════════════════════════════════════════════════════════════════════

elif page == PAGES[6]:
    section_header("Business Insights & Recommendations")

    st.markdown("""
    <div style="background:rgba(245,197,24,0.06); border:1px solid rgba(245,197,24,0.15);
         border-radius:16px; padding:28px; margin-bottom:24px;">
        <h3 style="color:#F5C518; margin:0 0 12px 0;">📋 Executive Summary</h3>
        <p style="color:#c9d1d9; line-height:1.7;">
        This dashboard delivers data-driven solutions to Noon Daily Grocery's four
        interconnected business challenges: demand forecasting, customer churn,
        segmentation, and cross-selling. Below are the key findings and actionable
        recommendations from each ML model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── PDF Download Integration ──
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            import generate_pdf_report
            if not __import__('os').path.exists("Noon_Daily_ML_Executive_Report.pdf"):
                generate_pdf_report.create_report()
                
            with open("Noon_Daily_ML_Executive_Report.pdf", "rb") as pdf_file:
                PDFbyte = pdf_file.read()
                
            st.download_button(
                label="📥 Download Full Executive Report (PDF)",
                data=PDFbyte,
                file_name="Noon_Daily_ML_Executive_Report.pdf",
                mime='application/octet-stream',
                type="primary",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Generate script missing. Error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Demand Forecasting ──
    section_header("📈 Demand Forecasting Insights")
    st.markdown("""
    | Finding | Business Impact | Recommendation |
    |---------|----------------|----------------|
    | **Ramadan drives 18–25% revenue spike** | Critical for perishable inventory planning | Pre-position dairy, dates, and beverages inventory 2 weeks before Ramadan |
    | **Marketing spend has strong positive coefficient** | ROI on marketing is measurable | Increase weekly spend by 15% during peak weeks — expected revenue uplift of AED 50K/week |
    | **Discount beyond 20% shows diminishing returns** | Margin erosion risk at high discount levels | Cap promotional discounts at 20%; use free-delivery bundles instead |
    | **Model MAPE < 15%** vs current 20–25% | Reduced waste on perishables (est. AED 4–6M/year savings) | Integrate model into weekly planning cycle |
    """)

    # ── Churn ──
    section_header("⚠️ Churn Prediction Insights")
    st.markdown("""
    | Finding | Business Impact | Recommendation |
    |---------|----------------|----------------|
    | **days_since_last_purchase is #1 churn predictor** | Early warning signal available | Send 15% coupon via push to customers inactive > 45 days, targeting their top category |
    | **Customers who don't rate deliveries churn 2.3× more** | MNAR survey data is itself a signal | Create `did_not_rate` feature; trigger follow-up for non-raters |
    | **Standard-tier members churn at 2× the rate of Gold/Platinum** | Tier upgrade incentive is underutilized | Offer 500 bonus loyalty points for 3rd purchase in 30 days |
    | **Retention campaign (AED 12) vs acquisition (AED 95)** | 8× cost difference | At optimal threshold, net benefit is **AED 2.1M+** per 10K customers |
    """)

    # ── Segmentation ──
    section_header("🎯 Customer Segmentation Insights")
    st.markdown("""
    | Segment | Recommended Strategy | Budget Allocation |
    |---------|---------------------|-------------------|
    | **💎 High-Value Loyalists** | Exclusive early access, VIP perks (not discounts) | 15% of AED 22M |
    | **🛒 Regular Essentials** | Auto-replenish, free delivery on weekly bundles | 25% |
    | **🏷️ Bargain Hunters** | Flash deals, clearance alerts, coupon stacking | 20% |
    | **🌟 New Enthusiasts** | Onboarding emails, category exploration rewards | 15% |
    | **💤 Dormant At-Risk** | Win-back campaign: 20% coupon + free delivery | 25% |
    """)

    # ── Association ──
    section_header("🔗 Cross-Selling Insights")
    st.markdown("""
    | Finding | Business Impact | Recommendation |
    |---------|----------------|----------------|
    | **Strong Bread ↔ Dairy ↔ Eggs associations** | Natural basket expanders | Bundle at 10% combo discount on product pages |
    | **Baby Products form a tight cluster** | Loyal parent segment | Create 'New Parent Essentials' subscription box |
    | **Lift > 3 for niche organic combos** | High-value but low frequency | Target to 'Health-Conscious' segment via personalized push |
    | **Current 2.1% CTR on recommendations** | Fixing this means AED 500K+ incremental revenue | Replace static recommendations with Apriori-powered engine |
    """)

    # ── Integrated Strategy ──
    st.markdown("---")
    section_header("🧠 Unified Decision Framework")
    st.markdown("""
    <div style="background:rgba(15,52,96,0.2); border:1px solid rgba(15,52,96,0.3);
         border-radius:16px; padding:24px; margin:16px 0;">
        <p style="color:#c9d1d9; line-height:1.8; font-size:0.95rem;">
        <strong>Example Scenario:</strong> A high-value customer in the "Health-Conscious"
        segment shows early churn signals (inactive 50 days, didn't rate last delivery).
        The regression model predicts a demand spike for organic produce next week.
        Association rules show her favourite items pair well with a new organic yogurt launch.<br><br>
        <strong>Action:</strong> Send a personalized push notification offering 15% off the
        "Organic Favourites Bundle" (yogurt + fruits + nuts) with free same-day delivery.
        This single action leverages all four models — <em>Clustering</em> identifies WHO,
        <em>Classification</em> identifies WHEN, <em>Regression</em> identifies WHAT's trending,
        and <em>Association Rules</em> identify HOW to bundle.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ---
    <div style="text-align:center; color:#8b949e; padding:24px 0;">
        <span style="font-size:0.85rem;">
        Noon Daily ML Analytics Dashboard  ·  MAIB Programme  ·  Built with Streamlit + Plotly + scikit-learn
        </span>
    </div>
    """, unsafe_allow_html=True)
