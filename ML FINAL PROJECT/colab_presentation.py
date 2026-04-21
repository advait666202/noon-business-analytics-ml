# colab_presentation.py
"""
Noon Daily ML Dashboard - Google Colab Presentation Runner
==========================================================
Instructions for Google Colab:
1. Zip and Upload the entire 'ML FINAL PROJECT' folder to your Google Drive, then unzip it.
2. In Colab, create a new cell and mount your drive:
   from google.colab import drive
   drive.mount('/content/drive')
3. Change directory to the project folder:
   %cd /content/drive/MyDrive/path_to_your_project/ML FINAL PROJECT
4. You can run this entire script using:
   !python colab_presentation.py
   
   OR, for the best presentation experience, copy each SECTION below into its 
   own separate Colab cell and run them one by one. Uncomment the `fig.show()` 
   lines to display the interactive Plotly graphs directly in Colab!
"""

print("Initializing libraries...")
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 for Windows machines printing emojis
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# CRITICAL for Plotly to render interactively directly inside Colab notebook cells
import plotly.io as pio
pio.renderers.default = "colab"

# IMPORT CORE LOGIC FROM OUR SOURCE FILES
from src.preprocessing import load_sales, load_profiles
from src.eda import plot_revenue_trend, plot_correlation_heatmap
from src.regression import prepare_regression_data, train_models, metrics_table
from src.classification import prepare_classification_data, train_classifiers, classification_metrics_table
from src.clustering import prepare_clustering_data, fit_kmeans, assign_business_labels, cluster_profiles
from src.association import run_apriori

# =====================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# =====================================================================
print("\n" + "="*60)
print("SECTION 1: DATA LOADING & EXECUTIVE OVERVIEW")
print("="*60)
sales = load_sales()
profiles = load_profiles()
print(f"✅ Data Loaded Successfully.")
print(f"- Sales Dataset Shape: {sales.shape}")
print(f"- Profiles Dataset Shape: {profiles.shape}")
print(f"\n📊 Key Performance Indicators:")
print(f"Total Revenue Generated: AED {sales['revenue_aed'].sum():,.0f}")
print(f"Platform Churn Rate:     {profiles['churned'].mean() * 100:.1f}%")


# =====================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================
print("\n" + "="*60)
print("SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)
print("Generating Correlation Heatmap for Sales variables...")
fig_corr = plot_correlation_heatmap(sales)
# To view the plot in Colab, uncomment the line below:
# fig_corr.show()
print("-> (Uncomment `fig_corr.show()` in Colab to view the interactive plot)")


# =====================================================================
# SECTION 3: DEMAND FORECASTING (REGRESSION)
# =====================================================================
print("\n" + "="*60)
print("SECTION 3: DEMAND FORECASTING (REGRESSION)")
print("="*60)
X_train, X_test, y_train, y_test, scaler, weekly = prepare_regression_data()
reg_results = train_models(X_train, y_train, X_test, y_test)
mt = metrics_table(reg_results)
print("Regression Model Comparison (Predicting Weekly Revenue):")
print("-" * 50)
print(mt.to_string(index=False))
print("-" * 50)
best_reg = mt.loc[mt["MAPE (%)"].idxmin()]
print(f"🏆 Best Model: {best_reg['Model']} with MAPE {best_reg['MAPE (%)']}%")


# =====================================================================
# SECTION 4: CHURN PREDICTION (CLASSIFICATION)
# =====================================================================
print("\n" + "="*60)
print("SECTION 4: CHURN PREDICTION (CLASSIFICATION)")
print("="*60)
cX_train, cX_test, cy_train, cy_test, c_scaler, le_dict, feat_names, full_df = prepare_classification_data(use_smote=True)
clf_results = train_classifiers(cX_train, cy_train, cX_test, cy_test)
cmt = classification_metrics_table(clf_results)
print("Classification Model Comparison (Predicting Customer Churn):")
print("-" * 50)
print(cmt.to_string(index=False))
print("-" * 50)
best_clf = cmt.loc[cmt["AUC-ROC"].idxmax()]
print(f"🏆 Best Model: {best_clf['Model']} with AUC-ROC {best_clf['AUC-ROC']:.4f}")


# =====================================================================
# SECTION 5: CUSTOMER SEGMENTATION (CLUSTERING)
# =====================================================================
print("\n" + "="*60)
print("SECTION 5: CUSTOMER SEGMENTATION (CLUSTERING)")
print("="*60)
X_scaled, km_scaler, df_raw = prepare_clustering_data()
# Using 5 clusters as our optimal K
labels, km_model = fit_kmeans(X_scaled, k=5)
df_clustered = assign_business_labels(df_raw, labels, 5)
segment_profiles = cluster_profiles(df_clustered)
print("Discovered Customer Segment Profiles:")
print("-" * 50)
print(segment_profiles.to_string())
print("-" * 50)


# =====================================================================
# SECTION 6: MARKET BASKET ANALYSIS (ASSOCIATION RULES)
# =====================================================================
print("\n" + "="*60)
print("SECTION 6: MARKET BASKET ANALYSIS (ASSOCIATION RULES)")
print("="*60)
print("Running Apriori Algorithm (Support=0.02, Confidence=0.4)...")
freq, rules = run_apriori(min_support=0.02, min_confidence=0.4)
if not rules.empty:
    print(f"\n✅ Discovered {len(rules)} Strong Association Rules.")
    print("Top 5 Rules by Lift (Strongest Product Relationships):")
    cols_to_show = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    print("-" * 80)
    print(rules[cols_to_show].head(5).to_string(index=False))
    print("-" * 80)
else:
    print("No rules found at this threshold.")

print("\n" + "="*60)
print("🏁 PRESENTATION SCRIPT EXECUTION COMPLETE.")
print("="*60)
