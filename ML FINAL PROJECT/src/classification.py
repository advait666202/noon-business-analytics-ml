"""
classification.py — Customer Churn Prediction for Noon Daily.

Models: Logistic Regression, Random Forest, Gradient Boosting.
Target: 'churned' (1 = no purchase in 90 days, 0 = active).
Handles class imbalance via SMOTE + class_weight='balanced'.
Includes threshold tuning and churn probability predictor.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
)

from src.preprocessing import merge_customer_data
from src.eda import _apply_theme, NOON_YELLOW, NOON_ACCENT, NOON_TEAL, PALETTE


# ═══════════════════════════════════════════════════════════════
# 1.  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

NUMERIC_FEATURES = [
    "age", "lifetime_spend_aed", "avg_order_value_aed",
    "purchase_frequency", "app_sessions_per_month",
    "days_since_last_purchase", "email_open_rate",
    "push_notification_ctr", "customer_service_tickets",
    "return_rate", "nps_score", "delivery_rating",
    "cart_abandonment_count", "nps_missing", "delivery_rating_missing",
]

CAT_FEATURES = ["gender", "nationality", "city", "membership_tier", "preferred_payment"]

TARGET = "churned"


def prepare_classification_data(use_smote: bool = False):
    """
    Merge profiles + engagement → encode → scale → split.
    Returns X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names, full_df.
    """
    df = merge_customer_data()

    # encode categoricals
    label_encoders = {}
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    feature_cols = NUMERIC_FEATURES + CAT_FEATURES
    df = df.dropna(subset=feature_cols + [TARGET])

    X = df[feature_cols].values
    y = df[TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # optional SMOTE oversampling on training set
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        except ImportError:
            pass  # fall back to class_weight if imblearn not installed

    return X_train, X_test, y_train, y_test, scaler, label_encoders, feature_cols, df


# ═══════════════════════════════════════════════════════════════
# 2.  TRAINING
# ═══════════════════════════════════════════════════════════════

def train_classifiers(X_train, y_train, X_test, y_test):
    """Train three classifiers and return metrics + fitted models."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "AUC-ROC": round(roc_auc_score(y_test, y_prob), 4),
        }
    return results


# ═══════════════════════════════════════════════════════════════
# 3.  METRICS TABLE
# ═══════════════════════════════════════════════════════════════

def classification_metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, vals in results.items():
        rows.append({
            "Model": name,
            "Accuracy": vals["Accuracy"],
            "Precision": vals["Precision"],
            "Recall": vals["Recall"],
            "F1-Score": vals["F1"],
            "AUC-ROC": vals["AUC-ROC"],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 4.  CONFUSION MATRIX PLOT
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_test, y_pred, model_name: str) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Active", "Churned"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, NOON_TEAL], [1, NOON_ACCENT]],
        text=cm, texttemplate="%{text}",
        textfont=dict(size=18),
        showscale=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted", yaxis_title="Actual",
        height=400, width=450,
    )
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 5.  ROC CURVE
# ═══════════════════════════════════════════════════════════════

def plot_roc_curves(y_test, results: dict) -> go.Figure:
    """Overlay ROC curves for all models."""
    fig = go.Figure()
    colors = [NOON_YELLOW, NOON_ACCENT, NOON_TEAL]
    for i, (name, vals) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, vals["y_prob"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f'{name} (AUC={vals["AUC-ROC"]:.3f})',
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(dash="dash", color="gray"),
    ))
    fig.update_layout(
        title="ROC Curves — Churn Classifiers",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 6.  FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def plot_feature_importance(model, feature_names: list, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return go.Figure()

    df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    df = df.nlargest(top_n, "Importance").sort_values("Importance")
    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="YlOrRd",
        title=f"Top {top_n} Churn Predictors",
    )
    fig.update_coloraxes(showscale=False)
    return _apply_theme(fig)


# ═══════════════════════════════════════════════════════════════
# 7.  THRESHOLD ANALYSIS
# ═══════════════════════════════════════════════════════════════

def threshold_analysis(y_test, y_prob, thresholds=None) -> pd.DataFrame:
    """Precision / Recall / F1 at different classification thresholds."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    rows = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        rows.append({
            "Threshold": t,
            "Precision": round(precision_score(y_test, y_pred_t, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred_t, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred_t, zero_division=0), 4),
        })
    return pd.DataFrame(rows)


def predict_churn_probability(model, scaler, label_encoders, feature_names,
                              user_inputs: dict) -> float:
    """
    Predict churn probability for a single customer from dashboard inputs.
    user_inputs: dict mapping feature name → raw value.
    """
    row = []
    for feat in feature_names:
        val = user_inputs.get(feat, 0)
        if feat in label_encoders:
            le = label_encoders[feat]
            val = le.transform([str(val)])[0] if str(val) in le.classes_ else 0
        row.append(float(val))

    row_scaled = scaler.transform([row])
    prob = model.predict_proba(row_scaled)[0][1]
    return float(prob)
