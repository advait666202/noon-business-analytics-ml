# 🛒 Noon Daily Grocery — ML Analytics Dashboard

> **Industry-grade, interactive machine learning dashboard** built for Noon Daily Grocery's Chief Strategy Officer. Delivers data-driven solutions to four interconnected business challenges using real transactional data.

---

## 📋 Business Problem

Noon Daily, the grocery and fresh delivery arm of the Middle East's largest e-commerce platform, faces four critical challenges:

| Challenge | Impact | ML Solution |
|-----------|--------|-------------|
| **Poor Demand Forecasting** | ~AED 18M/year in food waste + stockouts | Regression (Linear, Ridge, Lasso) |
| **Rising Customer Churn** | 30% churn; acquisition costs 8× retention | Classification (Logistic, RF, GBM) |
| **No Customer Segmentation** | AED 22M marketing budget spent uniformly | Clustering (K-Means + PCA) |
| **Missed Cross-selling** | 2.1% CTR vs 8–12% industry average | Association Rules (Apriori) |

---

## ✨ Features

### Dashboard Pages
1. **📊 Executive Overview** — KPIs, revenue trends with Ramadan highlights, interactive filters (city, category, date)
2. **🔍 Exploratory Data Analysis** — Data quality audit, distributions, correlation heatmap, bivariate plots
3. **📈 Regression** — Weekly demand forecasting with 3 models, actual-vs-predicted plots, revenue scenario simulator
4. **⚠️ Classification** — Churn prediction with 3 classifiers, ROC curves, confusion matrices, threshold slider, individual churn predictor
5. **🎯 Clustering** — Customer segmentation with elbow/silhouette, PCA scatter, radar profiles, business labels
6. **🔗 Association Rules** — Apriori basket analysis, top rules by lift, automated bundle recommendations
7. **💡 Business Insights** — Executive summary with actionable recommendations and estimated impact

### ML Models
- **Regression:** Linear, Ridge (α=1.0), Lasso (α=1.0) — R², Adj R², RMSE, MAPE
- **Classification:** Logistic Regression, Random Forest, Gradient Boosting — Accuracy, Precision, Recall, F1, AUC-ROC
- **Clustering:** K-Means (K=2–10) with Elbow + Silhouette — PCA visualisation
- **Association:** Apriori with configurable support/confidence — Lift-ranked rules

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Dashboard** | Streamlit |
| **Visualisation** | Plotly |
| **ML / Statistics** | scikit-learn, imbalanced-learn |
| **Association Rules** | mlxtend |
| **Data Processing** | pandas, NumPy |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
noon_dashboard/
│
├── app.py                          # Main Streamlit multi-page app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/                           # Datasets (CSV)
│   ├── noon_sales_transactions.csv
│   ├── noon_customer_profiles.csv
│   ├── noon_customer_engagement.csv
│   ├── noon_product_catalogue.csv
│   └── noon_market_basket.csv
│
└── src/                            # Modular source code
    ├── __init__.py
    ├── preprocessing.py            # Data loading, cleaning, feature engineering
    ├── eda.py                      # Plotly visualisation functions
    ├── regression.py               # Demand forecasting models
    ├── classification.py           # Churn prediction models
    ├── clustering.py               # Customer segmentation
    └── association.py              # Apriori basket analysis
```

---

## 🚀 Setup Instructions

### 1. Clone / Download the project

```bash
git clone <repository-url>
cd noon_dashboard
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📊 Dataset Description

| File | Records | Description |
|------|---------|-------------|
| `noon_sales_transactions.csv` | 10,000 | Orders with revenue, discounts, categories, Ramadan/weekend flags |
| `noon_customer_profiles.csv` | 5,000 | Demographics, spend behaviour, churn label |
| `noon_customer_engagement.csv` | 5,000 | Email/push engagement, NPS, delivery ratings (MNAR) |
| `noon_product_catalogue.csv` | 500 | Product details, brands, ratings, shelf life |
| `noon_market_basket.csv` | 8,000 | Co-purchased product IDs per transaction |

**Date range:** July 2022 – June 2024  
**Geography:** 8 cities across UAE & Saudi Arabia

---

## 🔑 Key Insights

1. **Ramadan drives 18–25% revenue spike** — pre-position dairy, dates, and beverages inventory
2. **Model MAPE < 15%** vs Noon's current 20–25% forecast error — potential AED 4–6M/year savings
3. **`days_since_last_purchase` is the #1 churn predictor** — activate retention at 45-day threshold
4. **Non-raters churn 2.3× more** — MNAR itself is a powerful engagement signal
5. **Retention campaign ROI is 8×** acquisition cost — net benefit of AED 2.1M+ per 10K customers
6. **5 distinct customer segments identified** — enabling personalised budget allocation
7. **Strong Bread ↔ Dairy ↔ Eggs basket associations** — bundle for 10% combo discount

---

## ☁️ Deployment Guide (Streamlit Cloud)

1. Push your project to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** and select your repository
4. Set:
   - **Main file path:** `app.py`
   - **Python version:** 3.10+
5. Click **Deploy** — your dashboard will be live in ~2 minutes

> **Note:** Ensure `requirements.txt` is in the root of your repository. Streamlit Cloud reads it automatically to install dependencies.

---

## 🔮 Future Improvements

- [ ] **Deep Learning:** LSTM / Prophet for time-series demand forecasting
- [ ] **Real-time data:** Connect to Noon's API for live transaction streaming
- [ ] **A/B Testing module:** Track impact of Apriori-powered bundles vs static recommendations
- [ ] **NLP module:** Sentiment analysis on customer reviews for product-level insights
- [ ] **Seasonal Apriori:** Separate Ramadan vs non-Ramadan basket analysis
- [ ] **Cost-benefit calculator:** Interactive financial model for churn intervention ROI
- [ ] **Egypt expansion module:** Transfer learning analysis for Cairo/Alexandria market entry

---

## 📜 License

This project was developed as part of the **MAIB Programme** (Master of Artificial Intelligence in Business) assignment.

---

<p align="center">
  Built with ❤️ using Streamlit · Plotly · scikit-learn
</p>
