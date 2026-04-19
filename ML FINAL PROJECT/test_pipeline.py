"""Quick smoke test for all modules."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("1. Testing preprocessing...", flush=True)
from src.preprocessing import load_sales, aggregate_weekly, merge_customer_data
sales = load_sales()
print(f"   Sales: {sales.shape}", flush=True)
weekly = aggregate_weekly(sales)
print(f"   Weekly: {weekly.shape}", flush=True)
cust = merge_customer_data()
print(f"   Customer merged: {cust.shape}", flush=True)

print("\n2. Testing regression...", flush=True)
from src.regression import prepare_regression_data, train_models
Xtr, Xte, ytr, yte, sc, wk = prepare_regression_data()
print(f"   Train: {Xtr.shape}, Test: {Xte.shape}", flush=True)
res = train_models(Xtr, ytr, Xte, yte)
for n, v in res.items():
    print(f"   {n}: MAPE={v['MAPE (%)']:.1f}%", flush=True)

print("\n3. Testing classification...", flush=True)
from src.classification import prepare_classification_data, train_classifiers
Xtr, Xte, ytr, yte, sc2, le, fn, df = prepare_classification_data()
print(f"   Train: {Xtr.shape}, Test: {Xte.shape}", flush=True)
res2 = train_classifiers(Xtr, ytr, Xte, yte)
for n, v in res2.items():
    print(f"   {n}: AUC={v['AUC-ROC']:.4f}", flush=True)

print("\n4. Testing clustering...", flush=True)
from src.clustering import prepare_clustering_data, fit_kmeans, elbow_silhouette
X, sc3, df3 = prepare_clustering_data()
labels, km = fit_kmeans(X, k=4)
print(f"   {len(set(labels))} clusters, {len(labels)} customers", flush=True)

print("\n5. Testing association...", flush=True)
from src.association import run_apriori, build_transaction_matrix
print("   Building transaction matrix...", flush=True)
mat = build_transaction_matrix()
print(f"   Matrix: {mat.shape}", flush=True)
print("   Running Apriori (min_support=0.02)...", flush=True)
freq, rules = run_apriori(min_support=0.02, min_confidence=0.4)
print(f"   Frequent itemsets: {len(freq)}, Rules: {len(rules)}", flush=True)

print("\n=== ALL TESTS PASSED ===", flush=True)
