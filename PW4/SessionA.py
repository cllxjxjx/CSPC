import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("data/concrete.csv")

print("--- Missing Values per Column ---")
print(df.isna().sum())

n_duplicates = df.duplicated().sum()
print(f"\nDuplicate rows found: {n_duplicates}")

df_clean = df.drop_duplicates().reset_index(drop=True)
print(f"Shape before cleaning: {df.shape}")
print(f"Shape after removing duplicates: {df_clean.shape}")

X = df_clean.drop(columns=["strength"])
y = df_clean["strength"]

print("\n--- Summary Statistics (df_clean.describe()) ---")
print(df_clean.describe().round(2))

os.makedirs("plots", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X["cement"], y, alpha=0.5, color="navy")
ax1.set_xlabel("Cement (kg/m³)")
ax1.set_ylabel("Compressive Strength (MPa)")
ax1.set_title("Strength vs Cement")
ax1.grid(True)

ax2.scatter(X["age"], y, alpha=0.5, color="darkgreen")
ax2.set_xlabel("Age (days)")
ax2.set_ylabel("Compressive Strength (MPa)")
ax2.set_title("Strength vs Age")
ax2.grid(True)

plt.tight_layout()
plt.savefig("plots/strength_vs_cement_age.png")
print("\nSaved plot to plots/strength_vs_cement_age.png")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n--- A3. Train / Test Split Sizes ---")
print(f"X_train size: {X_train.shape[0]} samples ({X_train.shape[0]/len(df_clean)*100:.1f}%)")
print(f"X_test size : {X_test.shape[0]} samples ({X_test.shape[0]/len(df_clean)*100:.1f}%)")

lin_reg = LinearRegression(fit_intercept=True)
lin_reg.fit(X_train, y_train)

y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

n = X_train.shape[0]
p = X_train.shape[1]
train_r2_adj = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)

print("\n--- A4. Linear Regression Results ---")
print(f"Intercept (beta_0): {lin_reg.intercept_:.4f} MPa")
for col, coef in zip(X_train.columns, lin_reg.coef_):
    unit = "MPa/day" if col == "age" else "MPa/(kg/m³)"
    print(f"{col:12s}: {coef:+.4f} {unit}")

print(f"\nTrain RMSE   : {train_rmse:.4f} MPa")
print(f"Test RMSE    : {test_rmse:.4f} MPa")
print(f"Test MAE     : {test_mae:.4f} MPa")
print(f"Train R²     : {train_r2:.4f}")
print(f"Train Adj R² : {train_r2_adj:.4f}")
print(f"Test R²      : {test_r2:.4f}")

residuals = y_test - y_test_pred

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X_test["age"], residuals, alpha=0.6, color="purple")
ax.axhline(0, color="red", linestyle="--")
ax.set_xscale("log")
ax.set_xlabel("Age (days, log scale)")
ax.set_ylabel("Residuals (True - Predicted Strength) [MPa]")
ax.set_title("Linear Regression: Residuals vs Age (Log Scale)")
ax.grid(True)

plt.tight_layout()
plt.savefig("plots/residuals_vs_age.png")
print("Saved residuals plot to plots/residuals_vs_age.png")

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

print("\n--- A5. Polynomial Regression (Degrees 1 to 4) ---")
results = []
n_train = X_train.shape[0]

for d in range(1, 5):
    pipeline = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    pipeline.fit(X_train, y_train)
    
    p_features = pipeline[0].n_output_features_
    
    y_tr_pred = pipeline.predict(X_train)
    y_te_pred = pipeline.predict(X_test)
    
    tr_rmse = np.sqrt(mean_squared_error(y_train, y_tr_pred))
    te_rmse = np.sqrt(mean_squared_error(y_test, y_te_pred))
    
    tr_r2 = r2_score(y_train, y_tr_pred)
    te_r2 = r2_score(y_test, y_te_pred)
    
    if n_train - p_features - 1 > 0:
        tr_r2_adj = 1 - (1 - tr_r2) * (n_train - 1) / (n_train - p_features - 1)
    else:
        tr_r2_adj = np.nan
        
    results.append({
        "Degree": d,
        "p (Features)": p_features,
        "Train RMSE": tr_rmse,
        "Test RMSE": te_rmse,
        "Train R2": tr_r2,
        "Train Adj R2": tr_r2_adj,
        "Test R2": te_r2
    })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

y_class = (y >= 40).astype(int)

Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print("\n--- A6. KNN Classification ---")
k_values = [1, 3, 5, 7, 15, 25, 51, 101]
knn_results = []

for K in k_values:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K))
    pipe.fit(Xc_train, c_train)
    tr_acc = pipe.score(Xc_train, c_train)
    te_acc = pipe.score(Xc_test, c_test)
    knn_results.append({"K": K, "Train Accuracy": tr_acc, "Test Accuracy": te_acc})

df_knn = pd.DataFrame(knn_results)
print(df_knn.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df_knn["K"], df_knn["Train Accuracy"], marker='o', label="Train Accuracy")
ax.plot(df_knn["K"], df_knn["Test Accuracy"], marker='s', label="Test Accuracy")
ax.set_xscale("log")
ax.set_xlabel("K (log scale)")
ax.set_ylabel("Accuracy")
ax.set_title("KNN: Accuracy vs K")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("plots/knn_accuracy_vs_k.png")

best_k = 7
knn_scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
knn_scaled.fit(Xc_train, c_train)
acc_scaled = knn_scaled.score(Xc_test, c_test)

knn_unscaled = KNeighborsClassifier(n_neighbors=best_k)
knn_unscaled.fit(Xc_train, c_train)
acc_unscaled = knn_unscaled.score(Xc_test, c_test)

print(f"\nScaled KNN (K={best_k}) Test Accuracy  : {acc_scaled:.4f}")
print(f"Unscaled KNN (K={best_k}) Test Accuracy: {acc_unscaled:.4f}")

split_accs = []
for rs in range(5):
    X_tr_s, X_te_s, c_tr_s, c_te_s = train_test_split(
        X, y_class, test_size=0.2, random_state=rs, stratify=y_class
    )
    model_s = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
    model_s.fit(X_tr_s, c_tr_s)
    split_accs.append(model_s.score(X_te_s, c_te_s))

print(f"\n5-Split Test Accuracies (K={best_k}): {[round(a, 4) for a in split_accs]}")
print(f"Mean: {np.mean(split_accs):.4f}, Std: {np.std(split_accs):.4f}, Range: [{min(split_accs):.4f}, {max(split_accs):.4f}]")

print("\n--- A6. Decision Tree Classification ---")
depths = [1, 2, 3, 4, 5, 6, 8, None]
dt_results = []

for d in depths:
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=d, min_samples_leaf=1, random_state=0)
    tree.fit(Xc_train, c_train)
    n_leaves = tree.get_n_leaves()
    tr_acc = tree.score(Xc_train, c_train)
    te_acc = tree.score(Xc_test, c_test)
    dt_results.append({"max_depth": str(d), "Leaves": n_leaves, "Train Accuracy": tr_acc, "Test Accuracy": te_acc})

df_dt = pd.DataFrame(dt_results)
print(df_dt.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 5))
x_plot = [d if d is not None else 10 for d in depths]
ax.plot(x_plot, df_dt["Train Accuracy"], marker='o', label="Train Accuracy")
ax.plot(x_plot, df_dt["Test Accuracy"], marker='s', label="Test Accuracy")
ax.set_xticks(x_plot)
ax.set_xticklabels([str(d) for d in depths])
ax.set_xlabel("Max Depth")
ax.set_ylabel("Accuracy")
ax.set_title("Decision Tree: Accuracy vs Depth")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("plots/dt_accuracy_vs_depth.png")

tree_depth2 = DecisionTreeClassifier(criterion="entropy", max_depth=2, min_samples_leaf=1, random_state=0)
tree_depth2.fit(Xc_train, c_train)
print("\n--- Top 2 Levels of Decision Tree ---")
print(export_text(tree_depth2, feature_names=list(X.columns)))

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

os.makedirs("results", exist_ok=True)

knn_chosen = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
knn_chosen.fit(Xc_train, c_train)
y_pred_knn = knn_chosen.predict(Xc_test)
tn_k, fp_k, fn_k, tp_k = confusion_matrix(c_test, y_pred_knn).ravel()

tree_chosen = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=1, random_state=0)
tree_chosen.fit(Xc_train, c_train)
y_pred_tree = tree_chosen.predict(Xc_test)
tn_t, fp_t, fn_t, tp_t = confusion_matrix(c_test, y_pred_tree).ravel()

metrics_data = [
    {
        "Model": "KNN (K=7)",
        "TN": tn_k, "FP": fp_k, "FN": fn_k, "TP": tp_k,
        "Accuracy": accuracy_score(c_test, y_pred_knn),
        "Precision": precision_score(c_test, y_pred_knn),
        "Recall": recall_score(c_test, y_pred_knn),
        "F1": f1_score(c_test, y_pred_knn)
    },
    {
        "Model": "Decision Tree (depth=4)",
        "TN": tn_t, "FP": fp_t, "FN": fn_t, "TP": tp_t,
        "Accuracy": accuracy_score(c_test, y_pred_tree),
        "Precision": precision_score(c_test, y_pred_tree),
        "Recall": recall_score(c_test, y_pred_tree),
        "F1": f1_score(c_test, y_pred_tree)
    }
]

df_metrics = pd.DataFrame(metrics_data)
print("\n--- A7. Classification Metrics Table ---")
print(df_metrics.to_string(index=False))

df_metrics.to_csv("results/A_metrics.csv", index=False)
print("\nSaved metrics to results/A_metrics.csv")

proba_knn = knn_chosen.predict_proba(Xc_test)[:, 1]
thresholds = [0.3, 0.5, 0.7, 0.9]
thresh_results = []

for t in thresholds:
    y_pred_t = (proba_knn >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(c_test, y_pred_t).ravel()
    prec = precision_score(c_test, y_pred_t, zero_division=0)
    rec = recall_score(c_test, y_pred_t, zero_division=0)
    thresh_results.append({
        "Threshold": t,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Precision": prec,
        "Recall": rec
    })

df_thresh = pd.DataFrame(thresh_results)
print("\n--- A7. KNN Threshold Sweep ---")
print(df_thresh.to_string(index=False))

from sklearn.inspection import permutation_importance

scaled_lr = make_pipeline(StandardScaler(), LinearRegression())
scaled_lr.fit(X_train, y_train)
std_coefs = scaled_lr[-1].coef_
raw_coefs = lin_reg.coef_

tree_imp = tree_chosen.feature_importances_

perm_knn = permutation_importance(knn_chosen, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
perm_tree = permutation_importance(tree_chosen, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)

df_importance = pd.DataFrame({
    "Feature": X.columns,
    "Raw Coef": raw_coefs,
    "Std Coef": std_coefs,
    "Tree Imp": tree_imp,
    "Perm Imp (KNN)": perm_knn.importances_mean,
    "Perm Imp (Tree)": perm_tree.importances_mean
})

print("\n--- A8. Feature Importances Table ---")
print(df_importance.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
features = X.columns

axes[0].barh(features, std_coefs, color='steelblue')
axes[0].set_title("Standardised Coefficients (LinReg)")
axes[0].set_xlabel("Impact (MPa / std dev)")

axes[1].barh(features, tree_imp, color='forestgreen')
axes[1].set_title("Tree Feature Importances")
axes[1].set_xlabel("Entropy/Gini Reduction")

axes[2].barh(features, perm_knn.importances_mean, color='darkorange')
axes[2].set_title("Permutation Importance (KNN F1)")
axes[2].set_xlabel("Mean F1 Score Drop")

plt.tight_layout()
plt.savefig("plots/feature_importances.png")
print("Saved feature importances plot to plots/feature_importances.png")

X_phys1 = X.copy()
X_phys1["age"] = np.log(X_phys1["age"])

X_tr_p1, X_te_p1, y_tr_p1, y_te_p1 = train_test_split(
    X_phys1, y, test_size=0.2, random_state=42
)

lr_p1 = LinearRegression()
lr_p1.fit(X_tr_p1, y_tr_p1)
p1_test_r2 = r2_score(y_te_p1, lr_p1.predict(X_te_p1))
p1_test_rmse = np.sqrt(mean_squared_error(y_te_p1, lr_p1.predict(X_te_p1)))

X_phys2 = X_phys1.copy()
binder = X_phys2["cement"] + X_phys2["slag"] + X_phys2["ash"]
X_phys2["wb_ratio"] = X_phys2["water"] / binder
X_phys2 = X_phys2.drop(columns=["water"])

X_tr_p2, X_te_p2, y_tr_p2, y_te_p2 = train_test_split(
    X_phys2, y, test_size=0.2, random_state=42
)

lr_p2 = LinearRegression()
lr_p2.fit(X_tr_p2, y_tr_p2)
p2_test_r2 = r2_score(y_te_p2, lr_p2.predict(X_te_p2))
p2_test_rmse = np.sqrt(mean_squared_error(y_te_p2, lr_p2.predict(X_te_p2)))

print("\n--- Bonus: Physics-Informed Feature Engineering ---")
print(f"1. Log(Age) Transformation : Test R² = {p1_test_r2:.4f}, Test RMSE = {p1_test_rmse:.4f} MPa (Features p = {X_phys1.shape[1]})")
print(f"2. Log(Age) + w/b Ratio    : Test R² = {p2_test_r2:.4f}, Test RMSE = {p2_test_rmse:.4f} MPa (Features p = {X_phys2.shape[1]})")