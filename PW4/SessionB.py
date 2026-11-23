import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

wine = load_wine(as_frame=True)
X = wine.data
cultivar = wine.target

print("--- Wine Dataset Description ---")
print(wine.DESCR)

print("\n--- Summary Statistics (X.describe()) ---")
desc = X.describe().T
desc['range'] = desc['max'] - desc['min']
print(desc[['mean', 'std', 'min', 'max', 'range']].round(2))

pca_raw = PCA()
pca_raw.fit(X)

exp_var_pc1 = pca_raw.explained_variance_ratio_[0]
loadings_pc1 = pd.Series(pca_raw.components_[0], index=X.columns).sort_values(ascending=False)

print("\n--- B2. PCA on Raw Data ---")
print(f"PC1 Explained Variance Ratio: {exp_var_pc1:.6f} ({exp_var_pc1*100:.2f}%)")
print("\nSorted PC1 Loadings:")
print(loadings_pc1)

scaler = StandardScaler()
Z = scaler.fit_transform(X)

pca = PCA()
pca.fit(Z)

exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

df_pca_summary = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(exp_var))],
    "Explained Variance Ratio": exp_var,
    "Cumulative Variance": cum_var
})

print("\n--- B3. Scaled PCA Summary ---")
print(df_pca_summary.to_string(index=False))

os.makedirs("plots", exist_ok=True)
fig, ax1 = plt.subplots(figsize=(8, 5))

components = np.arange(1, len(exp_var) + 1)
ax1.bar(components, exp_var, color="skyblue", alpha=0.7, label="Individual Variance")
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance Ratio", color="blue")
ax1.set_xticks(components)

ax2 = ax1.twinx()
ax2.plot(components, cum_var, color="red", marker="o", linewidth=2, label="Cumulative Variance")
ax2.axhline(0.90, color="gray", linestyle="--", label="90% Threshold")
ax2.set_ylabel("Cumulative Explained Variance", color="red")

plt.title("Scree Plot of Scaled Wine PCA")
plt.tight_layout()
plt.savefig("plots/wine_scree_plot.png")

loadings = pd.DataFrame(
    pca.components_[:2].T,
    columns=["PC1", "PC2"],
    index=X.columns
)

print("\n--- B3. PC1 Loadings (Sorted) ---")
print(loadings["PC1"].sort_values(ascending=False).to_string())

print("\n--- B3. PC2 Loadings (Sorted) ---")
print(loadings["PC2"].sort_values(ascending=False).to_string())

Z_pca = pca.transform(Z)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(Z_pca[:, 0], Z_pca[:, 1], color="darkred", alpha=0.7, edgecolors="k")
ax.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}%)")
ax.set_title("Wines on PC1-PC2 Plane (Single Color)")
ax.grid(True)
plt.tight_layout()
plt.savefig("plots/wine_pc1_pc2_unlabeled.png")

print("\n--- B4 (a). Inertia across Random Seeds (k=3, n_init=1) ---")
inertias_single = []
for rs in range(8):
    km_single = KMeans(n_clusters=3, init="k-means++", n_init=1, max_iter=300, random_state=rs)
    km_single.fit(Z)
    inertias_single.append(km_single.inertia_)
    print(f"random_state = {rs}: inertia = {km_single.inertia_:.4f}")

km_10 = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0)
km_10.fit(Z)
print(f"n_init=10 Best Inertia: {km_10.inertia_:.4f}")

print("\n--- B4 (b). Elbow Method (k=1..10, n_init=10) ---")
k_range = list(range(1, 11))
inertias_elbow = []

for k in k_range:
    km_elbow = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=0)
    km_elbow.fit(Z)
    inertias_elbow.append(km_elbow.inertia_)
    print(f"k = {k:2d}: inertia = {km_elbow.inertia_:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_range, inertias_elbow, marker='o', color='crimson', linewidth=2)
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax.set_title("K-Means Elbow Curve for Wine Dataset")
ax.set_xticks(k_range)
ax.grid(True)
plt.tight_layout()
plt.savefig("plots/kmeans_elbow.png")

counts = np.bincount(km_10.labels_)
centers_real = pd.DataFrame(
    scaler.inverse_transform(km_10.cluster_centers_),
    columns=X.columns
)

print("\n--- B5. K-Means Cluster Sizes (k=3) ---")
for i, count in enumerate(counts):
    print(f"Cluster {i}: {count} wines")

print("\n--- B5. Cluster Centers in Real Units ---")
print(centers_real.round(2).to_string())

fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(Z_pca[:, 0], Z_pca[:, 1], c=km_10.labels_, cmap="Set1", alpha=0.8, edgecolors="k")
ax.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}%)")
ax.set_title("Wines on PC1-PC2 Plane (Colored by K-Means Cluster)")
ax.grid(True)
plt.tight_layout()
plt.savefig("plots/wine_pc1_pc2_kmeans.png")

os.makedirs("results", exist_ok=True)

km_scaled = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0)
km_scaled.fit(Z)
ct_scaled = pd.crosstab(cultivar, km_scaled.labels_)
acc_scaled = ct_scaled.max(axis=0).sum() / len(cultivar)

km_raw = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0)
km_raw.fit(X)
ct_raw = pd.crosstab(cultivar, km_raw.labels_)
acc_raw = ct_raw.max(axis=0).sum() / len(cultivar)

km_pca2 = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0)
km_pca2.fit(Z_pca[:, :2])
ct_pca2 = pd.crosstab(cultivar, km_pca2.labels_)
acc_pca2 = ct_pca2.max(axis=0).sum() / len(cultivar)

df_agreement = pd.DataFrame([
    {"Version": "(i) Scaled Data (13D)", "Agreement": acc_scaled},
    {"Version": "(ii) Raw Data (13D)", "Agreement": acc_raw},
    {"Version": "(iii) PC1-PC2 Plane (2D)", "Agreement": acc_pca2}
])

print("\n--- B6. Crosstab: Scaled Data (13D) ---")
print(ct_scaled)

print("\n--- B6. Crosstab: Raw Data (13D) ---")
print(ct_raw)

print("\n--- B6. Crosstab: PC1-PC2 Plane (2D) ---")
print(ct_pca2)

print("\n--- B6. Agreement Table ---")
print(df_agreement.to_string(index=False))

df_agreement.to_csv("results/B_agreement.csv", index=False)
print("\nSaved agreement table to results/B_agreement.csv")

X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(
    X, cultivar, test_size=0.25, random_state=42, stratify=cultivar
)

pipe_knn_full = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
pipe_knn_full.fit(X_w_train, y_w_train)
acc_knn_full = pipe_knn_full.score(X_w_test, y_w_test)

pipe_knn_pca2 = make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(n_neighbors=5))
pipe_knn_pca2.fit(X_w_train, y_w_train)
acc_knn_pca2 = pipe_knn_pca2.score(X_w_test, y_w_test)

print("\n--- B8 Bonus: Supervised KNN Classification ---")
print(f"Supervised KNN (13D Full Scaled Data) Test Accuracy : {acc_knn_full:.4f} ({acc_knn_full*100:.2f}%)")
print(f"Supervised KNN (2D PCA Scaled Data) Test Accuracy   : {acc_knn_pca2:.4f} ({acc_knn_pca2*100:.2f}%)")
print(f"Unsupervised K-Means Agreement (from B6)           : {acc_scaled:.4f} ({acc_scaled*100:.2f}%)")