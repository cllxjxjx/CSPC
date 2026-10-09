import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Task 1: Load data and isolate continuous columns
df = pd.read_csv("heart.csv")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

cols = ["age", "chol", "trestbps", "thalach"]
df_cont = df[cols]

# Task 2: Visualize distributions (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='teal')
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.savefig("distributions.png")
print("\ndistributions.png saved successfully.")

# Task 3: Statistical normality checks
print("\n--- Normality Test Results (Shapiro-Wilk) ---")
for col in cols:
    stat, p_val = stats.shapiro(df[col].dropna())
    skew = df[col].skew()
    is_normal = "Approximately Normal" if p_val > 0.05 else "Not Normal (Skewed/Outliers)"
    print(f"{col:10s} | p-value: {p_val:.5f} | Skewness: {skew:+.2f} | Verdict: {is_normal}")
