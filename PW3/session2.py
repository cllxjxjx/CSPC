import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Task 4: Is the effect real? (Heart Disease)
df_heart = pd.read_csv("heart.csv")

# Split thalach into disease (target=1) and healthy (target=0)
disease_thalach = df_heart[df_heart['target'] == 1]['thalach'].dropna()
healthy_thalach = df_heart[df_heart['target'] == 0]['thalach'].dropna()

# Mann-Whitney U test (non-parametric since thalach was non-normal in Session 1)
stat, p_val = stats.mannwhitneyu(disease_thalach, healthy_thalach)

# Calculate means and Standard Error of the Mean (SEM)
m_dis, sem_dis = disease_thalach.mean(), disease_thalach.sem()
m_hea, sem_hea = healthy_thalach.mean(), healthy_thalach.sem()

print("--- Task 4: Heart Disease vs Max Heart Rate (thalach) ---")
print(f"Disease Group : Mean = {m_dis:.2f} ± {sem_dis:.2f} bpm")
print(f"Healthy Group : Mean = {m_hea:.2f} ± {sem_hea:.2f} bpm")
print(f"Mann-Whitney U statistic = {stat:.1f}, p-value = {p_val:.5e}")
print("Conclusion: " + ("Genuinely differ (p < 0.05)" if p_val < 0.05 else "No significant difference"))

# Plot means with uncertainty (SEM)
plt.figure(figsize=(6, 5))
sns.barplot(x='target', y='thalach', data=df_heart, capsize=0.1, errorbar='se', palette='Set2')
plt.xticks([0, 1], ['Healthy (0)', 'Disease (1)'])
plt.ylabel('Max Heart Rate (thalach)')
plt.title('Mean Max Heart Rate with Uncertainty (SEM)')
plt.tight_layout()
plt.savefig('thalach_target.png')


# Task 5: Do two things move together?

r_val, p_corr = stats.spearmanr(df_heart['age'], df_heart['thalach'])

print("\n--- Task 5: Correlation (Age vs Thalach) ---")
print(f"Spearman Correlation = {r_val:.3f}, p-value = {p_corr:.5e}")

plt.figure(figsize=(7, 5))
sns.regplot(x='age', y='thalach', data=df_heart, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
plt.title('Age vs Maximum Heart Rate')
plt.xlabel('Age (years)')
plt.ylabel('Max Heart Rate (thalach)')
plt.tight_layout()
plt.savefig('age_thalach.png')


# Task 6 & 7: Chemical-Exposure Mystery
df_chem = pd.read_csv("chemicals_cancer.csv")

print("\n--- Task 6: Naive Correlation Analysis ---")
corr_benzene = df_chem['benzene'].corr(df_chem['malignancy'])
corr_cadmium = df_chem['cadmium'].corr(df_chem['malignancy'])
print(f"Benzene vs Malignancy correlation : {corr_benzene:.3f}")
print(f"Cadmium vs Malignancy correlation : {corr_cadmium:.3f}")

print("\n--- Task 7: Controlling for Pollution Index (40 < pollution_index < 60) ---")
df_restricted = df_chem[(df_chem['pollution_index'] > 40) & (df_chem['pollution_index'] < 60)]
corr_ben_rest = df_restricted['benzene'].corr(df_restricted['malignancy'])
corr_cad_rest = df_restricted['cadmium'].corr(df_restricted['malignancy'])
print(f"Restricted Benzene vs Malignancy correlation : {corr_ben_rest:.3f}")
print(f"Restricted Cadmium vs Malignancy correlation : {corr_cad_rest:.3f}")


# Bonus: Categorical Unpredictability (Entropy)
props = df_heart['target'].value_counts(normalize=True)
entropy = -np.sum(props * np.log2(props)) # Shannon entropy in bits

print("\n--- Bonus: Target Category Unpredictability ---")
print("Class Proportions:\n", props.to_string())
print(f"Shannon Entropy = {entropy:.3f} bits (Max = 1.0 bit for perfectly balanced 50/50)")
