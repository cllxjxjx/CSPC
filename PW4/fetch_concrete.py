import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

concrete = fetch_ucirepo(id=165)

X = concrete.data.features
y = concrete.data.targets

print("--- Metadata ---")
print(concrete.metadata)

print("\n--- Variables Table ---")
print(concrete.variables)

X = X.copy()
X.columns = [
    "cement", "slag", "ash", "water", 
    "superplastic", "coarse_agg", "fine_agg", "age"
]
y = y.iloc[:, 0].rename("strength")

print("\n--- Renamed Features (X) ---")
print(X.head())

print("\n--- Renamed Target (y) ---")
print(y.head())

os.makedirs("data", exist_ok=True)
df_out = X.assign(strength=y)
df_out.to_csv("data/concrete.csv", index=False)
print("\nSuccessfully saved dataset to data/concrete.csv")