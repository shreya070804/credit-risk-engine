import pandas as pd

# Load cleaned data
df = pd.read_csv("../data/credit_data_cleaned.csv")

print("Cleaned data preview:")
print(df.head())

# Separate input features and target
X = df.drop("default", axis=1)
y = df["default"]

print("\nInput features before scaling:")
print(X.head())

from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Scale the input features
X_scaled = scaler.fit_transform(X)

print("\nScaled input (first 5 rows):")
print(X_scaled[:5])

# Convert scaled data back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled DataFrame preview:")
print(X_scaled_df.head())

# Add target back
df_scaled = X_scaled_df.copy()
df_scaled["default"] = y.values

# Save scaled data
df_scaled.to_csv("../data/credit_data_scaled.csv", index=False)

print("\nScaled data saved as credit_data_scaled.csv")
