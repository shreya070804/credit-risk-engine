import pandas as pd

# Load scaled data
df = pd.read_csv("../data/credit_data_scaled.csv")

print("Data preview:")
print(df.head())
print("\nMissing values check:")
print(df.isnull().sum())
print("\nData shape (rows, columns):")
print(df.shape)
X = df.drop("default", axis=1)
y = df["default"]

print("\nInput shape:", X.shape)
print("Target shape:", y.shape)

