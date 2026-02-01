import pandas as pd

# Load the data
df = pd.read_csv("../data/credit_data.csv")

print("Original data:")
print(df.head())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing numeric values with column mean
df["income"].fillna(df["income"].mean(), inplace=True)
df["age"].fillna(df["age"].mean(), inplace=True)
df["loan_amount"].fillna(df["loan_amount"].mean(), inplace=True)
df["credit_utilization"].fillna(df["credit_utilization"].mean(), inplace=True)
df["late_payments"].fillna(df["late_payments"].mean(), inplace=True)
df["employment_years"].fillna(df["employment_years"].mean(), inplace=True)

# Fix credit utilization range (0 to 1)
df["credit_utilization"] = df["credit_utilization"].clip(0, 1)

# Fix negative values (not allowed)
df["income"] = df["income"].clip(lower=0)
df["loan_amount"] = df["loan_amount"].clip(lower=0)
df["employment_years"] = df["employment_years"].clip(lower=0)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned data preview:")
print(df.head())

# Save cleaned data
df.to_csv("../data/credit_data_cleaned.csv", index=False)

print("\nCleaned data saved as credit_data_cleaned.csv")
