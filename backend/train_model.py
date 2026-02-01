import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------------------
# 1. CREATE DUMMY TRAINING DATA
# ---------------------------
np.random.seed(42)

rows = 1000

data = pd.DataFrame({
    "income": np.random.randint(20000, 150000, rows),
    "age": np.random.randint(21, 65, rows),
    "loan_amount": np.random.randint(5000, 500000, rows),
    "employment_years": np.random.randint(0, 40, rows),
    "late_payments": np.random.randint(0, 10, rows),
    "credit_utilization": np.random.uniform(0.1, 1.0, rows),
})

# Target: default (0 = no, 1 = yes)
data["default"] = (
    (data["late_payments"] > 3) |
    (data["credit_utilization"] > 0.7) |
    (data["loan_amount"] > data["income"] * 3)
).astype(int)

# ---------------------------
# 2. SPLIT DATA
# ---------------------------
X = data.drop("default", axis=1)
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 3. TRAIN MODEL
# ---------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# 4. SAVE MODEL
# ---------------------------
joblib.dump(model, "credit_risk_model.pkl")

print("✅ Model trained and saved as credit_risk_model.pkl")