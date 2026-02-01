import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load scaled data
df = pd.read_csv("../data/credit_data_scaled.csv")

X = df.drop("default", axis=1)
y = df["default"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "credit_risk_model.pkl")

print("Model trained and saved as credit_risk_model.pkl")

# Load saved model
loaded_model = joblib.load("credit_risk_model.pkl")

# New customer
new_customer = pd.DataFrame([{
    "income": 50000,
    "age": 30,
    "loan_amount": 150000,
    "credit_utilization": 0.5,
    "late_payments": 1,
    "employment_years": 4
}])

# Predict PD
pd_value = loaded_model.predict_proba(new_customer)[0][1]

# Decision logic
def loan_decision(pd):
    if pd < 0.3:
        return "APPROVE"
    elif pd < 0.6:
        return "REVIEW"
    else:
        return "REJECT"

print("PD:", round(pd_value, 2))
print("Decision:", loan_decision(pd_value))
