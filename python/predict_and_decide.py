import pandas as pd
from train_model import train

def loan_decision(pd_value):
    if pd_value < 0.3:
        return "APPROVE"
    elif pd_value < 0.6:
        return "REVIEW"
    else:
        return "REJECT"

model, _, _ = train()

new_customer = pd.DataFrame([{
    "income": 45000,
    "age": 28,
    "loan_amount": 120000,
    "credit_utilization": 0.65,
    "late_payments": 1,
    "employment_years": 3
}])

pd_value = model.predict_proba(new_customer)[0][1]

print("PD:", round(pd_value, 2))
print("Decision:", loan_decision(pd_value))
