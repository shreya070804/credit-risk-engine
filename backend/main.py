from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
import os

# ================= APP =================
app = FastAPI(title="Credit Risk Decision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_risk_model_v1.pkl")
LOG_FILE = os.path.join(BASE_DIR, "audit_log.csv")

model = joblib.load(MODEL_PATH)

# ================= MODELS =================
class Applicant(BaseModel):
    income: float
    age: int
    loan_amount: float
    employment_years: int
    late_payments: int
    credit_utilization: float
    risk_mode: str

# ================= EXPLANATION LOGIC =================
def generate_explanation(data):
    reasons = []

    if data.credit_utilization > 0.6:
        reasons.append("High credit utilization increased risk")

    if data.late_payments > 2:
        reasons.append("Multiple late payments detected")

    if data.loan_amount > data.income * 0.6:
        reasons.append("Loan amount is high compared to income")

    if data.employment_years < 2:
        reasons.append("Short employment history increased risk")

    if not reasons:
        reasons.append("All major risk factors are within acceptable range")

    return reasons

# ================= ROOT =================
@app.get("/")
def root():
    return {"status": "API running"}

# ================= PREDICT =================
@app.post("/predict")
def predict(payload: Applicant):
    X = pd.DataFrame([[ 
        payload.income,
        payload.age,
        payload.loan_amount,
        payload.credit_utilization,
        payload.late_payments,
        payload.employment_years
    ]], columns=[
        "income",
        "age",
        "loan_amount",
        "credit_utilization",
        "late_payments",
        "employment_years"
    ])

    pd_value = float(model.predict_proba(X)[0][1])

    if payload.risk_mode == "Conservative":
        approve, review = 0.25, 0.45
    elif payload.risk_mode == "Aggressive":
        approve, review = 0.40, 0.70
    else:
        approve, review = 0.30, 0.60

    if pd_value < approve:
        decision = "APPROVE"
    elif pd_value < review:
        decision = "REVIEW"
    else:
        decision = "REJECT"

    log_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pd": round(pd_value, 4),
        "decision": decision,
        "mode": payload.risk_mode
    }

    pd.DataFrame([log_row]).to_csv(
        LOG_FILE,
        mode="a",
        index=False,
        header=not os.path.exists(LOG_FILE)
    )

    return {
        "probability_of_default": round(pd_value, 4),
        "decision": decision,
        "risk_strategy": payload.risk_mode,
        "explanation": generate_explanation(payload)
    }

# ================= ADMIN AUDIT =================
@app.get("/admin/audit")
def audit(role: str | None = Header(default=None)):
    if role != "ADMIN":
        raise HTTPException(status_code=403, detail="Admin access required")

    if not os.path.exists(LOG_FILE):
        return []

    return pd.read_csv(LOG_FILE).to_dict(orient="records")