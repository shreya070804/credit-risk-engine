import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime
import os
import uuid

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Risk Decision Engine",
    layout="wide"
)

# ---------------- LOAD CONFIG ----------------
with open("config.json", "r") as f:
    CONFIG = json.load(f)

MODEL_VERSION = CONFIG["model_version"]
model = joblib.load(f"credit_risk_model_{MODEL_VERSION}.pkl")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### User Context")
    user_role = st.selectbox("Role", ["Analyst", "Manager", "Admin"])

    st.divider()

    if user_role == "Admin":
        risk_mode = st.selectbox(
            "Risk Strategy",
            list(CONFIG["risk_strategies"].keys())
        )
    else:
        risk_mode = "Balanced"
        st.caption("Risk Strategy: Balanced")

# ---------------- HEADER ----------------
st.markdown("## Credit Risk & Loan Decision Engine")
st.caption(f"Model Version: {MODEL_VERSION}")
st.divider()

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.1, 1])

# ================= LEFT: INPUT =================
with left:
    st.markdown("### Applicant Details")

    income = st.number_input("Income", min_value=0)
    age = st.number_input("Age", min_value=18)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    employment_years = st.number_input("Employment Years", min_value=0)
    credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.5)
    late_payments = st.number_input("Late Payments", min_value=0)

    run = st.button("Run Credit Check")

# ================= RIGHT: OUTPUT =================
with right:
    st.markdown("### Decision Output")

    if run:
        df = pd.DataFrame([{
            "income": income,
            "age": age,
            "loan_amount": loan_amount,
            "credit_utilization": credit_utilization,
            "late_payments": late_payments,
            "employment_years": employment_years
        }])

        pd_value = model.predict_proba(df)[0][1]

        approve_th = CONFIG["risk_strategies"][risk_mode]["approve_threshold"]
        review_th = CONFIG["risk_strategies"][risk_mode]["review_threshold"]

        if pd_value < approve_th:
            decision = "APPROVE"
        elif pd_value < review_th:
            decision = "REVIEW"
        else:
            decision = "REJECT"

        st.metric("Probability of Default (PD)", f"{pd_value:.2f}")
        st.metric("Decision", decision)
        st.metric("Risk Strategy", risk_mode)

        if decision == "APPROVE":
            st.success("Low Risk")
        elif decision == "REVIEW":
            st.warning("Medium Risk – Manual Review")
        else:
            st.error("High Risk")

        # ---- SAVE HISTORY ----
        log = {
            "decision_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pd": round(float(pd_value), 4),
            "final_decision": decision
        }

        pd.DataFrame([log]).to_csv(
            "loan_history.csv",
            mode="a",
            index=False,
            header=not os.path.exists("loan_history.csv")
        )

# ================= PORTFOLIO VIEW =================
if user_role in ["Manager", "Admin"] and os.path.exists("loan_history.csv"):
    st.divider()
    st.markdown("### Portfolio Overview")

    hist = pd.read_csv("loan_history.csv")
    hist["pd"] = pd.to_numeric(hist["pd"], errors="coerce")
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist = hist.dropna(subset=["timestamp"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Approval Rate", (hist["final_decision"] == "APPROVE").mean())
    c2.metric("Rejection Rate", (hist["final_decision"] == "REJECT").mean())
    c3.metric("Average PD", hist["pd"].mean())

    st.divider()

    colA, colB = st.columns(2)

    with colA:
        st.caption("Decision Mix")
        st.bar_chart(hist["final_decision"].value_counts())

    with colB:
        st.caption("PD Distribution")
        st.bar_chart(hist["pd"].dropna())

    st.divider()
    st.caption("PD Trend")
    st.line_chart(
        hist.sort_values("timestamp")[["timestamp", "pd"]].set_index("timestamp")
    )

# ================= ADMIN GOVERNANCE =================
if user_role == "Admin":
    st.divider()
    st.markdown("### Admin Governance")
    st.json(CONFIG)
