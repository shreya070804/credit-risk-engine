# 🏦 Credit Risk Engine – End-to-End Fintech Project

An end-to-end *AI-powered Credit Risk Decision System* that simulates how *banks and fintech companies* evaluate loan applications, make approval decisions, and monitor risk using dashboards.

This project combines *Machine Learning, Backend APIs, Frontend UI, and Business Analytics (Power BI)*.

---

## 🚀 Key Features

### 👤 Applicant Portal
- Modern, centered credit application form
- Numeric-only validated inputs
- Real-time *Probability of Default (PD)* prediction
- Clear decision: *APPROVE / REVIEW / REJECT*
- Explainable AI (decision reasons shown to user)

### 🛠 Admin Dashboard
- Admin-only access
- Complete audit log of all applications
- KPI metrics:
  - Total Applications
  - Approved Applications
  - Rejected Applications
- Real-time monitoring of credit decisions

### 🤖 Machine Learning
- Trained ML classification model
- Uses applicant financial & behavioral features
- Outputs probability of default (PD)
- Threshold-based decision logic (bank-style)

### 📊 Business Analytics (Power BI)
- Interactive dashboards
- Approval vs Rejection trends
- Risk distribution analysis
- Recruiter & business friendly insights

---

## 🧠 Tech Stack

*Frontend*
- React (Single Page Application)
- Fetch API for backend communication
- Clean, modern UI

*Backend*
- FastAPI
- Pydantic data validation
- CORS enabled
- CSV-based audit logging

*Machine Learning*
- Scikit-learn
- Joblib for model persistence

*Visualization*
- Power BI Desktop

---

## 📂 Project Structure

credit-risk-engine/
│
├── backend/
│   ├── main.py                 # FastAPI backend API
│   ├── train_model.py          # ML model training script
│   ├── credit_risk_model.pkl   # Trained ML model
│   └── audit_log.csv           # Auto-generated audit log
│
├── frontend/
│   └── src/
│       ├── App.js              # Main React application
│       ├── Landing.js          # Landing page
│       ├── Applicant.js        # Applicant dashboard
│       └── AdminDashboard.js   # Admin dashboard
│
├── .gitignore
└── README.md

