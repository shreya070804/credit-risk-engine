import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load scaled data
df = pd.read_csv("../data/credit_data_scaled.csv")

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pd_values = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, pd_values)
gini = 2 * auc - 1

print("AUC:", round(auc, 3))
print("Gini:", round(gini, 3))

def ks_statistic(y_true, pd_scores):
    data = pd.DataFrame({
        "y": y_true,
        "pd": pd_scores
    }).sort_values("pd")

    data["cum_good"] = np.cumsum(1 - data["y"]) / (1 - data["y"]).sum()
    data["cum_bad"] = np.cumsum(data["y"]) / data["y"].sum()

    ks = max(abs(data["cum_good"] - data["cum_bad"]))
    return ks

ks = ks_statistic(y_test.values, pd_values)
print("KS Statistic:", round(ks, 3))

def psi(expected, actual, buckets=10):
    expected_percents = np.histogram(expected, buckets)[0] / len(expected)
    actual_percents = np.histogram(actual, buckets)[0] / len(actual)

    psi_value = np.sum(
        (expected_percents - actual_percents) *
        np.log((expected_percents + 1e-6) / (actual_percents + 1e-6))
    )
    return psi_value

# Compare train PD vs test PD
pd_train = model.predict_proba(X_train)[:, 1]
psi_value = psi(pd_train, pd_values)

print("PSI:", round(psi_value, 3))
