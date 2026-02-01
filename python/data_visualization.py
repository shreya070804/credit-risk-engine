import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# Load scaled data
df = pd.read_csv("../data/credit_data_scaled.csv")

df["default"].value_counts().plot(kind="bar")
plt.title("Loan Repayment Status")
plt.xlabel("0 = Paid, 1 = Defaulted")
plt.ylabel("Number of People")
plt.show()

df["income"].plot(kind="hist", bins=10)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.show()

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pd_values = model.predict_proba(X_test)[:, 1]

plt.hist(pd_values, bins=10)
plt.title("Probability of Default (PD) Distribution")
plt.xlabel("PD value")
plt.ylabel("Count")
plt.show()

fpr, tpr, _ = roc_curve(y_test, pd_values)
auc_score = roc_auc_score(y_test, pd_values)

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
