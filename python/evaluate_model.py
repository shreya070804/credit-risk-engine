from sklearn.metrics import confusion_matrix, roc_auc_score
from train_model import train

def evaluate():
    model, X_test, y_test = train()
    pd_values = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, model.predict(X_test)))

    print("ROC-AUC Score:", roc_auc_score(y_test, pd_values))
