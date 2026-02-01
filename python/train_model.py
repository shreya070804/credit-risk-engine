from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from data_load import load_data

def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test
