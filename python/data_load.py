import pandas as pd

def load_data():
    df = pd.read_csv("../data/credit_data.csv")
    X = df.drop("default", axis=1)
    y = df["default"]
    return X, y

