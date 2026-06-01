import pandas as pd

def carregar_dados():
    return pd.read_csv('../datasets/pima_diabetes.csv', sep=',')