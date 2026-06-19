import pandas as pd

def carregar_dados():
    return pd.read_csv('./data/raw/pima_diabetes.csv', sep=',')