


import pandas as pd

def load_data(file_path='H:/My Drive/BISI II/Data Science/Term Assignments/Unsupervised_Clustering_Solution/data/mall_customers.csv') -> pd.DataFrame:
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Implement data preprocessing steps here
    return df