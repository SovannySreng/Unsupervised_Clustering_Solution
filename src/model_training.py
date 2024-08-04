
import pandas as pd
from sklearn.cluster import KMeans

def train_model(df: pd.DataFrame, n_clusters: int):
    model = KMeans(n_clusters=n_clusters)
    model.fit(df)
    return model