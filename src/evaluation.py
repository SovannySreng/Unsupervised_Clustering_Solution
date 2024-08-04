
import pandas as pd
from sklearn.metrics import silhouette_score

def evaluate_model(model, df: pd.DataFrame):
    labels = model.labels_
    score = silhouette_score(df, labels)
    print("Silhouette Score:", score)