
# src/evaluation.py

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
def calculate_wcss(df, k_values):
    WCSS = []
    for k in k_values:
        kmodel = KMeans(n_clusters=k).fit(df[['Annual_Income', 'Spending_Score']])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
    return WCSS

def calculate_silhouette_scores(df, k_values):
    silhouette_scores = []
    for k in k_values:
        kmodel = KMeans(n_clusters=k).fit(df[['Annual_Income', 'Spending_Score']])
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[['Annual_Income', 'Spending_Score']], ypred)
        silhouette_scores.append(sil_score)
    return silhouette_scores

def calculate_silhouette_scores_all_features(df, k_values):
    silhouette_scores = []
    for k in k_values:
        kmodel = KMeans(n_clusters=k).fit(df[['Age', 'Annual_Income', 'Spending_Score']])
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[['Age', 'Annual_Income', 'Spending_Score']], ypred)
        silhouette_scores.append(sil_score)
    return silhouette_scores