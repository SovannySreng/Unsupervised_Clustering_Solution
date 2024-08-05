
# src/model_training.py

from sklearn.cluster import KMeans

def train_kmeans(df, n_clusters):
    kmodel = KMeans(n_clusters=n_clusters).fit(df[['Annual_Income', 'Spending_Score']])
    df['Cluster'] = kmodel.labels_
    return kmodel, df

def train_kmeans_all_features(df, n_clusters):
    kmodel = KMeans(n_clusters=n_clusters).fit(df[['Age', 'Annual_Income', 'Spending_Score']])
    return kmodel

def get_cluster_centers(kmodel):
    return kmodel.cluster_centers_

def get_cluster_labels(kmodel):
    return kmodel.labels_