
# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_clusters(df):
    sns.scatterplot(x='Annual_Income', y='Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.show()

def plot_elbow(k_values, WCSS):
    wss = pd.DataFrame({'cluster': k_values, 'WSS_Score': WCSS})
    wss.plot(x='cluster', y='WSS_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.show()

def plot_silhouette(k_values, silhouette_scores):
    wss = pd.DataFrame({'cluster': k_values, 'Silhouette_Score': silhouette_scores})
    wss.plot(x='cluster', y='Silhouette_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Plot')
    plt.show()