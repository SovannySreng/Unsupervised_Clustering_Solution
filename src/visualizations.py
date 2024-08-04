
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(df: pd.DataFrame, model):
    labels = model.labels_
    df['Cluster'] = labels
    sns.pairplot(df, hue='Cluster', palette='viridis')
    plt.show()

def plot_histograms(df: pd.DataFrame, num_cols: list):
    df[num_cols].hist(figsize=(14, 14))
    plt.show()

def plot_categorical_distribution(df: pd.DataFrame, cat_cols: list):
    for col in cat_cols:
        if col != 'Cluster':  # Assuming 'Cluster' is the name of the cluster column
            pd.crosstab(df[col], df['Cluster'], normalize='index').plot(kind='bar', figsize=(8, 4), stacked=True)
            plt.ylabel('Cluster Percentage')
            plt.title(f'Distribution of {col} by Cluster')
            plt.show()