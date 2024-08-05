# main.py

from src.data_preprocessing import load_data, preprocess_data
from src.eda import plot_pairplot
from src.model_training import train_kmeans, train_kmeans_all_features
from src.evaluation import calculate_wcss, calculate_silhouette_scores, calculate_silhouette_scores_all_features
from src.visualizations import plot_clusters, plot_elbow, plot_silhouette


def main():
    # Load and preprocess data
    df = load_data('H:/My Drive/BISI II/Data Science/Term Assignments/Unsupervised_Clustering_Solution/data/mall_customers.csv')
    df = preprocess_data(df)
    
    # Perform EDA
    plot_pairplot(df)
    
      
    # Train KMeans Model
    kmodel, df = train_kmeans(df, n_clusters=5)
    
    # Visualize Clusters
    plot_clusters(df)
    
    # Evaluate using Elbow Method
    k_values = range(3, 9)
    WCSS = calculate_wcss(df, k_values)
    plot_elbow(k_values, WCSS)
    
    # Evaluate using Silhouette Score
    silhouette_scores = calculate_silhouette_scores(df, k_values)
    plot_silhouette(k_values, silhouette_scores)
    
    # Train and Evaluate using all features
    silhouette_scores_all_features = calculate_silhouette_scores_all_features(df, k_values)
    plot_silhouette(k_values, silhouette_scores_all_features)

if __name__ == "__main__":
    main()