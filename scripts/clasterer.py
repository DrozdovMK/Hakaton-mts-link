import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
import numpy as np
from embedding_model import *
#import matplotlib.pyplot as plt
#import pandas as pd


class Clasterer:
    def __init__(self, method='svd', n_components=2, max_clusters=10):
        """Инициализация SVD + выбор кластеризации (K-Means и DBSCAN)"""
        self.n_components = n_components
        self.max_clusters = max_clusters
        self.method = method

        if self.method == 'svd':
            self.reducer = TruncatedSVD(n_components=self.n_components)
        elif self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components)
        else:
            raise ValueError("Invalid method. Choose 'svd' or 'pca'.")

    def transform(self, embeddings):
        """Уменьшает размерность и применяет два метода кластеризации (K-Means и DBSCAN)"""
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Входные данные должны быть массивом numpy.")
        if embeddings.shape[1] < self.n_components:
            raise ValueError(f"Число компонент ({self.n_components}) не может быть больше числа признаков ({embeddings.shape[1]})")

        embeddings_reduced = self.reducer.fit_transform(embeddings)
        kmeans_labels, kmeans_silhouette = self.kmeans_clustering(embeddings_reduced)
        dbscan_labels, dbscan_silhouette = self.dbscan_clustering(embeddings_reduced)

        if kmeans_silhouette > dbscan_silhouette:
            print(f"Лучший метод: K-Means с силуэтным коэффициентом {kmeans_silhouette:.4f}")
            return embeddings_reduced, kmeans_labels
        else:
            print(f"Лучший метод: DBSCAN с силуэтным коэффициентом {dbscan_silhouette:.4f}")
            return embeddings_reduced, dbscan_labels

    def kmeans_clustering(self, embeddings):
        """Подбор гиперпараметров и кластеризация с помощью K-Means (начинаем с 3 кластеров)"""
        best_k = 3
        best_silhouette = -1
        best_labels = None

        for k in range(3, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            silhouette_avg = silhouette_score(embeddings, labels)

            print(f"K-Means (k={k}): силуэтный коэффициент = {silhouette_avg:.4f}")

            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_k = k
                best_labels = labels

        print(f"Лучший K для K-Means: {best_k}, силуэтный коэффициент: {best_silhouette:.4f}")
        return best_labels, best_silhouette

    def dbscan_clustering(self, embeddings):
        """Подбор гиперпараметров и кластеризация с помощью DBSCAN (минимум 3 кластера)"""
        best_eps = 0.1
        best_min_samples = 5
        best_silhouette = -1
        best_labels = None

        for eps in np.arange(0.1, 2.0, 0.1):
            for min_samples in range(3, 11):
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(embeddings)

                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # количество кластеров без шумовых точек (-1)

                if num_clusters >= 3:
                    silhouette_avg = silhouette_score(embeddings, labels)

                    print(f"DBSCAN (eps={eps:.1f}, min_samples={min_samples}): силуэтный коэффициент = {silhouette_avg:.4f}")

                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_eps = eps
                        best_min_samples = min_samples
                        best_labels = labels

        if best_labels is None:
            raise ValueError("Не удалось найти параметры DBSCAN, чтобы получить минимум 3 кластера.")

        print(f"Лучший eps для DBSCAN: {best_eps}, лучший min_samples: {best_min_samples}, силуэтный коэффициент: {best_silhouette:.4f}")
        return best_labels, best_silhouette

    #def plot_clusters(self, embeddings_reduced, labels):
    #    """Отображает кластеризованные данные на 2D-графике"""
    #    plt.figure(figsize=(10, 6))
    #    plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], c=labels, cmap='viridis', s=50, marker='o')
    #    plt.colorbar()
    #    plt.title('Clusters Visualization')
    #    plt.show()

# data = pd.read_csv("motivations.csv")
# phrases = data['Phrase'].tolist()
#
# embedder = RuBertEmbedder()
# embeddings = embedder.embed(phrases)
#
# clusterer = Clasterer(n_components=2, max_clusters=10)
# embeddings_reduced, cluster_labels = clusterer.transform(embeddings)
#
# print("Размерность после SVD:", embeddings_reduced.shape)
# print("Метки кластеров:", cluster_labels)
#
# clusterer.plot_clusters(embeddings_reduced, cluster_labels)