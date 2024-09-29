import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
#import matplotlib.pyplot as plt


class Clasterer:
    def __init__(self, method='svd', n_components=50, threshold=0.5, max_clusters=10):
        """Инициализация SVD + Birch.
               Параметры:
               n_components : int, опционально (по умолчанию 50)
                   Количество компонент для SVD.
               threshold : float, опционально (по умолчанию 0.5)
                   Порог чувствительности для Birch.
                max_clusters : int, опционально (по умолчанию 10)
               Максимальное количество кластеров для оценки, если не указано n_clusters.
               """
        self.n_components = n_components
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.method = method

        if self.method == 'svd':
            self.reducer = TruncatedSVD(n_components=self.n_components)
        elif self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components)
        else:
            raise ValueError("Invalid method. Choose 'svd' or 'pca'.")

    def transform(self, embeddings):
        """ Обучает кластеризатор на данных
            Параметры:
               embeddings : np.ndarray
                   Входной массив эмбеддингов (размерности N x M).

            Возвращает:
               embeddings_svd : np.ndarray
                   Массив эмбеддингов после применения SVD (размерности N x n_components).
               labels : np.ndarray
                   Массив меток кластеризации, соответствующих каждому эмбеддингу.
               """

        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Входные данные должны быть массивом numpy.")
        if embeddings.shape[1] < self.n_components:
            raise ValueError(f"Число компонент SVD ({self.n_components}) не может быть больше числа признаков ({embeddings.shape[1]})")

        embeddings_reduced = self.reducer.fit_transform(embeddings)

        #силуэтный коэффициент
        best_n_clusters = 2
        best_silhouette_score = -1
        best_labels = None

        for n_clusters in range(2, self.max_clusters + 1):
            birch_model = Birch(threshold=self.threshold, n_clusters=n_clusters)
            labels = birch_model.fit_predict(embeddings_reduced)

            if len(np.unique(labels)) > 1:  # избегаем случаев, когда все данные в одном кластере
                silhouette_avg = silhouette_score(embeddings_reduced, labels)
                print(f'n_clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}')

                if silhouette_avg > best_silhouette_score:
                    best_silhouette_score = silhouette_avg
                    best_n_clusters = n_clusters
                    best_labels = labels

        print(f'Best n_clusters: {best_n_clusters}, Best Silhouette Score: {best_silhouette_score:.4f}')
        self.birch = Birch(threshold=self.threshold, n_clusters=best_n_clusters)
        final_labels = self.birch.fit_predict(embeddings_reduced)

        return embeddings_reduced, final_labels


    def predict(self, embeddings):
        """Понижает размерность и возвращает метки кластеров.

            Параметры:
               embeddings : np.ndarray
                   Входной массив эмбеддингов (размерности N x M).

            Возвращает:
               labels : np.ndarray
                   Массив меток кластеризации.
               """
        _, labels = self.transform(embeddings)
        return labels

    #def plot_clusters(self, embeddings_svd, labels):
    #    tsne = TSNE(n_components=2, random_state=42)
    #    embeddings_2d = tsne.fit_transform(embeddings_svd)

    #    plt.figure(figsize=(10, 6))
    #    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', s=50, marker='o')
    #    plt.colorbar()
    #    plt.title('t-SNE Clusters Visualization')
    #    plt.show()


# embeddings = np.random.rand(100, 300)
# clusterer = Clasterer(n_components=50, threshold=0.7, max_clusters=10)
# embeddings_svd, cluster_labels = clusterer.transform(embeddings)
# print("Размерность после SVD:", embeddings_svd.shape)
# print("Метки кластеров:", cluster_labels)