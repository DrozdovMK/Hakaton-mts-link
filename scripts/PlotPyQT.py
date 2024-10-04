import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QListWidget,
    QGraphicsEllipseItem,
    QGraphicsScene
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scripts.summarizer import gpt_summarizer
import pyqtgraph as pg
from PyQt5.QtGui import QTransform
from PyQt5.QtCore import pyqtSignal, QThread, QLineF
from sklearn.decomposition import PCA
from scripts.profanity_check import profanity_processing

class ProgressThread(QThread):
    result_ready = pyqtSignal(list)
    def __init__(self, mainwindow, parent=None):
        super().__init__()
        self.mainwindow = mainwindow
    def set_attrs(self, offline, count_offline_words, max_gpt_responses, answers, embeddings, clusters):
        self.offline = offline
        self.count_offline_words = count_offline_words
        self.max_gpt_responses = max_gpt_responses
        self.answers = answers
        self.embeddings = embeddings
        self.clusters = clusters
    def run(self):
        self.mainwindow.create_data(self.offline,self.count_offline_words,self.max_gpt_responses)
        results = [
            self.mainwindow.embeddings_clustered,
            self.mainwindow.answers_clustered,
            self.mainwindow.cluster_summaries]
        self.result_ready.emit(results)


def ellipse_settings(points):
    centroid = np.mean(points, axis=0)
    pca = PCA(n_components=2)
    pca.fit(points)
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_
    x_radius, y_radius = np.sqrt(eigenvalues) * 2
    angle = np.degrees(np.arctan2(*eigenvectors[0][::-1]))
    return centroid, x_radius, y_radius, angle
    

class CustomEllipseItem(QGraphicsEllipseItem):
    def __init__(self, centriod, width, height, angle, label):
        super().__init__(-width, -height, width*2, height*2)
        self.setRotation(angle)
        self.setPos(centriod[0], centriod[1])
        self.setPen(pg.mkPen(color=(0, 0, 200), width=2))
        self.setBrush(pg.mkBrush(100, 100, 250, 80))
        self.label = label

class BarChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = ProgressThread(mainwindow=self)
        self.censor = profanity_processing()
        self.listWidget = QListWidget()
        mainLayout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        mainLayout.addWidget(self.plot_widget, 1)
        self.setLayout(mainLayout)
        mainLayout.addWidget(self.listWidget, 1)
        
    def apply(self, embeddings, answers, clusters, offline, count_offline_words, max_gpt_responses ):
        self.offline = offline
        self.count_offline_words = count_offline_words
        self.max_gpt_responses = max_gpt_responses
        self.answers = answers
        self.embeddings = embeddings
        self.clusters = clusters
        self.thread.result_ready.connect(self.on_result_ready) 
        # Выносим долгую часть в отдельный поток
        self.thread.set_attrs(offline, count_offline_words, max_gpt_responses, answers, embeddings, clusters)
        self.thread.start()


    def draw_clusters(self):
        """
        Отрисовывает кластеры
        """
        # Очищаем график перед перерисовкой
        self.plot_widget.clear()
        # Сохраняем объекты кругов для обработки событий
        for i, embed_clustered in enumerate(self.embeddings_clustered):
            centroid, width, height, angle = ellipse_settings(embed_clustered)
            scatter = pg.ScatterPlotItem(x = embed_clustered[:, 0],
                                         y = embed_clustered[:, 1],
                                         pen=pg.mkPen(width=1, color='r'), symbol='o', size=1)

            self.plot_widget.addItem(scatter)
            ellipse = CustomEllipseItem(centroid, width, height, angle, label = i)
            ellipse.setData(0, i)
            self.plot_widget.addItem(ellipse)
            # Добавляем название кластера в  центр круга
            text = pg.TextItem("Кластер {}".format(i), anchor=(0.5, 0.5), color=(255, 255, 255))
            text.setPos(centroid[0], centroid[1])
            text.setData(0, i)
            self.plot_widget.addItem(text)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_click)
    
    def on_click(self, event):
        pos = event.scenePos()
        items = self.plot_widget.scene().items(pos)

        # Сортируем элементы по расстоянию до точки pos
        
        items.sort(key=lambda item: QLineF(item.mapToScene(item.boundingRect().center()), pos).length())
        for item in items:
            if isinstance(item.data, np.ndarray):
                pass
            else:
                cluster_index = item.data(0) 
                if cluster_index is not None:
                    self.showSuggestions(cluster_index)
                    break
    
    
    def create_mouse_press_event_handler(self, label):
        def mouse_press_event(event):
            self.showSuggestions(label)
        return mouse_press_event

    def create_data(self, offline, count_offline_words, max_gpt_responses):
        """
        Находит обобщающие кластеры и разделяет эмбеддинги по кластерам
        """
        # Обобщающие фразы для каждого кластера
        self.summarizer = gpt_summarizer(offline, count_offline_words, max_gpt_responses)
        indexes = np.unique(self.clusters, return_index=True)[1]
        self.embeddings_clustered = []
        self.answers_clustered = []
        self.cluster_summaries = {}
        for i in range(len(indexes)):

            emb_clust = self.embeddings[self.clusters == i]
            ans_clust = self.answers[self.clusters == i]
            center_clust = np.mean(emb_clust, axis=0)
            distances = np.linalg.norm(emb_clust - center_clust, axis=1)
            phrase_distance_pairs = list(zip(ans_clust,emb_clust, distances))
            phrase_distance_pairs.sort(key=lambda x: x[2], reverse=False)
            sorted_phrases = np.array([pair[0] for pair in phrase_distance_pairs])
            sorted_emb = np.array([pair[1] for pair in phrase_distance_pairs])
            self.embeddings_clustered.append(sorted_emb)
            self.answers_clustered.append(sorted_phrases)
            self.cluster_summaries[i] = self.summarizer.summarize(sorted_phrases,
                                                                  sorted_emb)
            
    def showSuggestions(self, category_index):
        # Очищаем список предложений
        self.listWidget.clear()
        self.listWidget.addItem("Описание кластера {}: ".format(category_index))
        clustered_answers = self.answers_clustered[category_index]
        description = self.cluster_summaries[category_index]
        self.listWidget.addItem("Количество объектов в кластере:{}".format(len(clustered_answers)))
        self.listWidget.addItem(description)
        self.listWidget.addItem("============================================")
        # Заполняем список предложений
        for answer in clustered_answers:
            self.listWidget.addItem(self.censor.transform(answer))
    def on_result_ready(self, result):
        # Обработка результата после завершения потока
        self.embeddings_clustered, self.answers_clustered, self.cluster_summaries = result
        self.thread.quit()  # Завершение потока
        self.thread.wait()  # Ожидание завершения потока
        self.worker_thread = None
        self.draw_clusters()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    embeddings = np.load("dima/эмбеддинги2d.npy")
    clusters = np.load("dima/метки_кластеров.npy")
    answers = np.load("dima/ответы_сотрудников.npy")
    window = BarChartWidget()
    window.apply(embeddings, answers, clusters)
    window.initUI()
    window.show()
    sys.exit(app.exec_())