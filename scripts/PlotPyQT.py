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
from PyQt5.QtCore import pyqtSignal, QThread
from sklearn.decomposition import PCA
from scripts.profanity_check import profanity_processing

# посчитали количество каждого числа в кластере
def count_numbers(numbers):
  counts = {}
  for number in numbers:
    if number in counts:
      counts[number.item()]["size"] += 1
    else:
      counts[number.item()] = {}
      counts[number.item()]["size"] = 1
  return counts

# заполнили для каждого кластера всевозможные координаты
def add_coordinats(data, clusters, embeddings):
    for i, embedding in enumerate(embeddings):
        if "xy" in data[clusters[i].item()]:
            data[clusters[i].item()]["xy"].append(embedding)
        else:
            data[clusters[i].item()]["xy"] = [embedding]

    return data
def ellipse_settings(points):
    centroid = np.mean(points, axis=0)
    pca = PCA(n_components=2)
    pca.fit(points)
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_
    x_radius, y_radius = np.sqrt(eigenvalues) * 2
    angle = np.degrees(np.arctan2(*eigenvectors[0][::-1]))
    return centroid, x_radius, y_radius, angle
    

# для каждой координаты найдем центр
def center_of_coordinates(data):
    for clust in data:
       data[clust]["x"] = np.mean(np.array(data[clust]["xy"]), axis=0).tolist()[0]
       data[clust]["y"] = np.mean(np.array(data[clust]["xy"]), axis=0).tolist()[1]
       data[clust].pop("xy")

    return data

def add_text_value(data, clusters, answers):
    for i, clust in enumerate(clusters):
        if "text" in data[clusters[i].item()]:
            data[clusters[i].item()]["text"].append(answers[i].item())
        else:
            data[clusters[i].item()]["text"] = [answers[i].item()]
    return data

class CustomEllipseItem(QGraphicsEllipseItem):
    clicked = pyqtSignal(object)
    def __init__(self, centriod, width, height, angle, label):
        super().__init__(-width, -height, width*2, height*2)
        self.setRotation(angle)
        self.setPos(centriod[0], centriod[1])
        self.setPen(pg.mkPen(color=(0, 0, 200), width=2))
        self.setBrush(pg.mkBrush(100, 100, 250, 80))
        self.label = label

class CustomGraphicsScene(QGraphicsScene):
    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), QTransform())
        if isinstance(item, CustomEllipseItem):
            item.on_click()

class BarChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.censor = profanity_processing()
        self.radius_const = 10
        self.setWindowTitle("Облако слов")
        # Создаем фигуру и холст для графика
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        # Создаем элементы интерфейса
        self.listWidget = QListWidget()
        # Собираем все элементы в макет
        mainLayout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        mainLayout.addWidget(self.plot_widget, 1)
        self.setLayout(mainLayout)
        mainLayout.addWidget(self.listWidget, 1)
        
    def apply(self, embeddings, answers, clusters, offline, count_offline_words, max_gpt_responses ):
    
        self.summarizer = gpt_summarizer(offline, count_offline_words, max_gpt_responses )
        self.answers = answers
        self.embeddings = embeddings
        self.clusters = clusters
        self.data_plot = count_numbers(self.clusters)
        self.data_plot = add_coordinats(self.data_plot, self.clusters, self.embeddings)
        self.data_plot = center_of_coordinates(self.data_plot)
        self.data_plot = add_text_value(self.data_plot, self.clusters, self.answers)
        self.data = {}
        self.suggestions = {}
        
        for i in self.data_plot:
            self.suggestions[i] = self.data_plot[i]["text"]
            self.data[i] = self.data_plot[i]["size"]
        self.initUI()

    def initUI(self):
        self.create_data()
        self.draw_clusters()

    def draw_clusters(self):
        """
        Отрисовывает кластеры
        """
        # Очищаем график перед перерисовкой
        self.plot_widget.clear()
        # Сохраняем объекты кругов для обработки событий
        for i, embed_clustered in enumerate(self.embeddings_clustered):
            centroid, width, height, angle = ellipse_settings(embed_clustered)
            # Добавим точки
            scatter = pg.ScatterPlotItem(x = embed_clustered[:, 0],
                                         y = embed_clustered[:, 1],
                                         pen=pg.mkPen(width=1, color='r'), symbol='o', size=1)
            self.plot_widget.addItem(scatter)
            
            # Создаем эллипс
            ellipse = CustomEllipseItem(centroid, width, height, angle, label = i)
            ellipse.mousePressEvent = self.create_mouse_press_event_handler(ellipse.label)
            # Добавляем эллипс на график
            self.plot_widget.addItem(ellipse)
            
            # Добавляем обобщающую фразу в центр круга
            text = pg.TextItem("Кластер {}".format(i), anchor=(0.5, 0.5), color=(255, 255, 255))
            text.setPos(centroid[0], centroid[1])
            text.setData(0, i)
            text.setData(1, self.censor.transform(self.cluster_summaries[i]))
            self.plot_widget.addItem(text)

    
    def create_mouse_press_event_handler(self, label):
        def mouse_press_event(event):
            self.showSuggestions(label)
        return mouse_press_event

    def create_data(self):
        """
        Находит обобщающие кластеры и разделяет эмбеддинги по кластерам
        """
        # Обобщающие фразы для каждого кластера
        indexes = np.unique(self.clusters, return_index=True)[1]
        self.embeddings_clustered = []
        self.cluster_summaries = {}
        for i in range(len(indexes)):
            self.cluster_summaries[i] = self.censor.transform(self.summarizer.summarize(self.answers[self.clusters == i],
                                                                  self.embeddings[self.clusters == i]))
            self.embeddings_clustered.append(self.embeddings[self.clusters == i])
            
    def plot_graph(self):
        fig = go.Figure()
        for key, values in self.data_plotly.items():
            fig.add_trace(go.Scatter(
                x=[values["x"]],
                y=[values["y"]],
                mode='markers',
                marker=dict(
                    size=values["size"],
                    color='blue',
                    opacity=0.8
                ),
                text=values["text"],
                hovertemplate='<b>%{text}</b> \
                    X: %{x:.2f} \
                    Y: %{y:.2f} \
                    Size: %{marker.size:.2f}<extra></extra>'
            ))
        fig.update_layout(
            title="Интерактивный график",
            xaxis_title="X",
            yaxis_title="Y"
        )
        pio.write_image(fig, "plotly_figure.svg")
        
    def showSuggestions(self, category_index):
        # Очищаем список предложений
        self.listWidget.clear()
        description = self.cluster_summaries[category_index]
        self.listWidget.addItem(description)
        # Заполняем список предложений
        for suggestion in self.suggestions[category_index]:
            self.listWidget.addItem(self.censor.transform(suggestion))

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