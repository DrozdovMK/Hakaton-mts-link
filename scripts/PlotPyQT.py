import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QGraphicsEllipseItem
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

import pyqtgraph as pg


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


class BarChartWidget(QWidget):
    def __init__(self):
        super().__init__()
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
        
    def apply(self, embeddings, answers, clusters):
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
            self.suggestions[str(i)] = self.data_plot[i]["text"]
            self.data[str(i)] = self.data_plot[i]["size"]
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
        self.circles = []
        for i, (center, size) in enumerate(zip(self.cluster_centers, self.cluster_sizes)):
            # Размер круга зависит от количества фраз в кластере
            radius = size * 0.05
            # Создаем круг
            circle = QGraphicsEllipseItem(-radius, -radius, radius*2, radius*2)
            circle.setPen(pg.mkPen(color=(0, 0, 200), width=2))
            circle.setBrush(pg.mkBrush(100, 100, 250, 80))
            circle.setPos(center[0], center[1])
            circle.setData(1, self.cluster_summaries[i])
            # Добавляем круг на график
            self.plot_widget.addItem(circle)
            self.circles.append(circle)
            # Добавляем обобщающую фразу в центр круга
            text = pg.TextItem(f'Кластер {self.cluster_summaries[i]}', anchor=(0.5, 0.5), color=(0, 0, 0))
            text.setPos(center[0], center[1])
            text.setData(1, self.cluster_summaries[i])
            self.plot_widget.addItem(text)

        # Настраиваем событие клика мыши
        self.plot_widget.scene().sigMouseClicked.connect(self.on_click)


    def create_data(self):
        """
        Находит центры кластеров и их размер
        """
        # Набор коротких фраз
        self.phrases = self.answers
        # Метки кластеров
        self.labels = self.clusters
        # Обобщающие фразы для каждого кластера
        indexes = np.unique(self.clusters, return_index=True)[1]
        self.cluster_summaries = [self.clusters[index] for index in sorted(indexes)]
        # Вычисляем центры и размеры кластеров
        self.cluster_centers = []
        self.cluster_sizes = []
        for i in range(len(self.cluster_summaries)):
            indices = np.where(self.labels == i)[0]
            center = np.mean(self.embeddings[indices], axis=0)
            size = len(indices)
            self.cluster_centers.append(center)
            self.cluster_sizes.append(size)

    def on_click(self, event):
        pos = event.scenePos()
        items = self.plot_widget.scene().items(pos)
        # Сортируем элементы по расстоянию до точки pos
        from PyQt5.QtCore import QPointF, QLineF
        items.sort(key=lambda item: QLineF(item.mapToScene(item.boundingRect().center()), pos).length())
        for item in items:
            cluster_index = item.data(1) 
            # print(cluster_index)
            if cluster_index is not None:
                self.showSuggestions(str(cluster_index))
                break

    
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
        
    def showSuggestions(self, category):
        # Очищаем список предложений
        self.listWidget.clear()
        # Заполняем список предложений
        for suggestion in self.suggestions[category]:
            self.listWidget.addItem(suggestion)

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