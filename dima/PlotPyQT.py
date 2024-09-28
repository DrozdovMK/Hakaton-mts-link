import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget
)
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
import plotly.graph_objects as go
from plotly.offline import iplot

import numpy as np
# Стандартное импортирование plotly
# from chart_studio import plotly
# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly.offline import iplot
embeddings = list(np.load("эмбеддинги2d.npy"))
clusters = list(np.load("метки_кластеров.npy"))

# посичтали количесвто каждого числа в кластере
def count_numbers(numbers):
  counts = {}
  for number in numbers:
    if number in counts:
      counts[number.item()]["size"] += 1
    #   counts[number]["xy"].append(embeddings[i])
    else:
      counts[number.item()] = {}
      counts[number.item()]["size"] = 1
    #   counts[number]["xy"] = [embeddings[i]]
  return counts
data = count_numbers(clusters)

# заполнили для каждого кластера всевозможные координаты
def add_coordinats(data, clusters, embeddings):
    for i, embedding in enumerate(embeddings):
        if "xy" in data[clusters[i].item()]:
            data[clusters[i].item()]["xy"].append(embedding)
        else:
            data[clusters[i].item()]["xy"] = [embedding]

    return data
data = add_coordinats(data, clusters, embeddings)

# для каждой координаты найдем центр
def center_of_coordinates(data):
    for clust in data:
    #    print(data[clust])
    #    print(data[clust]["xy"])
    #    print(np.mean(data[clust]["xy"], axis=0).tolist()[0])
       data[clust]["x"] = np.mean(np.array(data[clust]["xy"]), axis=0).tolist()[0]
       data[clust]["y"] = np.mean(np.array(data[clust]["xy"]), axis=0).tolist()[1]
       data[clust].pop("xy")

    return data

data_pot = center_of_coordinates(data)

answers = np.load("ответы_сотрудников.npy")
def add_text_value(data, clustesr, answers):
    for i, clust in enumerate(clusters):
        if "text" in data[clustesr[i].item()]:
            data[clustesr[i].item()]["text"].append(answers[i].item())
        else:
            data[clustesr[i].item()]["text"] = [answers[i].item()]

    return data
    
data = add_text_value(data, clusters, answers)


class BarChartApp(QWidget):
    def __init__(self, data, suggestions):
        super().__init__()

        self.data = data
        self.suggestions = suggestions

        self.data_plotly = data_pot
        self.radius_const = 30

        for i in self.data_plotly:
            self.data_plotly[i]["size"] = self.data_plotly[i]["size"]*self.radius_const

        # self.data_plotly = {
        #     "деньги": {"x": 10, "y": 12, "size": 36, "text": ["мани", "бабки"]},
        #     "интерес": {"x": 14, "y": 5, "size": 50, "text": ["интерес1", "интерес2"]},
        #     "время": {"x": 30, "y": 20, "size": 100, "text": ["секунда", "часы"]}
        # }

        self.initUI()

    def initUI(self):
        self.setWindowTitle("BarChart App")

        # Создаем фигуру и холст для графика
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Создаем элементы интерфейса
        self.label = QLabel("Выберите категорию")
        self.listWidget = QListWidget()
        self.buttonLayout = QHBoxLayout()
        self.button_plot = QPushButton("Построить график")

        # Создаем кнопки для каждой категории
        for key in self.data:
            button = QPushButton(key)
            button.clicked.connect(lambda checked, key=key: self.showSuggestions(key))
            self.buttonLayout.addWidget(button)

        
        self.button_plot.clicked.connect(self.plot_graph)

        # Собираем все элементы в макет
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.canvas)
        mainLayout.addWidget(self.label)
        mainLayout.addWidget(self.listWidget)
        mainLayout.addLayout(self.buttonLayout)
        mainLayout.addWidget(self.label)
        mainLayout.addWidget(self.button_plot)

        self.setLayout(mainLayout)

        # Рисуем график
        self.drawBarChart()

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
                    Size: %{marker.size}<extra></extra>'
            ))

        fig.update_layout(
            title="Интерактивный график",
            xaxis_title="X",
            yaxis_title="Y"
        )

        iplot(fig)

    def drawBarChart(self):
        # Очищаем предыдущий график
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Рисуем столбчатый график
        ax.bar(self.data.keys(), self.data.values())
        ax.set_xlabel("Категории")
        ax.set_ylabel("Значения")

        # Обновляем холст
        self.canvas.draw()

    def showSuggestions(self, category):
        # Очищаем список предложений
        self.listWidget.clear()

        # Заполняем список предложений
        for suggestion in self.suggestions[category]:
            self.listWidget.addItem(suggestion)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Пример данных
    data = {
        "деньги": 59,
        "интерес": 30,
        "время": 100,
        "удовольствие": 80,
        "успех": 95,
        "счастье": 70
    }

    suggestions = {
        "деньги": [ "зп", "бабки", "мани", "бабосики", "зп", "бабки", "мани", "бабосики", "зп", "бабки", "мани", "бабосики", "зп", "бабки", "мани", "бабосики", "зп", "бабки", "мани", "бабосики", "зп", "бабки", "мани", "бабосики"],
        "интерес": [ "велик", "качаалка"],
        "время": [ "часики", "график", "расписание", "длительность"],
        "удовольствие": [ "вкусности", "плойка", "спа", "бассейн"],
        "успех": [ "Грамота", "диплом", "тачка"],
        "счастье": [ "семья", "дети"]
    }

    window = BarChartApp(data, suggestions)
    window.show()
    sys.exit(app.exec_())