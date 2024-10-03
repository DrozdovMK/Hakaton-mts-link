import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsEllipseItem, QGraphicsScene,
    QVBoxLayout, QWidget, QTextEdit
)
from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtCore import Qt, QPointF
import math

# Класс для эллипса
class EllipseItem(QGraphicsEllipseItem):
    def __init__(self, center, a, b, angle, index, *args, **kwargs):
        """
        center: кортеж (x, y)
        a: полуось по X
        b: полуось по Y
        angle: угол поворота в градусах
        index: уникальный идентификатор эллипса
        """
        super().__init__(*args, **kwargs)
        self.center = QPointF(*center)
        self.a = a
        self.b = b
        self.angle = angle
        self.index = index  # Идентификатор для отображения информации

        # Установка размеров и позиции эллипса
        self.setRect(-a, -b, 2*a, 2*b)
        self.setPos(self.center)
        self.setRotation(angle)

        # Настройка внешнего вида
        self.setBrush(QBrush(Qt.transparent))
        self.setPen(QPen(Qt.blue, 2))

        # Включение обработки событий мыши
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)

    def mousePressEvent(self, event):
        """
        Обработка события нажатия мыши на эллипс.
        Передаёт информацию о эллипсе в главное окно.
        """
        self.scene().parent.ellipse_clicked(self, event)
        super().mousePressEvent(event)

# Главное окно приложения
class MainWindow(QMainWindow):
    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Приложение с Эллипсами на PyQt5 и PyQtGraph")
        self.setGeometry(100, 100, 800, 600)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Вертикальный layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Создание сцены и представления
        self.scene = QGraphicsScene()
        self.scene.parent = self  # Передаём ссылку на главное окно

        self.view = self.create_graphics_view()
        self.view.setScene(self.scene)

        layout.addWidget(self.view)

        # Текстовый виджет под графиком
        self.text_widget = QTextEdit()
        self.text_widget.setReadOnly(True)
        layout.addWidget(self.text_widget)

        # Список эллипсов
        self.ellipses = []
        self.create_ellipses(data)

    def create_graphics_view(self):
        from PyQt5.QtWidgets import QGraphicsView
        view = QGraphicsView()
        # view.setRenderHint(view.viewport().Antialiasing)
        view.setMouseTracking(True)
        return view

    def create_ellipses(self, data):
        """
        data: список списков, каждый внутренний список содержит
              [центр (x, y), полуось a, полуось b, угол]
        """
        for idx, ellipse_data in enumerate(data):
            center, a, b, angle = ellipse_data
            ellipse = EllipseItem(center, a, b, angle, idx)
            self.scene.addItem(ellipse)
            self.ellipses.append(ellipse)

    def ellipse_clicked(self, clicked_ellipse, event):
        """
        Обработчик клика по эллипсу.
        Если эллипс пересекается с другими, выбирается ближайший к точке клика.
        """
        # Координаты клика в сцене
        click_pos = self.view.mapToScene(event.pos().x(), event.pos().y())

        # Получение всех эллипсов под точкой клика
        items = self.scene.items(QPointF(click_pos))

        # Фильтрация только эллипсов
        ellipses_under_click = [item for item in items if isinstance(item, EllipseItem)]

        if not ellipses_under_click:
            return  # Нет эллипсов под кликом

        if len(ellipses_under_click) == 1:
            selected = ellipses_under_click[0]
        else:
            # Если несколько эллипсов, выбрать ближайший их центру
            min_distance = float('inf')
            selected = None
            for ellipse in ellipses_under_click:
                distance = self.distance(click_pos, ellipse.center)
                if distance < min_distance:
                    min_distance = distance
                    selected = ellipse

        # Отображение информации
        info = f"Эллипс #{selected.index}:\n" \
               f"Центр: ({selected.center.x()}, {selected.center.y()})\n" \
               f"Полуось a: {selected.a}\n" \
               f"Полуось b: {selected.b}\n" \
               f"Угол поворота: {selected.angle}°"
        self.text_widget.setText(info)

    @staticmethod
    def distance(p1, p2):
        return math.hypot(p1.x() - p2.x(), p1.y() - p2.y())

def main():
    # Пример данных
    data = [
        [(100, 100), 50, 30, 30],
        [(150, 150), 40, 20, -45],
        [(200, 100), 60, 40, 15],
        # Добавьте больше эллипсов по необходимости
    ]

    app = QApplication(sys.argv)
    window = MainWindow(data)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()