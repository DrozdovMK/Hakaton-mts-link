import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsItem
from PyQt5.QtGui import QPen, QBrush, QTransform, QPainter
from PyQt5.QtCore import QRectF, Qt

class CustomEllipse(QGraphicsEllipseItem):
    def __init__(self, center_x, center_y, radius_x, radius_y, angle, num_points=100, pen=None, brush=None):
        super().__init__()

        self.center_x = center_x
        self.center_y = center_y
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.angle = angle

        # Создаем прямоугольник для эллипса
        self.setRect(-radius_x, -radius_y, 2 * radius_x, 2 * radius_y)

        # Устанавливаем стиль обводки и заливки
        self.setPen(pen or QPen(Qt.black))
        self.setBrush(brush or QBrush(Qt.green))

        # Устанавливаем трансформацию (поворот и смещение центра)
        self.setTransform(QTransform().translate(center_x, center_y).rotate(angle))

        # Генерируем набор точек на границе эллипса
        self.points = self.generate_points(num_points)

    def generate_points(self, num_points):
        # Генерация углов в радианах
        angles = np.linspace(0, 2 * np.pi, num_points)
        points = []

        for angle in angles:
            x = self.center_x + self.radius_x * np.cos(angle) * np.cos(np.radians(self.angle)) - \
                self.radius_y * np.sin(angle) * np.sin(np.radians(self.angle))

            y = self.center_y + self.radius_x * np.cos(angle) * np.sin(np.radians(self.angle)) + \
                self.radius_y * np.sin(angle) * np.cos(np.radians(self.angle))

            points.append((x, y))

        return points

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)

        # Рисуем точки на границе эллипса
        pen = QPen(Qt.red)
        painter.setPen(pen)
        for point in self.points:
            painter.drawPoint(point[0], point[1])

class MyGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)

        scene = QGraphicsScene()
        self.setScene(scene)

        ellipse = CustomEllipse(center_x=100, center_y=100, radius_x=80, radius_y=50, angle=30, num_points=100)
        scene.addItem(ellipse)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    view = MyGraphicsView()
    view.setGeometry(100, 100, 400, 400)
    view.show()
    sys.exit(app.exec_())