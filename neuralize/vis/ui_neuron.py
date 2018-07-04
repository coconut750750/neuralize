from PyQt5.QtGui import QPen, QBrush, QColor
from PyQt5.QtCore import Qt

class Neuron:
    def __init__(self, pos, size, activation, color=QColor(255, 0, 0)):
        self.pos = pos
        self.size = size
        self.color = color
        self.update_brush_color(activation)

    def draw(self, painter):
        pen = QPen(self.color)
        pen.setWidth(3)
        painter.setPen(pen)

        ellipse_config = (self.pos[0] - self.size[0] / 2, self.pos[1] - self.size[1] / 2, *self.size)
        painter.setBrush(QBrush(QColor(255, 255, 255), Qt.SolidPattern));
        painter.drawEllipse(*ellipse_config)
        
        brush = QBrush(self.brush_color, Qt.SolidPattern)
        painter.setBrush(brush);
        painter.drawEllipse(*ellipse_config)

    def update_brush_color(self, new_activation):
        self.activation = new_activation
        new_alpha = (new_activation * 255)
        self.brush_color = QColor(*self.color.getRgb()[:3], new_alpha)