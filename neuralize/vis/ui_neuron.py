from PyQt5.QtGui import QPen, QBrush, QColor
from PyQt5.QtCore import Qt

class Neuron:
    def __init__(self, pos, size, bias, color=QColor(255, 0, 0)):
        self.pos = pos
        self.size = size
        self.base_color = color
        self.pen_color = color
        self.update_brush_color(bias)

    def draw(self, painter):
        pen = QPen(self.pen_color)
        pen.setWidth(3)
        painter.setPen(pen)

        ellipse_config = (self.pos[0] - self.size[0] / 2, self.pos[1] - self.size[1] / 2, *self.size)
        painter.setBrush(QBrush(QColor(255, 255, 255), Qt.SolidPattern));
        painter.drawEllipse(*ellipse_config)
        
        brush = QBrush(self.brush_color, Qt.SolidPattern)
        painter.setBrush(brush);
        painter.drawEllipse(*ellipse_config)

    def update_brush_color(self, new_bias):
        self.bias = new_bias
        new_alpha = new_bias / 2 + 0.5
        new_alpha = (new_alpha * (255 / 2)) + (255 / 4)
        self.brush_color = QColor(*self.base_color.getRgb()[:3], new_alpha)
