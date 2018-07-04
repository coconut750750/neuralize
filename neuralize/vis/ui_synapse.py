from PyQt5.QtGui import QPen, QColor
from PyQt5.QtCore import Qt

class Synapse:
    def __init__(self, from_neuron, to_neuron, weight, color=QColor(0, 0, 255)):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight
        half_rgb = [i / 2 for i in color.getRgb()[:3]]
        weighted_half_rgb = [i * weight for i in half_rgb]
        weighted_rgb = [value + half_rgb[i] for i, value in enumerate(weighted_half_rgb)]
        print(weight, weighted_rgb)
        self.color = QColor(*weighted_rgb, 255)

    def draw(self, painter):
        pen = QPen(self.color)
        pen.setWidth(3);
        painter.setPen(pen);
        painter.drawLine(*self.from_neuron.pos, *self.to_neuron.pos)

    def update_weight(self, new_weight):
        self.weight = new_weight
        self.color = self.color
