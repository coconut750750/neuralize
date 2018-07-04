from PyQt5.QtGui import QPen, QColor
from PyQt5.QtCore import Qt

class Synapse:
    def __init__(self, from_neuron, to_neuron, weight, color=QColor(0, 0, 255)):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight
        self.color = color
        self.width = weight
        self.set_width(weight)

    def draw(self, painter):
        pen = QPen(self.color)
        pen.setWidth(self.width);
        painter.setPen(pen);
        painter.drawLine(*self.from_neuron.pos, *self.to_neuron.pos)

    def update_weight(self, new_weight):
        self.weight = new_weight
        self.set_width(new_weight)

    def set_width(self, weight):
        self.width = abs(weight)