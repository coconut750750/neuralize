import sys, random

from PyQt5.QtWidgets import QPushButton, QMainWindow
from PyQt5.QtGui import QPainter, QBrush, QColor, QPen
from PyQt5.QtCore import Qt

from neuralize.vis.ui_neuron import Neuron
from neuralize.vis.ui_synapse import Synapse

class NeuralizeMainWindow(QMainWindow):
    def __init__(self, neural_net):
        super().__init__()
        self.neural_net = neural_net
        self.init_ui()
        self.init_ui_net()

    def _train_net(self):
        print('Hello World')

    def init_ui(self):      
        self.setGeometry(0, 0, 500, 500)
        self.setWindowTitle('Neuralize')

        self.button = QPushButton('Test', self)
        self.button.clicked.connect(self._train_net)
        self.button.resize(100,32)
        self.button.move(50, 50)

        self.showMaximized()

    def init_ui_net(self):
        self.neurons = []
        self.synapses = []
        self.create_neurons()
        self.create_synapses()

    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.HighQualityAntialiasing)
        self.draw_net(painter)
        painter.end()

    def draw_net(self, painter):
        pen = QPen(QColor(255, 0, 0))
        brush = QBrush(QColor(255, 0, 0), Qt.SolidPattern)
        painter.setPen(pen)
        painter.setBrush(brush);
        
        self.draw_synapses(painter)
        self.draw_neurons(painter)

    def create_neurons(self):
        size = self.size()
        ui_layer_width = size.width() // (self.neural_net.num_layers)
        for layer in range(self.neural_net.num_layers):
            layer_size = self.neural_net.layer_sizes[layer]
            ui_layer_height = y = size.height() // (layer_size)
            layer_neurons = []
            for i in range(layer_size):
                n = Neuron((ui_layer_width * (layer + 0.5), ui_layer_height * (i + 0.5)), (50, 50), 100)
                layer_neurons.append(n)
            self.neurons.append(layer_neurons)
        
    def draw_neurons(self, painter):
        for layer in self.neurons:
            for neuron in layer:
                neuron.draw(painter)

    def create_synapses(self,):
        for layer in range(self.neural_net.num_layers - 1):
            synapse_layer = self.neural_net.synapses[layer]
            self.synapses.append([])
            for i, from_node_synapses in enumerate(synapse_layer):
                max_weight = max(from_node_synapses)
                min_weight = min(from_node_synapses)
                max_weight = max(max_weight, -min_weight)
                ui_synapse_list = [Synapse(self.neurons[layer][i], self.neurons[layer + 1][j], weight/max_weight) for j, weight in enumerate(from_node_synapses)]
                self.synapses[layer].append(ui_synapse_list)

    def draw_synapses(self, painter):
        for layer in self.synapses:
            for i, ui_synapse_list in enumerate(layer):
                for ui_synapse in ui_synapse_list:
                    ui_synapse.draw(painter)
