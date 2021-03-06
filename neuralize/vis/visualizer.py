import sys, random

from PyQt5.QtWidgets import QPushButton, QLabel, QMainWindow
from PyQt5.QtGui import QPainter, QBrush, QColor, QPen
from PyQt5.QtCore import Qt, QTimer

from neuralize.vis.ui_neuron import Neuron
from neuralize.vis.ui_synapse import Synapse

class NeuralizeMainWindow(QMainWindow):
    def __init__(self, ui_neural_net, training_input, expected_output, test_input, test_output):
        super().__init__()
        self.neural_net = ui_neural_net
        self.training_input = training_input
        self.expected_output = expected_output
        self.test_input = test_input
        self.test_output = test_output
        self.init_ui()
        self.init_ui_net()

    def _train_iterations(self):
        for i in range(10):
            ongoing = self.neural_net.train_one_iteration(self.training_input, self.expected_output)
        self.update_synapses()
        self.update_neurons()
        self.repaint()
        if not ongoing:
            self.timer.disconnect()

    def _start_training(self):
        self.training_button.setEnabled(False)
        self.timer = QTimer(self)
        self.timer.start(100)
        self.timer.timeout.connect(self._train_iterations)

    def init_ui(self):      
        self.setGeometry(0, 0, 1000, 1000)
        self.setWindowTitle('Neuralize')

        self.training_button = QPushButton('Start training', self)
        self.training_button.clicked.connect(self._start_training)
        self.training_button.resize(128, 32)
        self.training_button.move(25, 25)
        
        self.show()

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

        self.draw_metadata(painter)
        self.draw_net(painter)
        painter.end()

    def draw_metadata(self, painter):
        self._set_paint_color(painter, QColor(255, 0, 0))
        training_progress = self.neural_net.get_current_iteration() / self.neural_net.teaching_iterations
        painter.drawRect(0, 0, (training_progress) * 1000, 10)

        performance = self.neural_net.calculate_performance(self.test_input, self.test_output) / 100
        self._set_paint_color(painter, QColor(0, 255, 0))
        painter.drawRect(0, 10, (performance) * 1000, 10)

    def draw_net(self, painter):
        self._set_paint_color(painter, QColor(255, 0, 0))

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
                n = Neuron((ui_layer_width * (layer + 0.5), ui_layer_height * (i + 0.5)), (50, 50), 0)
                layer_neurons.append(n)
            self.neurons.append(layer_neurons)
    
    def create_synapses(self):
        for layer, synapse_layer in enumerate(self.neural_net.synapses):
            self.synapses.append([])
            for i, from_node_synapses in enumerate(synapse_layer):
                max_weight = max(max(from_node_synapses), -min(from_node_synapses))
                ui_synapse_list = [Synapse(self.neurons[layer][i], self.neurons[layer + 1][j], weight/max_weight) for j, weight in enumerate(from_node_synapses)]
                self.synapses[layer].append(ui_synapse_list)

    def update_synapses(self):
        for layer, synapse_layer in enumerate(self.neural_net.synapses):
            for from_index, from_node_synapses in enumerate(synapse_layer):
                for dest_index, synapse in enumerate(from_node_synapses):
                    self.synapses[layer][from_index][dest_index].update_weight(synapse)

    def update_neurons(self):
        for i, layer in enumerate(self.neural_net.biases):
            max_bias = max(max(layer[0]), -min(layer[0]))
            for j, bias in enumerate(layer[0]):
                self.neurons[i + 1][j].update_brush_color(bias / max_bias)

    def draw_neurons(self, painter):
        for layer in self.neurons:
            for neuron in layer:
                neuron.draw(painter)

    def draw_synapses(self, painter):
        for layer in self.synapses:
            for ui_synapse_list in layer:
                for ui_synapse in ui_synapse_list:
                    ui_synapse.draw(painter)

    def _set_paint_color(self, painter, qcolor):
        pen = QPen(qcolor)
        brush = QBrush(qcolor, Qt.SolidPattern)
        painter.setPen(pen)
        painter.setBrush(brush)
