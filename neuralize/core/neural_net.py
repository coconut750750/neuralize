import numpy as np
import pickle
from neuralize.core.sigmoid import SigmoidActivation
from neuralize.core.tanh import TanHActivation

class NeuralNet(object):
    def __init__(self, num_layers, layer_sizes, activations,
                 teaching_iterations=10000,
                 tau=50000, kappa=1.0, num_batches=10):
        '''
        Creates a Neural Net object
        num_layers -- number of neuron layers including the input and output layer
        layer_sizes -- number of neurons in each layer
        activations -- the activation function for each layer excluding input
        teaching_iterations -- number of times to teach neural net
        tau, kappa -- used to determine how fast neural net should learn
        num_batches -- number of batches to split train data into every learn iteration
        '''
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.teaching_iterations = teaching_iterations
        self.tau = tau
        self.kappa = kappa
        self.num_batches = num_batches
        self.synapses = [np.random.random((self.layer_sizes[i], self.layer_sizes[i + 1])) * 2 - 1\
                         for i in range(self.num_layers - 1)]
        self.biases = [np.random.random((1, self.layer_sizes[i + 1])) * 2 - 1\
                       for i in range(self.num_layers - 1)]

    def _forward(self, input):
        outputs = [input]
        
        for i in range(self.num_layers - 1):
            z = np.dot(outputs[i], self.synapses[i]) + self.biases[i]
            outputs.append(self.activations[i].compute(z))
        
        return outputs

    def _backward(self, expected_output, layer_activations):
        last_layer_error = expected_output - layer_activations[self.num_layers - 1]

        l = self.num_layers - 1
        prev_delta = self._adjust(l, last_layer_error, layer_activations)

        for l in range(self.num_layers - 2, 0, -1):
            layer_error = prev_delta.dot(self.synapses[l].T)
            prev_delta = self._adjust(l, layer_error, layer_activations)

    def _adjust(self, layer_num, layer_error, layer_activations):
        prev_delta = layer_error * self.activations[layer_num - 1].gradient(layer_activations[layer_num])
        self.synapses[layer_num - 1] += layer_activations[layer_num - 1].T.dot(prev_delta) * self.learning_rate
        bias_delta = np.sum(prev_delta, axis=0)
        self.biases[layer_num - 1] += bias_delta * self.learning_rate
        return prev_delta

    def train(self, training_input, expected_output, display_progress=False):
        batch_size = len(training_input) // self.num_batches + 1
        for i in range(self.teaching_iterations):
            self.learning_rate = np.power((self.tau + i), -self.kappa)
            for j in range(0, len(training_input), batch_size):
                layer_activations = self._forward(training_input[j: j + batch_size])
                self._backward(expected_output[j: j + batch_size], layer_activations)
            if display_progress:
                print("Learning Iteration: {}".format(i))

    def predict(self, input):
        return self._forward(input)[-1]

    def calculate_loss(self, input, expected):
        predicted = self.predict(input)
        return round(abs(np.mean(expected - predicted)), 8)

    def calculate_performance(self, test_in, test_expected):
        test_predicted = self.predict(test_in)
        total = len(test_predicted)
        correct = 0
        for i, predict_data in enumerate(test_predicted):
            correct += 1 if predict_data.argmax() == test_expected[i].argmax() else 0
        return round(correct / total * 100, 2)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            nn = pickle.load(f)
            nn.tau = 50000
            nn.kappa = 1.0
            nn.num_batches = 10
            return nn

    def __str__(self):
        description = 'Neural Network:\n'
        description += 'Layers: {}\n'.format(self.num_layers)
        description += 'Layer sizes: {}\n'.format(self.layer_sizes)
        description += 'Layer Activations: {}\n'.format(self.activations)
        description += 'Iterations: {}\n'.format(self.teaching_iterations)
        description += 'Initial learn rate: {}'.format(np.power(self.tau, -self.kappa))
        return description