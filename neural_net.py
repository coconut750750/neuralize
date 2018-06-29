import numpy as np

class NeuralNet(object):
    def __init__(self, num_layers, layer_sizes,
                 teaching_iterations=1000,
                 learning_rate=1):
        '''
        Creates a Neural Net object
        num_layers -- number of neuron layers including the input and output layer
        layer_sizes -- number of neurons in each layer
        teaching_iterations -- number of times to teach neural net
        learning_rate -- how fast neural net should learn
        '''
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.teaching_iterations = teaching_iterations
        self.learning_rate = learning_rate

        self.synapses = [np.random.random((self.layer_sizes[i], self.layer_sizes[i + 1])) * 2 - 1\
                         for i in range(self.num_layers - 1)]
        self.biases = [np.random.random((1, self.layer_sizes[i + 1])) * 2 - 1\
                         for i in range(self.num_layers - 1)]

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)

    def _forward(self, input):
        outputs = [input]
        
        for i in range(self.num_layers - 1):
            z = np.dot(outputs[i], self.synapses[i]) + self.biases[i]
            outputs.append(self.sigmoid(z))
        
        return outputs

    def backward(self, expected_output, layer_activations):
        last_layer_error = expected_output - layer_activations[self.num_layers - 1]
        prev_delta = last_layer_error * self.sigmoidPrime(layer_activations[self.num_layers - 1])
        self.synapses[self.num_layers - 2] += layer_activations[self.num_layers - 2].T.dot(prev_delta) * self.learning_rate

        for i in range(self.num_layers - 2, 0, -1):
            layer_error = prev_delta.dot(self.synapses[i].T)
            prev_delta = layer_error * self.sigmoidPrime(layer_activations[i])
            self.synapses[i - 1] += layer_activations[i - 1].T.dot(prev_delta) * self.learning_rate

    def train(self, input, expected_output):
        for i in range(self.teaching_iterations):
            layer_activations = self._forward(input)
            self.backward(expected_output, layer_activations)

    def predict(self, input):
        return self._forward(input)[-1]
