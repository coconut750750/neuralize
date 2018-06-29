import numpy as np
from random import shuffle

from iris_data import iris_data, iris_labels

class NeuralNet(object):
    def __init__(self, num_layers, layer_sizes,
                 training_iterations=1000,
                 learning_rate=1):
        '''
        Creates a Neural Net object
        num_layers -- number of neuron layers including the input and output layer
        layer_sizes -- number of neurons in each layer
        training_iterations -- number of times to train neural net
        learning_rate -- how fast neural net should learn
        '''
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.training_iterations = training_iterations
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
        for i in range(self.training_iterations):
            layer_activations = self._forward(input)
            self.backward(expected_output, layer_activations)

    def predict(self, input):
        return self._forward(input)[-1]

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    all_data = np.array(iris_data)

    num_data = len(all_data)
    training_percent = 0.8
    training_size = int(num_data * training_percent)

    poss_labels = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    labels = np.array([poss_labels[int(i[0]) - 1] for i in all_data])
    features = np.array([i[1:] for i in all_data])
    features = (features - features.min(axis=0)) / features.max(axis=0)

    training_set = np.array(features[0: training_size])
    expected = np.array(labels[0: training_size])

    test_data = np.array(features[training_size:])
    test_expected = np.array(labels[training_size:])

    neural_net = NeuralNet(3, [4, 3, 3], training_iterations=50000, learning_rate=0.05)

    predicted = neural_net.predict(training_set)
    initial_performance = np.mean(expected - predicted)

    neural_net.train(training_set, expected)

    predicted = neural_net.predict(training_set)
    trained_performance = np.mean(expected - predicted)

    test_predicted = neural_net.predict(test_data)
    print("Loss: " + str(np.mean(test_expected - test_predicted)))
    for i, predict_data in enumerate(test_predicted):
        # print(predict_data, test_expected[i])
        prediction = iris_labels[predict_data.argmax() + 1]
        actual = iris_labels[test_expected[i].argmax() + 1]
        print("Predicted: {}\t Actual: {}".format(prediction, actual))

