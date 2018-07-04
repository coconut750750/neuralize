import numpy as np
from neuralize.core.neural_net import NeuralNet

class UINeuralNet(NeuralNet):
    def __init__(self, num_layers, layer_sizes, activations,
                 teaching_iterations=10000,
                 tau=50000, kappa=0.5, gamma=-1, num_batches=1):
        super().__init__(num_layers, layer_sizes, activations, teaching_iterations, tau, kappa, gamma, num_batches)
        self.iters_left = teaching_iterations

    def train_one_iteration(self, training_input, expected_output):
        if not self.iters_left:
            return False

        self.iters_left -= 1
        iteration = self.teaching_iterations - self.iters_left
        self.learning_rate = np.power((self.tau + iteration), -self.kappa)

        layer_activations = self._forward(training_input)
        self._backward(expected_output, layer_activations)

        return True