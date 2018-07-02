import numpy as np

from neuralize.core.activation import Activation

class TanHActivation(Activation):
    def __init__(self):
        pass

    def compute(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1.0 - np.tanh(x) ** 2
