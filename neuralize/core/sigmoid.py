import numpy as np

from neuralize.core.activation import Activation

class SigmoidActivation(Activation):
    def __init__(self):
        pass

    def compute(self, x):
        exp = np.exp(x)
        return exp / (exp + 1)

    def gradient(self, x):
        return x * (1 - x)
