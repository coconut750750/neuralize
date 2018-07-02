import numpy as np

from neuralize.core.activation import Activation

class ReluActivation(Activation):
    def __init__(self):
        pass

    def compute(self, x):
        return np.maximum(x, 0)

    def gradient(self, x):
        return x > 0
