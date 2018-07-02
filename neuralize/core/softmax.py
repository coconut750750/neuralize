import numpy as np

from neuralize.core.activation import Activation

class SoftmaxActivation(Activation):
    def __init__(self):
        pass

    def compute(self, x):
        exp = np.exp(x)
        return exp / exp.sum()

    def gradient(self, x):
        s = x.reshape((-1, 1))
        return np.diag(s) - np.dot(s.T, s)
