import numpy as np

from neuralize.core.activation import Activation

class SoftmaxActivation(Activation):
    def __init__(self):
        self.name = 'Softmax'

    def compute(self, x):
        exp = np.exp(x)
        return exp / exp.sum(axis=1, keepdims=True)

    def gradient(self, x):
        return 1
        # s = x.reshape((-1, 1))
        # return np.diag(s) - np.dot(s.T, s)
