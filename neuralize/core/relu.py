import numpy as np

from neuralize.core.activation import Activation

class ReluActivation(Activation):
    def __init__(self, alpha=0):
        self.alpha = alpha
        self.name = 'ReLU'

    def compute(self, x):
        return np.maximum(x, x * self.alpha)

    def gradient(self, x):
        grad = x > 0
        grad = grad.astype(float)
        grad *= (1 - self.alpha)
        grad += self.alpha
        return grad
