class Activation:
    def __init__(self):
        self.name = 'Base Activation'

    def compute(self, x):
        raise RuntimeError('Please implement compute() in your subclass.')

    def gradient(self, x):
        raise RuntimeError('Please implement gradient() in your subclass.')

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)
