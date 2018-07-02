class Activation:
    def compute(self, x):
        raise RuntimeError('Please implement compute() in your subclass.')

    def gradient(self, x):
        raise RuntimeError('Please implement gradient() in your subclass.')
