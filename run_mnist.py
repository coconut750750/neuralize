import numpy as np

from neuralize.core.neural_net import NeuralNet
from neuralize.data.mnist_data import setup_data, data_to_str

def train_mnist():
    learn_set, learn_expected, test_set, test_expected = setup_data()

    inputs, outputs = len(learn_set[0]), len(learn_expected[0])

    neural_net = NeuralNet(3, [inputs, 16, outputs],
                           teaching_iterations=100,
                           learning_rate=0.0001)

    neural_net.train(learn_set, learn_expected)

    loss = neural_net.calculate_loss(learn_set, learn_expected)
    performance = neural_net.calculate_performance(test_set, test_expected)
    print('Loss: {}\tTest performance: {}'.format(loss, performance))

    np.set_printoptions(suppress=True)
    sample_index = 3
    single_test = test_set[sample_index]
    print(neural_net.predict(single_test))
    print(test_expected[sample_index])

    with open('sample_expected.txt', 'w') as f:
        f.write(data_to_str(single_test))

if __name__ == '__main__':
    train_mnist()