import numpy as np
import matplotlib.pyplot as plt

from iris_data import setup_data, iris_labels
from neural_net import NeuralNet

def calc_loss_of_nn(neural_net, learn_set, learn_expected):
    predicted = neural_net.predict(learn_set)
    return round(abs(np.mean(learn_expected - predicted)), 8)

def calc_performance_of_nn(neural_net, test_set, test_expected):
    test_predicted = neural_net.predict(test_set)
    total = len(test_predicted)
    correct = 0
    for i, predict_data in enumerate(test_predicted):
        correct += 1 if predict_data.argmax() == test_expected[i].argmax() else 0
    return round(correct / total, 4)

def collect_neural_net_data(num_layers, layer_sizes, teaching_iterations, learning_rate):
    global learn_set, learn_expected, test_data, test_expected
    neural_net = NeuralNet(num_layers, layer_sizes, teaching_iterations, learning_rate)
    neural_net.train(learn_set, learn_expected)
    loss = calc_loss_of_nn(neural_net, learn_set, learn_expected)
    performance = calc_performance_of_nn(neural_net, test_data, test_expected)
    print(num_layers, layer_sizes, teaching_iterations, learning_rate, loss, performance, sep="\t")
    return loss, performance

def plot(xs, y1s, y2s):
    fig, ax1 = plt.subplots()
    ax1.plot(xs, y1s, 'b-', label="Loss")
    ax1.set_xlabel('hidden layer size')
    ax1.set_ylabel('loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(xs, y2s, 'r-', label="Performance")
    ax2.set_ylabel('performance', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    learn_set, learn_expected, test_data, test_expected = setup_data(0.75)
    print('layers', 'layer sizes', 'iters', 'rate', 'loss\t', 'performance', sep="\t")
    trials = 5
    hidden_layer_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    losses = []
    performances = []
    for size in hidden_layer_sizes:
        total_loss, total_perf = 0, 0
        for i in range(trials):
            loss, performance = collect_neural_net_data(3, [4, size, 3], 10000, 0.05)
            total_loss += loss
            total_perf += performance
        losses.append(total_loss / trials)
        performances.append(total_perf / trials)
    plot(hidden_layer_sizes, losses, performances)
