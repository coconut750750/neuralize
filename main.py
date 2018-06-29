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
    return round(correct / total * 100, 2)

def collect_neural_net_data(num_layers, layer_sizes, teaching_iterations, learning_rate):
    global learn_set, learn_expected, test_data, test_expected
    neural_net = NeuralNet(num_layers, layer_sizes, teaching_iterations, learning_rate)
    neural_net.train(learn_set, learn_expected)
    loss = calc_loss_of_nn(neural_net, learn_set, learn_expected)
    performance = calc_performance_of_nn(neural_net, test_data, test_expected)
    print(num_layers, layer_sizes, teaching_iterations, learning_rate, loss, performance, sep="\t")
    return loss, performance

def plot(xs, y1s, y2s, x_label=""):
    fig, ax1 = plt.subplots()
    ax1.plot(xs, y1s, 'b-', label="Loss")
    ax1.set_xlabel(x_label)
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

def run_trials(list_of_args, trials=5):
    losses = []
    performances = []
    for args in list_of_args:
        total_loss, total_perf = 0, 0
        for i in range(trials):
            loss, performance = collect_neural_net_data(*args)
            total_loss += loss
            total_perf += performance
        losses.append(total_loss / trials)
        performances.append(total_perf / trials)
    return losses, performances

def compare_num_layers(trials=5):
    num_layers = [2, 3, 4, 5, 6]
    args = []
    for num in num_layers:
        args.append((num, [4] + [3] * (num - 2) + [3], 10000, 0.05))
    losses, performances = run_trials(args, trials=trials)
    plot(num_layers, losses, performances, x_label='number of layers')

def compare_layer_sizes(trials=5):
    hidden_layer_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    args = []
    for size in hidden_layer_sizes:
        args.append((3, [4, size, 3], 10000, 0.05))
    losses, performances = run_trials(args, trials=trials)
    plot(hidden_layer_sizes, losses, performances, x_label='hidden layer size')

def compare_learning_iterations(trials=5):
    learning_iterations = [1000, 2500, 5000, 7500, 10000]
    args = []
    for iters in learning_iterations:
        args.append((3, [4, 3, 3], iters, 0.05))
    losses, performances = run_trials(args, trials=trials)
    plot(learning_iterations, losses, performances, x_label='learning iterations')

def compare_learning_rate(trials=5):
    learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75]
    args = []
    for rate in learning_rate:
        args.append((3, [4, 3, 3], 10000, rate))
    losses, performances = run_trials(args, trials=trials)
    plot(learning_rate, losses, performances, x_label='learning rate')

if __name__ == '__main__':
    learn_set, learn_expected, test_data, test_expected = setup_data(0.75)
    print('layers', 'layer sizes', 'iters', 'rate', 'loss\t', 'performance', sep="\t")
    compare_num_layers(5)
