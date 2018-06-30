from neuralize.data.fruit_data import setup_data
from neuralize.vis.visualize import plot_2lines
from neuralize.core.neural_net import NeuralNet

from main import calc_performance_of_nn

def collect_neural_net_data(num_layers, layer_sizes, teaching_iterations, learning_rate):
    global learn_set, learn_expected, test_set, test_expected
    neural_net = NeuralNet(num_layers, layer_sizes, teaching_iterations, learning_rate)
    neural_net.train(learn_set, learn_expected)
    loss = neural_net.calculate_loss(learn_set, learn_expected)
    performance = calc_performance_of_nn(neural_net, test_set, test_expected)
    print(num_layers, layer_sizes, teaching_iterations, learning_rate, loss, performance, sep="\t")
    return loss, performance

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

def compare_num_layers(num_inputs, num_outputs, trials=5):
    avg_nodes = (num_inputs + num_outputs) // 2
    num_layers = [2, 3, 4, 5, 6]
    args = []
    for num in num_layers:
        args.append((num, [num_inputs] + [avg_nodes] * (num - 2) + [num_outputs], 10000, 0.05))
    losses, performances = run_trials(args, trials=trials)
    plot_2lines(num_layers, losses, performances, x_label='number of layers')

def compare_layer_sizes(num_inputs, num_outputs, trials=5):
    hidden_layer_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    args = []
    for size in hidden_layer_sizes:
        args.append((3, [num_inputs, size, num_outputs], 10000, 0.05))
    losses, performances = run_trials(args, trials=trials)
    plot_2lines(hidden_layer_sizes, losses, performances, x_label='hidden layer size')

def compare_learning_iterations(num_inputs, num_outputs, trials=5):
    avg_nodes = (num_inputs + num_outputs) // 2
    learning_iterations = [1000, 2500, 5000, 7500, 10000]
    args = []
    for iters in learning_iterations:
        args.append((3, [num_inputs, avg_nodes, num_outputs], iters, 0.05))
    losses, performances = run_trials(args, trials=trials)
    plot_2lines(learning_iterations, losses, performances, x_label='learning iterations')

def compare_learning_rate(num_inputs, num_outputs, trials=5):
    avg_nodes = (num_inputs + num_outputs) // 2
    learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75]
    args = []
    for rate in learning_rate:
        args.append((3, [num_inputs, avg_nodes, num_outputs], 10000, rate))
    losses, performances = run_trials(args, trials=trials)
    plot_2lines(learning_rate, losses, performances, x_label='learning rate')

if __name__ == '__main__':
    learn_set, learn_expected, test_set, test_expected = setup_data(0.75)
    print('layers', 'layer sizes', 'iters', 'rate', 'loss\t', 'performance', sep="\t")
    compare_num_layers(len(learn_set[0]), len(learn_expected[0]), 5)
