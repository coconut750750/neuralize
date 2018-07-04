from neuralize.core.neural_net import NeuralNet
from neuralize.core.sigmoid import SigmoidActivation
from neuralize.data.fruit_data import setup_data
from neuralize.vis.graph import plot_2lines

def collect_neural_net_data(num_layers, layer_sizes, teaching_iterations, tau):
    global learn_set, learn_expected, test_set, test_expected
    neural_net = NeuralNet(num_layers, layer_sizes, [SigmoidActivation()] * num_layers, teaching_iterations, tau)
    neural_net.train(learn_set, learn_expected)
    loss = neural_net.calculate_loss(learn_set, learn_expected)
    performance = neural_net.calculate_performance(test_set, test_expected)
    print(num_layers, layer_sizes, teaching_iterations, tau, loss, performance, sep="\t")
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

def compare_tau(num_inputs, num_outputs, trials=5):
    avg_nodes = (num_inputs + num_outputs) // 2
    tau = [5, 10, 20, 50, 100, 150]
    args = []
    for rate in tau:
        args.append((3, [num_inputs, avg_nodes, num_outputs], 10000, rate))
    losses, performances = run_trials(args, trials=trials)
    plot_2lines(tau, losses, performances, x_label='learning rate')

if __name__ == '__main__':
    learn_set, learn_expected, test_set, test_expected = setup_data(0.75)
    print('layers', 'layer sizes', 'iters', 'tau ', 'loss\t', 'performance', sep="\t")
    compare_layer_sizes(len(learn_set[0]), len(learn_expected[0]), 5)
