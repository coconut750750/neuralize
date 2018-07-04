from optparse import OptionParser

from neuralize.core.neural_net import NeuralNet
from neuralize.core.sigmoid import SigmoidActivation
from neuralize.core.tanh import TanHActivation
from neuralize.data.mnist_data import setup_data

parser = OptionParser()
parser.add_option('-s', '--save', dest='save_file',
                  help='save new neural network to FILE', metavar='FILE')
parser.add_option('-l', '--load', dest='load_file',
                  help='load neural network from FILE', metavar='FILE')
parser.add_option('-t', '--teach', type='int', dest='more_iters', 
                  help='teach a loaded neural network more', metavar='INT')

def after_iteration(nn, iter):
    print("Learning Iteration: {}".format(iter))

def run_mnist_nn(save_file=None, load_file=None, more_iters=0):
    learn_set, learn_expected, test_set, test_expected = setup_data()
    inputs, outputs = len(learn_set[0]), len(learn_expected[0])

    if load_file:
        neural_net = NeuralNet.load(load_file)
        if more_iters:
            orig_iters = neural_net.teaching_iterations
            neural_net.teaching_iterations = more_iters
            neural_net.train(learn_set, learn_expected, after_iteration)
            neural_net.teaching_iterations = orig_iters + more_iters
    else:
        neural_net = NeuralNet(4, [inputs, 40, 30, outputs],
                               [SigmoidActivation(), SigmoidActivation(), SigmoidActivation()],
                               teaching_iterations=1000,
                               tau=50000, kappa=1.0, num_batches=10)
        neural_net.train(learn_set, learn_expected,
                         display_progress=True)    

    loss = neural_net.calculate_loss(learn_set, learn_expected)
    performance = neural_net.calculate_performance(test_set, test_expected)
    print('Loss: {}\tTest performance: {}'.format(loss, performance))
    print(neural_net)
    if save_file:
        neural_net.save(save_file)

if __name__ == '__main__':
    options = parser.parse_args()[0]
    run_mnist_nn(options.save_file, options.load_file, options.more_iters)
