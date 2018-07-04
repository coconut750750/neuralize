import sys
from PyQt5.QtWidgets import QApplication

from neuralize.core.neural_net import NeuralNet
from neuralize.core.sigmoid import SigmoidActivation
from neuralize.data.iris_data import setup_data
from neuralize.vis.visualizer import NeuralizeMainWindow

def after_iteration(nn, iter):
    pass
    # print("Learning Iteration: {}".format(iter))

def run_iris_nn():
    learn_set, learn_expected, test_set, test_expected = setup_data()
    inputs, outputs = len(learn_set[0]), len(learn_expected[0])

    neural_net = NeuralNet(3, [inputs, 5, outputs],
                           [SigmoidActivation(), SigmoidActivation()],
                           teaching_iterations=10000,
                           tau=100, kappa=0.5)
    

    neural_net.train(learn_set, learn_expected, after_iteration)    

    loss = neural_net.calculate_loss(learn_set, learn_expected)
    performance = neural_net.calculate_performance(test_set, test_expected)
    print('Loss: {}\tTest performance: {}'.format(loss, performance))
    print(neural_net)
    return neural_net

if __name__ == '__main__':
    app = QApplication(sys.argv)
    neural_net = run_iris_nn()
    main_window = NeuralizeMainWindow(neural_net)
    sys.exit(app.exec_())
