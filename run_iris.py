import sys
from PyQt5.QtWidgets import QApplication

from neuralize.vis.ui_neural_net import UINeuralNet
from neuralize.core.sigmoid import SigmoidActivation
from neuralize.data.iris_data import setup_data
from neuralize.vis.visualizer import NeuralizeMainWindow

def after_iteration(nn, iter, activations):
    print("Learning Iteration: {}".format(iter))

def get_ui_neural_net(inputs, outputs):
    neural_net = UINeuralNet(3, [inputs, 7, outputs],
                             [SigmoidActivation(), SigmoidActivation()],
                             teaching_iterations=1000,
                             tau=100, kappa=0.5)
    
    return neural_net

if __name__ == '__main__':
    learn_set, learn_expected, test_set, test_expected = setup_data()
    inputs, outputs = len(learn_set[0]), len(learn_expected[0])

    app = QApplication(sys.argv)
    neural_net = get_ui_neural_net(inputs, outputs)
    main_window = NeuralizeMainWindow(neural_net, learn_set, learn_expected)
    sys.exit(app.exec_())
