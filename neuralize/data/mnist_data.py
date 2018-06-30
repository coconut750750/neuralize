import pickle, gzip
import numpy as np

network_outputs = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

def setup_data():
    with gzip.open('res/mnist.pkl.gz', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        learn_set, valid_set, test_set = u.load()

    learn_inputs = learn_set[0]
    learn_labels = np.array([network_outputs[i] for i in learn_set[1]])

    test_inputs = test_set[0]
    test_labels = np.array([network_outputs[i] for i in test_set[1]])
    return learn_inputs, learn_labels, test_inputs, test_labels

def data_to_str(data):
    entire_str = ""
    for i in range(0, 784, 28):
        row_str = ""
        for num in data[i: i + 28]:
            row_str += '{:<3} '.format(str(num)[:3])
        entire_str += '{}\n'.format(row_str)
    return entire_str
