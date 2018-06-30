import numpy as np
from random import shuffle

def setup_data(all_data, network_outputs, learning_percent):
    shuffle(all_data)
    learning_size = int(len(all_data) * learning_percent)
    labels = np.array([network_outputs[int(i[0]) - 1] for i in all_data])
    
    features = np.array([i[1:] for i in all_data])
    features = (features - features.min(axis=0)) / features.max(axis=0)

    learn_set = np.array(features[0: learning_size])
    learn_expected = np.array(labels[0: learning_size])

    test_set = np.array(features[learning_size:])
    test_expected = np.array(labels[learning_size:])

    return learn_set, learn_expected, test_set, test_expected
