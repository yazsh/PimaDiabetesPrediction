import tensorflow as tf
import numpy as np
import pandas as pd

def create_bias(hidden_units):
    return tf.Variable(np.zeros(hidden_units))


def create_weight(old_features, new_features):
    return tf.Variable(np.random.randn(old_features, new_features))


def create_layer(prev_layer, weight, bias, activation_function=None):
    z = tf.matmul(prev_layer, weight)
    z = z + bias

    if activation_function is not None:
        z = activation_function(z)

    return z


def create_network(input_features, units, activation_functions):
    layer = create_layer(input_features, create_weight(units[0], units[1]), create_bias(units[1]),
                         activation_functions[0])

    for x in range(2, len(units)):
        layer = create_layer(layer, create_weight(units[x - 1], units[x]), create_bias(units[x]),
                             activation_functions[x - 1])

    return layer


def format_data(file_path):
    train = pd.read_csv(file_path, index_col=0)
    labels = train['label']
    labels = np.expand_dims(labels, 1)
    features = train.drop('label', axis=1)
    return features, labels
