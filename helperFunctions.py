import tensorflow as tf
import numpy as np


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
