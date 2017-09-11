import tensorflow as tf
import numpy as np
import pandas as pd


def cost_compute(prediction, correct_values):
    return tf.losses.sigmoid_cross_entropy(logits=prediction, multi_class_labels=correct_values)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = correct_values))


def create_layer(prev_layer, weight, bias, activation_function=None):
    z = tf.matmul(prev_layer, weight)
    z = tf.add(z, bias)

    if activation_function is not None:
        z = activation_function(z)

    return z


def format_data(file_path):
    train = pd.read_csv(file_path, index_col=0)
    labels = train['label']
    labels = np.expand_dims(labels, 1)
    features = train.drop('label', axis=1)
    return features, labels
