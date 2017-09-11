import pandas as pd

import tensorflow as tf
from helperFunctions import *

input_features = 8
n_hidden_units1 = 10
n_hidden_units2 = 14
n_hidden_units3 = 12
n_hidden_units4 = 1

rate = .001

weights = dict(
            w1=tf.Variable(tf.random_normal([input_features, n_hidden_units1])),
            w2=tf.Variable(tf.random_normal([n_hidden_units1, n_hidden_units2])),
            w3=tf.Variable(tf.random_normal([n_hidden_units2, n_hidden_units3])),
            w4=tf.Variable(tf.random_normal([n_hidden_units3, n_hidden_units4]))
            )

biases = dict(
            b1=tf.Variable(tf.zeros([n_hidden_units1])),
            b2=tf.Variable(tf.zeros([n_hidden_units2])),
            b3=tf.Variable(tf.zeros([n_hidden_units3])),
            b4=tf.Variable(tf.zeros([n_hidden_units4]))
            )

train_features, train_labels = format_data("/Users/yazen/Desktop/datasets/PimaDiabetes/train.csv")
test_features, test_labels = format_data("/Users/yazen/Desktop/datasets/PimaDiabetes/test.csv")

x = tf.placeholder("float32", [None, 8])
y = tf.placeholder("float32", [None, 1])

layer = create_layer(x, weights['w1'], biases['b1'], tf.nn.relu)
layer = create_layer(layer, weights['w2'], biases['b2'], tf.nn.tanh)
layer = create_layer(layer, weights['w3'], biases['b3'], tf.nn.relu)
Z4 = create_layer(layer, weights['w4'], biases['b4'],tf.nn.sigmoid)

cost = cost_compute(Z4, y)
optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(1, 1000):
        _, c = sess.run([optimizer, cost], feed_dict={x: train_features, y: train_labels})
        print("Iteration " + str(iteration) + " cost: " + str(c))

    prediction = tf.round(Z4)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), "float"))

    prediction = sess.run(prediction, feed_dict={x: train_features, y: train_labels})

    print(np.append(prediction, train_labels, 1))

    print(accuracy.eval({x: train_features, y: train_labels}))
    print(accuracy.eval({x: test_features, y: test_labels}))


