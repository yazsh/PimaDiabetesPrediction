import pandas as pd
from helperFunctions import *
import tensorflow as tf

learning_rate = .001

train_features, train_labels = format_data("/Users/yazen/Desktop/datasets/PimaDiabetes/train.csv")
test_features, test_labels = format_data("/Users/yazen/Desktop/datasets/PimaDiabetes/test.csv")

x = tf.placeholder("float64", [None, 8])
y = tf.placeholder("float64", [None, 1])

layer = create_network(x, [8, 100, 1000, 500, 1],
                       [tf.nn.relu, tf.nn.tanh, tf.nn.relu, tf.nn.sigmoid])


cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=layer, multi_class_labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iterator in range(0, 500):
        _, c = sess.run([optimizer, cost], feed_dict={x: train_features, y: train_labels})
        print("Iteration " + str(iterator) + " cost: " + str(c))

    prediction = tf.round(layer)
    train_predictions = sess.run(prediction, feed_dict={x:train_features, y:train_labels})
    test_predictions = sess.run(prediction, feed_dict={x:test_features, y:test_labels})

    train_equality_array = tf.equal(prediction, train_labels)
    test_equality_array = tf.equal(prediction, test_labels)

    train_accuracy = sess.run(tf.reduce_mean(tf.cast(train_equality_array, "float")),
                              feed_dict={x: train_features, y: train_labels})
    test_accuracy = sess.run(tf.reduce_mean(tf.cast(test_equality_array, "float")),
                             feed_dict={x: test_features, y: test_labels})


    comparison = np.append(train_predictions,train_labels, axis=1)
    print(comparison[:20])
    print("train accuracy " + str(train_accuracy))
    print("test accuracy " + str(test_accuracy))
