import pandas as pd
from helperFunctions import *


learning_rate = .01

train = pd.read_csv("/Users/yazen/Desktop/datasets/PimaDiabetes/train.csv", index_col=0)
test = pd.read_csv("/Users/yazen/Desktop/datasets/PimaDiabetes/test.csv", index_col=0)

train_labels = train['label']
train_features = train.drop('label', axis=1)

train_labels = np.expand_dims(train_labels, 1)
x = tf.placeholder("float64",[None, 8])

#layer = create_network(x, [8, 20, 30, 40, 50, 20, 1], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.sigmoid])

layer = create_network(x, [8, 1],
                       [tf.nn.relu, tf.nn.relu])
y = tf.placeholder("float64", [None, 1])

cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=layer, labels=y)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iter in range(0, 50):
        _, c = sess.run([optimizer, cost], feed_dict={x: train_features, y: train_labels})
        print("Iteration " + str(iter) + " cost: " + str(c))
