import pandas as pd
from helperFunctions import *

train = pd.read_csv("/Users/yazen/Desktop/datasets/PimaDiabetes/train.csv", index_col=0)
test = pd.read_csv("/Users/yazen/Desktop/datasets/PimaDiabetes/test.csv", index_col=0)


x = tf.placeholder("float64",[None, 10])

layer = create_network(x, [10, 20, 30, 40, 50, 20, 1],
                       [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.sigmoid])

y = tf.placeholder("float32", [None, 1])
cost = tf.losses.log_loss(predictions=layer, labels=y)

