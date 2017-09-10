import tensorflow as tf
import pandas as pd
from helperFunctions import *

train = pd.read_csv("/Users/yazen/Desktop/datasets/PimaDiabetes/train.csv")
test = pd.read_csv("/Users/yazen/Desktop/datasets/PimaDiabetes/test.csv")

train.columns = ["pregnancy", "plasma/glucose concentration", "blood pressure","tricep skin fold thickness", "serum insulin", "body mass index", "diabetes pedigree function", "age", "label"]
test.columns = ["pregnancy", "plasma/glucose concentration", "blood pressure","tricep skin fold thickness", "serum insulin", "body mass index", "diabetes pedigree function", "age", "label"]


x = tf.placeholder("float32",[None, 10])

layer = create_network(x,[10,20,30,40,50,20,1],[tf.nn.relu,tf.nn.relu,tf.nn.relu,tf.nn.relu,tf.nn.relu,tf.nn.relu,tf.nn.sigmoid)])

y = tf.placeholder("float32", [None, 1])
cost = tf.losses.log_loss(predictions=layer, labels=y)