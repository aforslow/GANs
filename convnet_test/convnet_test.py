import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('/MNIST_data', one_hot=True)
mnist = tf.reshape(mnist, [-1, 28, 28, 1])

# Define the network
x = tf.placeholder([-1, 28, 28, 1])

conv1 = tf.nn.conv2d(x, 32, 5, activation=tf.nn.relu)
conv2 = tf.nn.conv2d(x, 64, 3, activation=tf.nn.relu)

# Flatten data to connect with fully connected layer
fc1 = tf.contrib.layers.flatten(conv2)
fc1 = tf.layers.dense(fc1, 1024)
fc1 = tf.layers.dropout(fc1, rate=0.75, training=True)

out = tf.layers.dense(fc1, 10)
