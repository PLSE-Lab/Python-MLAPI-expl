# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

from subprocess import check_output

import csv
import tensorflow as tf
import random
import pandas as pd
import numpy as np
from time import time

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

from tensorflow.contrib import layers
from tensorflow.contrib import learn

seed = 7
np.random.seed(seed)

df = pd.read_csv('../input/diabetes.csv')

X = df[
    ["Pregnancies", 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

y = df[['Outcome']]

df['NotDiabetes'] = 1 - df['Outcome']
y = df[['Outcome', 'NotDiabetes']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

learning_rate = tf.train.exponential_decay(learning_rate=0.01,
                                           global_step=1,
                                           decay_steps=X_train.shape[0],
                                           decay_rate=0.95,
                                           staircase=True)

# Parameters

training_epochs = 150
batch_size = 10
display_step = 10

# Neural Network Parameters

n_hidden_1 = 12
n_hidden_2 = 8
n_hidden_3 = 1
n_input = X_train.shape[1]
n_classes = y_train.shape[1]
dropout = 0.5

# TensorFlow Graph input

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float")
keep_prob = tf.placeholder(tf.float32)

# Create NN model

def neural_network(x, weights, biases, dropout):
    # Hidden layer with relu activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with relu activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

    #out_layer = tf.nn.dropout(out_layer, dropout)
    return out_layer

# Layers weight and bias

weights = {
    'h1': tf.Variable(tf.random_uniform(shape=(n_input,n_hidden_1),minval=0, maxval=0.005, dtype=tf.float32, seed=0)),
    'h2': tf.Variable(tf.random_uniform(shape=(n_hidden_1, n_hidden_2),minval=0, maxval=0.005, dtype=tf.float32, seed=0)),
    'h3': tf.Variable(tf.random_uniform(shape=(n_hidden_2, n_hidden_3),minval=0, maxval=0.005, dtype=tf.float32, seed=0)),
    'out': tf.Variable(tf.random_uniform(shape=(n_hidden_3, n_classes),minval=0, maxval=0.005, dtype=tf.float32, seed=0))
}

biases = {
    'b1': tf.Variable(tf.random_uniform([n_hidden_1])),
    'b2': tf.Variable(tf.random_uniform([n_hidden_2])),
    'b3': tf.Variable(tf.random_uniform([n_hidden_3])),
    'out': tf.Variable(tf.random_uniform([n_classes]))
}

# Constructing model

pred = neural_network(x, weights, biases, keep_prob)

# Defining loss and optimizer

cost = tf.nn.l2_loss(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables

init = tf.global_variables_initializer()

# Running first session

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train) / batch_size)

        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(y_train, total_batch)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization operation (backprop) and cost operation(to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("epoch:", '%d' % (epoch + 1), "cost=", "{:.4f}".format(avg_cost))

    # Test model

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test, keep_prob: 1}))