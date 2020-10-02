#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2,w_3):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))# The \sigma function
    h2   = tf.nn.sigmoid(tf.matmul(h,w_2))
    yhat = tf.matmul(h2, w_3)  # The \varphi function
    return yhat

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]
    print(target)
    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():
    train_x = pd.read_csv("../input/cancerinhibitors/cdk2_train.csv",dtype=np.float32)
    test_x = pd.read_csv("../input/cancerinhibitors/cdk2_test.csv",dtype=np.float32)
    train_y = pd.read_csv("../input/cancerinhibitors/cdk2_trainlabels.csv",dtype=np.float32)
    test_y = pd.read_csv("../input/cancerinhibitors/cdk_2testlabels.csv",dtype=np.float32)


    train_x=train_x.values
    test_x=test_x.values
    train_y=train_y.values
    test_y=test_y.values
   
    
    nb_classes = 2
    targets = train_y.reshape(-1)
    targets=targets.astype(int,casting='unsafe')
    train_y = np.eye(nb_classes)[targets]
    print(test_x.shape)
    targets = test_y.reshape(-1)
    targets=targets.astype(int,casting='unsafe')
    test_y = np.eye(nb_classes)[targets]
    print(test_y.shape)
    #print(test_y)
   
    # Layer's sizes
    x_size = train_x.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 1000               # Number of hidden nodes
    h2_size = 500
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float32", shape=[None, x_size])
    y = tf.placeholder("float32", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, h2_size))
    w_3 = init_weights((h2_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2 , w_3)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(10):
        # Train with each example
        for i in range(len(train_x)):
            sess.run(updates, feed_dict={X: train_x[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_x, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_x, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()
if __name__ == '__main__':
    main()

