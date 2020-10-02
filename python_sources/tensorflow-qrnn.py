#!/usr/bin/env python
# coding: utf-8

# This kernel illustrates a basic Tensorflow implementation of the [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576).
# 
# The original implementation from the authors is based on [PyTorch](https://github.com/salesforce/pytorch-qrnn) but was challenging to run on GPU kernel (because it requires some libraries like cupy).
# 
# The code below has been based on [this](https://github.com/icoxfog417/tensorflow_qrnn/) implementation.
# 
# The dataset used here is the UCI ML hand-written digits datasets from scikit-learn. 

# In[ ]:


import tensorflow as tf

class QRNN():

    def __init__(self, in_size, size, conv_size=2):
        self.kernel = None
        self.batch_size = -1
        self.conv_size = conv_size
        self.c = None
        self.h = None
        self._x = None
        if conv_size == 1:
            self.kernel = QRNNLinear(in_size, size)
        elif conv_size == 2:
            self.kernel = QRNNWithPrevious(in_size, size)
        else:
            self.kernel = QRNNConvolution(in_size, size, conv_size)

    def _step(self, f, z, o):
        with tf.variable_scope("fo-Pool"):
            # f,z,o is batch_size x size
            f = tf.sigmoid(f)
            z = tf.tanh(z)
            o = tf.sigmoid(o)
            self.c = tf.multiply(f, self.c) + tf.multiply(1 - f, z)
            self.h = tf.multiply(o, self.c)  # h is size vector

        return self.h

    def forward(self, x):
        length = lambda mx: int(mx.get_shape()[0])

        with tf.variable_scope("QRNN/Forward"):
            if self.c is None:
                # init context cell
                self.c = tf.zeros([length(x), self.kernel.size], dtype=tf.float32)

            if self.conv_size <= 2:
                # x is batch_size x sentence_length x word_length
                # -> now, transpose it to sentence_length x batch_size x word_length
                _x = tf.transpose(x, [1, 0, 2])

                for i in range(length(_x)):
                    t = _x[i] # t is batch_size x word_length matrix
                    f, z, o = self.kernel.forward(t)
                    self._step(f, z, o)
            else:
                c_f, c_z, c_o = self.kernel.conv(x)
                for i in range(length(c_f)):
                    f, z, o = c_f[i], c_z[i], c_o[i]
                    self._step(f, z, o)

        return self.h


class QRNNLinear():

    def __init__(self, in_size, size):
        self.in_size = in_size
        self.size = size
        self._weight_size = self.size * 3  # z, f, o
        with tf.variable_scope("QRNN/Variable/Linear"):
            initializer = tf.random_normal_initializer()
            self.W = tf.get_variable("W", [self.in_size, self._weight_size], initializer=initializer)
            self.b = tf.get_variable("b", [self._weight_size], initializer=initializer)

    def forward(self, t):
        # x is batch_size x word_length matrix
        _weighted = tf.matmul(t, self.W)
        _weighted = tf.add(_weighted, self.b)

        # now, _weighted is batch_size x weight_size
        f, z, o = tf.split(_weighted, num_or_size_splits=3, axis=1)  # split to f, z, o. each matrix is batch_size x size
        return f, z, o


class QRNNWithPrevious():

    def __init__(self, in_size, size):
        self.in_size = in_size
        self.size = size
        self._weight_size = self.size * 3  # z, f, o
        self._previous = None
        with tf.variable_scope("QRNN/Variable/WithPrevious"):
            initializer = tf.random_normal_initializer()
            self.W = tf.get_variable("W", [self.in_size, self._weight_size], initializer=initializer)
            self.V = tf.get_variable("V", [self.in_size, self._weight_size], initializer=initializer)
            self.b = tf.get_variable("b", [self._weight_size], initializer=initializer)

    def forward(self, t):
        if self._previous is None:
            self._previous = tf.get_variable("previous", [t.get_shape()[0], self.in_size], initializer=tf.random_normal_initializer())

        _current = tf.matmul(t, self.W)
        _previous = tf.matmul(self._previous, self.V)
        _previous = tf.add(_previous, self.b)
        _weighted = tf.add(_current, _previous)

        f, z, o = tf.split(_weighted, num_or_size_splits=3, axis=1)  # split to f, z, o. each matrix is batch_size x size
        self._previous = t
        return f, z, o


class QRNNConvolution():

    def __init__(self, in_size, size, conv_size):
        self.in_size = in_size
        self.size = size
        self.conv_size = conv_size
        self._weight_size = self.size * 3  # z, f, o

        with tf.variable_scope("QRNN/Variable/Convolution"):
            initializer = tf.random_normal_initializer()
            self.conv_filter = tf.get_variable("conv_filter", [conv_size, in_size, self._weight_size], initializer=initializer)

    def conv(self, x):
        # !! x is batch_size x sentence_length x word_length(=channel) !!
        _weighted = tf.nn.conv1d(x, self.conv_filter, stride=1, padding="SAME", data_format="NWC")

        # _weighted is batch_size x conved_size x output_channel
        _w = tf.transpose(_weighted, [1, 0, 2])  # conved_size x  batch_size x output_channel
        _ws = tf.split(_w, num_or_size_splits=3, axis=2) # make 3(f, z, o) conved_size x  batch_size x size
        return _ws


# In[ ]:


import os
import unittest
import time
import functools
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow import nn
from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def qrnn_forward(X, size, n_class, batch_size, conv_size):
    in_size = int(X.get_shape()[2])

    qrnn = QRNN(in_size=in_size, size=size, conv_size=conv_size)
    hidden = qrnn.forward(X)

    with tf.name_scope("QRNN-Classifier"):
        W = tf.Variable(tf.random_normal([size, n_class]), name="W")
        b = tf.Variable(tf.random_normal([n_class]), name="b")
        output = tf.add(tf.matmul(hidden, W), b)

        return output
 
def baseline_forward(X, size, n_class):
        shape = X.get_shape()
        seq = tf.transpose(X, [1, 0, 2]) 

        with tf.name_scope("LSTM"):
            lstm_cell = LSTMCell(size, forget_bias=1.0)
            outputs, states = nn.dynamic_rnn(time_major=True, cell=lstm_cell, inputs=seq, dtype=tf.float32)

        with tf.name_scope("LSTM-Classifier"):
            W = tf.Variable(tf.random_normal([size, n_class]), name="W")
            b = tf.Variable(tf.random_normal([n_class]), name="b")
            output = tf.matmul(outputs[-1], W) + b

        return output

def check_by_digits(graph, qrnn=-1, baseline=False, random=False):
    digits = load_digits()
    horizon, vertical, n_class = (8, 8, 10)  # 8 x 8 image, 0~9 number(=10 class)
    size = 128  # state vector size
    batch_size = 128
    images = digits.images / np.max(digits.images)  # simple normalization
    target = np.array([[1 if t == i else 0 for i in range(n_class)] for t in digits.target])  # to 1 hot vector
    learning_rate = 0.001
    train_iter = 1000
    summary_dir = os.path.dirname("./summary")

    with tf.name_scope("placeholder"):
        X = tf.placeholder(tf.float32, [batch_size, vertical, horizon])
        y = tf.placeholder(tf.float32, [batch_size, n_class])

    if qrnn > 0:
        pred = qrnn_forward(X, size, n_class, batch_size, conv_size=qrnn)
        summary_dir += "/qrnn"
    elif baseline:
        pred = baseline_forward(X, size, n_class)
        summary_dir += "/lstm"
    else:
        pred = random_forward(X, size, n_class)            
        summary_dir += "/random"
        
    with tf.name_scope("optimization"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.name_scope("evaluation"):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope("summary"):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(summary_dir, graph)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_iter):
            indices = np.random.randint(len(digits.target) - batch_size, size=batch_size)
            _X = images[indices]
            _y = target[indices]
            sess.run(optimizer, feed_dict={X: _X, y: _y})

            if i % 100 == 0:
                _loss, _accuracy, _merged = sess.run([loss, accuracy, merged], feed_dict={X: _X, y: _y})
                writer.add_summary(_merged, i)
                print("Iter {}: loss={}, accuracy={}".format(i, _loss, _accuracy))
            
        with tf.name_scope("test-evaluation"):
            acc = sess.run(accuracy, feed_dict={X: images[-batch_size:], y: target[-batch_size:]})
            print("Testset Accuracy={}".format(acc))
    

with timer("QRNN"):
    with tf.Graph().as_default() as qrnn:
        check_by_digits(qrnn, qrnn=5)
    
with timer("LSTM"):
    with tf.Graph().as_default() as ltsm:
        check_by_digits(ltsm, baseline=True)
        


# In[ ]:




