# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Feature selection
train.plot(kind='scatter', x='x', y='y')
features_cols = ['x']
predict_cols = ['y']

# Nulls manipulation
train = train.dropna()

#Normalization
train_dfx = (train[features_cols] - train[features_cols].mean())/train[features_cols].std()
train_dfy = train[predict_cols]
test_dfx = (test[features_cols] - train[features_cols].mean())/train[features_cols].std()
test_dfy = test[predict_cols]

learning_rate = 0.001
training_epochs = 10000
batch_size = 500
display = 1000
seed = 200
(hidden1_size, hidden2_size) = (100, 50)

X = tf.placeholder(tf.float32, shape=[None,train_dfx.shape[1]])
Y = tf.placeholder(tf.float32, shape=[None, train_dfy.shape[1]])
training_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(500).repeat().batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().batch(test_dfx.shape[0])
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,training_dataset.output_shapes)
dx, dy = iterator.get_next()
training_init_op = iterator.make_initializer(training_dataset)
test_init_op = iterator.make_initializer(test_dataset)
with tf.Session() as sess:
    W1 = tf.Variable(tf.random_normal([train_dfx.shape[1], hidden1_size], seed = seed))
    b1 = tf.Variable(tf.random_normal([hidden1_size], seed = seed))
    z1 = tf.nn.relu(tf.add(tf.matmul(dx,W1), b1))
    W2 = tf.Variable(tf.random_normal([hidden1_size, hidden2_size], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([hidden2_size], seed = seed))
    z2 = tf.nn.relu(tf.add(tf.matmul(z1,W2), b2))
    W3 = tf.Variable(tf.random_normal([hidden2_size, train_dfy.shape[1]], stddev=0.1))
    b3 = tf.Variable(tf.random_normal([train_dfy.shape[1]], seed = seed))
    h = tf.add(tf.matmul(z2,W3), b3)                                    
    loss = tf.reduce_mean(tf.pow(h - dy, 2)) + 0.1 * tf.nn.l2_loss(W3)
    update = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(training_init_op, feed_dict={X: train_dfx.values, Y: train_dfy.values})
    for epoch in range(training_epochs):
        sess.run(update)
        if (epoch + 1) % display == 0:
            print("iter: {}, loss: {:.3f}".format(epoch + 1, sess.run(loss)))            
print("Training Finished!")
print("Train MSE:", sess.run(loss))
sess.run(test_init_op, feed_dict={X: test_dfx.values, Y: test_dfy.values})
print("Test MSE:", sess.run(loss))