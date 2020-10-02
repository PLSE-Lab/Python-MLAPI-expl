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
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

batch_size = 128
test_size = 100
epsilon = 1e-3

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        
    plt.show()
    return fig

def init_weights(shape,var_name):
    return tf.Variable(tf.random_normal(shape,stddev=0.01, name=var_name))

def convert_to_mat(X_mb):
    select_batch = np.zeros([len(X_mb),784])
    for i in range(len(X_mb)):
        select_batch[i,:] = np.asmatrix(X_mb[i])

    return select_batch

def batch_norm_wrapper(inputs, is_training, name = "batch", decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon,name=name)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon,name=name)


def model(X, w, w2, w3, w4, w_o, is_training, p_keep_hidden):
    l1i = tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME',name="l1")
    l1bn = batch_norm_wrapper(l1i,is_training,name="l1a")
    #l1a = tf.nn.relu(l1bn,name="l1b")
    l1a = tf.nn.relu(l1i,name="l1b")
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME',name="l1c")

    l2i = tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME',name="l2")
    l2bn = batch_norm_wrapper(l2i,is_training,name="l2a")
    l2a = tf.nn.relu(l2i,name="l2b")
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME',name="l2c")

    l3i = tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME',name="l3")
    l3bn = batch_norm_wrapper(l3i,is_training,name="l3a")
    l3a = tf.nn.relu(l3i,name="l3b")
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME',name="l3c")
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)

    l4 = tf.nn.relu(tf.matmul(l3, w4),name="l4a")
    l4 = tf.nn.dropout(l4, p_keep_hidden,name="l4a")

    pyx = tf.matmul(l4, w_o,name="l4b")
    return pyx

def momentum_optimizer(loss):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.0003,                # Base learning rate.
        batch,  # Current index into the dataset.
        100 // 4,          # Decay step - this decays 4 times throughout training process.
        0.95,                # Decay rate.
        staircase=True)
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=batch,var_list=var_list)
    optimizer=tf.train.AdamOptimizer(learning_rate,0.95).minimize(loss,global_step=batch)
    return optimizer

def define_mnist(X_mb, Y_mb, num, total_img):
    Bd_matlab = []
    for i in range(total_img):
        tempy = Y_mb[i,:]
        if np.argmax(tempy) == num:
            tempx = X_mb[i,:]
            Bd_matlab.append(tempx)

    X_feed = convert_to_mat(Bd_matlab)
    return X_feed

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

mnist_tr = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
mnist_te = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

with tf.variable_scope("classifier"):
    X = tf.placeholder("float", [None, 28, 28, 1], name="X")
    Y = tf.placeholder("float", [None, 10], name="Y")

    w = init_weights([3, 3, 1, 32], "W1")       # 3x3x1 conv, 32 outputs
    w2 = init_weights([3, 3, 32, 64], "W2")     # 3x3x32 conv, 64 outputs
    w3 = init_weights([3, 3, 64, 128], "W3")    # 3x3x32 conv, 128 outputs
    w4 = init_weights([128 * 4 * 4, 625], "W4") # FC 128 * 4 * 4 inputs, 625 outputs
    w_o = init_weights([625, 10], "W5")         # FC 625 inputs, 10 outputs (labels)

    p_keep_hidden = tf.placeholder("float", name="keep")
    py_x = model(X, w, w2, w3, w4, w_o, True, p_keep_hidden)

    correct_pred = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y))
    #cost = tf.nn.softmax(logits=py_x)
    train_op = momentum_optimizer(cost)
    predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    #saver = tf.train.import_meta_graph("Classification_Model/Model-D.ckpt.meta")
    #saver.restore(sess, './Classification_Model/Model-D.ckpt')

    #saver = tf.train.Saver()
    test_indices = np.arange(len(teX)) # Get A Test Batch
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:test_size]

    print(np.mean(np.argmax(teY[test_indices], axis=1) ==
                        sess.run(predict_op, feed_dict={X: np.reshape(teX[test_indices], [-1, 28, 28, 1]),
                                                        p_keep_hidden: 1.0})))

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: np.reshape(trX[start:end],[-1,28,28,1]), Y: trY[start:end], p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: np.reshape(teX[test_indices], [-1, 28, 28, 1]),
                                                         p_keep_hidden: 1.0})))
        cost_alt = tf.nn.softmax(logits=py_x)

        #print("O: {} ".format(sess.run(cost_alt, feed_dict={X: mnist_te[test_indices],
        #                                                 p_keep_hidden: 1.0})))

        #plot(mnist_te[test_indices])