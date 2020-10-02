# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../model/"]).decode("utf8"))

train_set = pd.read_csv("../input/fashion-mnist_train.csv")
test_set = pd.read_csv("../input/fashion-mnist_test.csv")


y_train = train_set["label"].as_matrix()
X_train = train_set.drop("label", axis=1).as_matrix()

y_test = test_set["label"].as_matrix()
X_test = test_set.drop("label", axis=1).as_matrix()

# plt.imshow(X[2].reshape(28,28))
# plt.show()

# Implementing LeNet5 with tensorflow
height = 28
width = 28
channels = 1
n_inputs = height * width
n_outputs = 10

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')

with tf.name_scope("Conv1"):
    conv1 = tf.layers.conv2d(X_reshaped, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)

with tf.name_scope("Conv2"):
    conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, 64 * 14 * 14])
    pool3_flat_drop = tf.layers.dropout(pool3_flat, 0.25, training=training)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat_drop, 128, activation=tf.nn.relu)
    fc1_drop = tf.layers.dropout(fc1, 0.5, training=training)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs)
    Y_proba = tf.nn.softmax(logits)

with tf.name_scope("train"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

def batch_generator(X, y, size=50, nb = 50):
    nb_instance, nb_features = X.shape
    p = np.random.permutation(nb_instance)
    X = X[p]
    y = y[p]
    for i in range(nb):
        a = np.random.choice(nb_instance, size)
        yield X[a], y[a]

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

n_epochs = 2
batch_size = 50

best_loss_val = np.infty
check_interval = 500
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        generator = batch_generator(X_train, y_train)
        iteration = 0
        for X_batch, y_batch in generator:
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: X_test,
                                                y: y_test})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
            iteration += 1
        acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Epoch {}, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
                  epoch, acc_val * 100, best_loss_val))
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

    if best_model_params:
        restore_model_params(best_model_params)
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final accuracy on test set:", acc_test)
    save_path = saver.save(sess, "./model/my_mnist_model")