# from tensorflow.examples.tutorials.mnist import input_data
# %matplotlib inline
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder



# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#### DATA PREPARATION

dataFolder = "../input"

train = pd.read_csv(dataFolder + "/train.csv")
test = pd.read_csv(dataFolder + "/test.csv")


# Pop the label coloumn from the dataFrame
labels = np.array(train.pop('label'))
# Convert the labels [1,2,3] -> [0,1,2] and reshape from (n,) -> (n,1)
labels = LabelEncoder().fit_transform(labels)[:, None]
# Convert the labels to one-hot encoding and transform the returned sparse
# matrix to dense
labels = OneHotEncoder().fit_transform(labels).todense()
# Convert dataFrame to numpy array in float32 values, and scale to 
# zero mean, unit variance for each feature
data = StandardScaler().fit_transform(np.float32(train.values))
# Reshape the data to BS x 28 x 28 x 1. (-1 in reshape indicates that the 
# value is inferred from the remaining parameters)
data = data.reshape(-1, 28, 28, 1)
# Extract a validation set
VALID = 10000
train_data, valid_data = data[:-VALID], data[-VALID:]
train_labels, valid_labels = labels[:-VALID], labels[-VALID:]

#### MODEL

tf_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
tf_labels = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')

# Conv layer 1
# Define weights for conv layer 1. 32 5x5 kernels applied on depth 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(tf_data, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Conv layer 2

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Fully connected layer 2 with dropout
keep_prob = tf.placeholder(tf.float32)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

#### LOSS FUNCTION

# Logits are unnormalised inputs of a neural networkS
tf_loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=y_conv))

tf_acc = tf.reduce_mean(tf.to_float(
        tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))))

train_step = tf.train.AdamOptimizer(1e-4).minimize(tf_loss)

#### TRAINING
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# Number of training steps
STEPS = 1500
# Batch size
BATCH = 100
# Returns an object that returns idx of STEPS batches, each batch having
# BATCH number of idx of training samples.
history = [(0, np.nan, 10)]
ss = ShuffleSplit(n_splits=STEPS, train_size=BATCH)
for step, (idx, _) in enumerate(ss.split(train_data, train_labels), start=1):
    fd = {tf_data: train_data[idx], tf_labels: train_labels[idx], 
        keep_prob: 0.5}
    train_step.run(feed_dict=fd)

    if step % 100 == 0:
        fd = {tf_data: valid_data, tf_labels: valid_labels, keep_prob: 0.5}
        # sess.run(var, dict) == var.run(dict) == var.eval(dict)
        valid_loss, valid_accuracy = sess.run([tf_loss, tf_acc], feed_dict=fd)
        history.append((step, valid_loss, valid_accuracy))
        print("Step %i \t Valid. Acc. = %f" % (step, valid_accuracy), end='\n')


    # if i % 100 == 0:
    #     train_accuracy = accuracy.eval(feed_dict={
    #         x: batch[0], y_: batch[1], keep_prob: 1.0})
    #     print("step %d, training accuracy %g" % (i, train_accuracy))

    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#### RESULTS

# Zero mean, unit variance.
tf_pred = tf.nn.softmax(y_conv)
test_data = StandardScaler().fit_transform(np.float32(test.values))
test_data = test_data.reshape(-1, 28, 28, 1)
test_labels = np.zeros(test_data.shape[0])
print('EVALUATING TEST SAMPLES')
for i in range(0, len(test_data), BATCH):
    idx2 = i + 100
    if (i % 1000) == 0:
        print(str(i) + '/' + str(len(test_data)))
    test_pred = tf_pred.eval(feed_dict={tf_data: test_data[i:idx2], 
        keep_prob: 1})
    test_labels[i:idx2] = np.argmax(test_pred, axis=1)


# print("test accuracy %g" % accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

submission = pd.DataFrame(data={'ImageId': (np.arange(test_labels.shape[0])+1),
    'Label': test_labels})

submission.to_csv('submission.csv', index=False)

sess.close()