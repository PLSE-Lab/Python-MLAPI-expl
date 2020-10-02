# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Load the training and test files
trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")

# MNIST
NUM_CLASSES = 10
NUM_ITERATIONS = 1000
DROPOUT = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE = 100

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import tensorflow.contrib.slim as slim

# Create the CNN model
def createCNN(images, dropout):
    numFilters = [256, 512, 1024, 2048, 4096, 4096]
    filterSize = [3, 3]
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        # 28x28
        net = slim.conv2d(images, numFilters[0], filterSize, padding='SAME', scope='Conv1_1')
        net = slim.conv2d(net, numFilters[1], filterSize, padding='SAME', scope='Conv1_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        
        # 14x14
        net = slim.conv2d(net, numFilters[2], filterSize, padding='SAME', scope='Conv2_1')
        net = slim.conv2d(net, numFilters[3], filterSize, padding='SAME', scope='Conv2_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        
        # 7x7
        net = slim.conv2d(net, numFilters[4], filterSize, padding='VALID', scope='Conv3_1')
        net = slim.conv2d(net, numFilters[5], filterSize, padding='VALID', scope='Conv3_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        
        # 3x3
        # Fully-connected blocks
        net = slim.fully_connected(net, 4096, scope='fc4')
        net = slim.dropout(net, dropout, scope='dropout4')
        net = slim.fully_connected(net, NUM_CLASSES, activation_fn=None, scope='fc5')
        
        return net
        
        
# Create network placeholders
imagePlaceholder = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="imagePlaceholder")
# oneHotLabelsPlaceholder = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES], name="labelsPlaceholder")
labelsPlaceholder = tf.placeholder(dtype=tf.int32, shape=[None], name="labelsPlaceholder")
dropoutPlaceholder = tf.placeholder(dtype=tf.float32, name="inputKeepProbability")

# Create the network
logits = createCNN(imagePlaceholder, dropoutPlaceholder)

# Create the accuracy node
correct_predictions = tf.equal(tf.argmax(logits, 1), labelsPlaceholder)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

# Specify the loss function
# slim.losses.softmax_cross_entropy(onehot_labels=oneHotLabelsPlaceholder, logits=logits)
slim.losses.sparse_softmax_cross_entropy(labels=labelsPlaceholder, logits=logits)
loss = losses.get_total_loss() # XE automatically added to the list by Slim

# Specify the optimization parameters
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

# Create the optimization step
optimize = optimizer.minimize(loss)

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Variables initialization operator
init = tf.global_variables_initializer()

# Start the main training loop

with tf.Session() as session:
    for i in range(NUM_ITERATIONS):
        # Fetch a batch of data from the system
        
        feed_dict = {imagePlaceholder: imagesBatch, labelsPlaceholder: labelsBatch, dropoutPlaceholder: DROPOUT}
        session.run([loss, optimizer], feed_dict=feed_dict)
    
    # Save final model weights to disk
    saver.save(sess, "models/checkpoint")
        
# Create the predictions

with tf.Session() as session:
    # Fetch a batch of data from the system
    pass