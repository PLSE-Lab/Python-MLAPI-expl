# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.

# Find GPU device
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))

# Import data
raw_data = pd.read_csv("../input/train.csv")
sub_data = pd.read_csv("../input/test.csv")

# # Split training and validation dataset
# train, val = train_test_split(raw_data, test_size=0.1, random_state=1)

# # Construct train_X, train_y, val_X, val_y
# train_y = train["label"]
# train_X = train.drop("label", axis=1).values
# val_y = val["label"]
# val_X = val.drop("label", axis=1).values

# # One-hot encoding y
# train_Y = to_categorical(train_y, num_classes=10, dtype="float32")
# val_Y = to_categorical(val_y, num_classes=10, dtype="float32")

# Construct train_X, train_y using all data
train_y = raw_data["label"]
train_X = raw_data.drop("label", axis=1).values/255
# One-hot encoding y
train_Y = to_categorical(train_y, num_classes=10, dtype="float32")

# data to be predicted
sub_X = sub_data.values/255
sub_m = sub_X.shape[0]

tf.reset_default_graph()

# Define placeholder
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
    
# Define how to construct batch
def to_batch(X, Y, batch_size=128, seed=1):
    m = X.shape[0]
    num_batches = int(m/batch_size)
    batches = []
    np.random.seed(seed)
    permutation = np.random.permutation(m)
    X_shuffled = np.take(X, permutation, axis=0)  # ???
    Y_shuffled = np.take(Y, permutation, axis=0) # ???
    for i in range(num_batches):
        bt_X = X_shuffled[i*batch_size : (i + 1)*batch_size]
        bt_Y = Y_shuffled[i*batch_size : (i + 1)*batch_size]
        batches.append((bt_X, bt_Y))
    if(m%batch_size !=0):
        bt_X = X_shuffled[num_batches*batch_size : m]
        bt_Y = Y_shuffled[num_batches*batch_size : m]
        batches.append((bt_X, bt_Y))
    return batches
        
    
# Define Weight and bias
def weight_variable(shape, name):
    return tf.get_variable(name, initializer=tf.random_normal(shape=shape, stddev=0.1))
def bias_variable(shape, name):
    return tf.get_variable(name, initializer=tf.constant(0.1, shape=shape))
    
# Construct CNN model

learning_rate = 0.001
epochs = 50
batch_size = 256
with tf.device("/gpu:0"):
    # 1st layer (convolutional + max pooling)
    W_1_1 = weight_variable([5, 5, 1, 32], "w11")
    b_1_1 = bias_variable([32], "b11")
    C_1_1 = tf.nn.conv2d(X, W_1_1, strides=[1, 1, 1, 1], padding="SAME")
    A_1_1 = tf.nn.relu(tf.add(C_1_1, b_1_1))
    # try more than 2 cnn
    W_1 = weight_variable([5, 5, 32, 32], "w1")
    b_1 = bias_variable([32], "b1")
    C_1 = tf.nn.conv2d(A_1_1, W_1, strides=[1, 1, 1, 1], padding="SAME")
    A_1 = tf.nn.relu(tf.add(C_1, b_1))
    A_1 = tf.nn.max_pool(A_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    # 2nd layer (convolutional + max pooling)
    W_2_1 = weight_variable([5, 5, 32, 64], "w21")
    b_2_1 = bias_variable([64], "b21")
    C_2_1 = tf.nn.conv2d(A_1, W_2_1, strides=[1, 1, 1, 1], padding="SAME")
    A_2_1 = tf.nn.relu(tf.add(C_2_1, b_2_1))
    # try more than 2 cnn
    W_2 = weight_variable([5, 5, 64, 64], "w2")
    b_2 = bias_variable([64], "b2")
    C_2 = tf.nn.conv2d(A_2_1, W_2, strides=[1, 1, 1, 1], padding="SAME")
    A_2 = tf.nn.relu(tf.add(C_2, b_2))
    A_2 = tf.nn.max_pool(A_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    # 3rd layer (fully connected)
    A_2 = tf.reshape(A_2, shape=[-1, 7*7*64])
    W_3 = weight_variable([7*7*64, 512], "w3")
    b_3 = bias_variable([512], "b3")
    A_3 = tf.nn.relu(tf.add(tf.matmul(A_2, W_3), b_3))
    # A_3_drop=tf.nn.dropout(A_3, 0.3)
    # # 4th layer (fully connected)
    # W_4 = weight_variable([2048, 1024], "w4")
    # b_4 = bias_variable([1024], "b4")
    # A_4 = tf.nn.relu(tf.add(tf.matmul(A_3, W_4), b_4))
    # last layer (softmax)
    W_5 = weight_variable([512, 10], "w5")
    b_5 = bias_variable([10], "b5")
    A_5 = tf.add(tf.matmul(A_3, W_5), b_5)
    prediction = tf.nn.softmax(A_5)
    
    # Compute cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=A_5, labels=y))
    
    # Define accuracy
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # Initializer
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    seed = 0
    print("Total iteration is %i" % epochs)
    for i in range(epochs):
        seed += 1
        batches = to_batch(train_X, train_Y, batch_size=batch_size, seed=seed)
        for batch in batches:
            (t_x, t_y) = batch
            sess.run(optimizer, feed_dict={X:t_x.reshape((-1, 28, 28, 1)), y:t_y})
        # if(i%10 == 0):
        #     print("%i iteration accuracy is: %f" %(i, sess.run(accuracy, feed_dict={X:train_X.reshape((-1, 28, 28, 1)), y:train_Y})))
        #     # test accuracy
        #     print("Accuracy on test data: %f" % sess.run(accuracy, feed_dict={X:val_X.reshape((-1, 28, 28, 1)), y:val_Y}))
        # print("%i iteration finished" % i)
        # if(i%10 == 0):
        #     print("%i iteration training accuracy is: %f" % (i, sess.run(accuracy, feed_dict={X:train_X.reshape((-1, 28, 28, 1)), y:train_Y})))
        print("%i iteretion finished" % i)
    
    # make prediction and output results
    sub_y = sess.run(prediction, feed_dict={X:sub_X.reshape((-1, 28, 28, 1))})
    # sub = pd.DataFrame({"ImageId": np.arange(1,sub_m + 1), "Label": sub_y})
    sub = pd.DataFrame()
    sub["ImageID"] = np.arange(1, sub_m + 1)
    sub["Label"] = np.argmax(sub_y, axis=1)
    sub.to_csv("submission.csv", index=False)
    # sub["ImageId"] = submission_df.index + 1
    # pd.DataFrame({"ImageId": sub["ImageId"], "Label": sub["Label"]}).to_csv("submission.csv", index=False)
    print("Submission file finished")