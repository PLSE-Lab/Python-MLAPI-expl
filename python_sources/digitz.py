import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#import the data into a pandas dataframe and then split it into train/test cases
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#-------------setting up training data---------

#create a one-hot vector
yDict = []
for num in train['label']:
        yTemp = []
        for i in range(10):
                if i != num:
                        yTemp.append(0)
                else:
                        yTemp.append(1)
        yDict.append(yTemp)
y_train = pd.DataFrame(yDict)
x_train = train.drop('label', axis=1)


#-------------setting up test data---------

#create a one-hot vector
x_test = test

#-------------------------Setting up tensorflow-------------------------
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32, [None,10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

yy = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = 0
        end = 50
        while end <= len(x_train):
                train_step.run(feed_dict={x: x_train.iloc[start:end], y_: y_train.iloc[start:end], keep_prob: 0.5})
                start = end
                end += 50
        prediction = []
        for j in range(len(x_test)):
                classification = yy.eval(feed_dict={x: x_test.iloc[j].values.reshape(1,784), keep_prob:0.5})
                i=0
                num = np.argmax(classification[0])
                prediction.append(num)
        labelId = x_test.index.tolist()
        labelId = [x+1 for x in labelId]


results = pd.DataFrame({'ImageId': labelId, 'Label':prediction}).sort_values('ImageId')
results.to_csv('results.csv', index=False)
