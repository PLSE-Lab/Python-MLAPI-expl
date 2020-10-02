import pandas as pd
import numpy as np
import tensorflow as tf
from math import trunc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.



def make_one_hot(m):
    result = pd.DataFrame((np.asarray(m)[:,None] == np.arange(10)).astype(int))
    return result


train_data = pd.read_csv("../input/train.csv", delimiter=',')
train_labels = make_one_hot(train_data.ix[:, 0])
train_inputs = train_data.ix[:, 1:]
test_inputs = pd.read_csv("../input/test.csv", delimiter=',')

print(test_inputs.shape)
print(test_inputs.iloc[100:100+100,:].shape)


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
                        
                        
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.random_normal([784,10], mean=0.5, stddev=1))
b = tf.Variable(tf.random_normal([10], mean=0.5, stddev=1))


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


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1500):
      batch_xs = train_inputs.sample(n=100)
      batch_ys = train_labels.loc[batch_xs.index]
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    #   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    

    result = []
    for i in range(0, len(test_inputs), 100):
       end = min(i+100, len(test_inputs))
       r = tf.nn.softmax(sess.run(y_conv, feed_dict={x: test_inputs.iloc[i:end, :], keep_prob: 1.0})).eval().tolist()
       result = result + r

       
    f = open("results.csv","w+")
    f.write("ImageId,Label\n")
    for i in range(0, len(result)):
        x = 0
        for j in range(0, 10):
            if(result[i][j] == 1):
                x = j
        f.write("{},{}\n".format(i+1, x))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: test_inputs, y_: test_labels}))