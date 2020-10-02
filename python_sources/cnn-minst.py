import pandas as pd
import numpy as np
import tensorflow as tf

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
train_x = train.drop('label', axis=1)
train_y = pd.DataFrame(np.zeros([len(train['label']), 10]), columns=[0,1,2,3,4,5,6,7,8,9])

# one-hot encoding
for idx in train.index:
    col = train.ix[idx, 'label']
    train_y.ix[idx, col] = 1

print(train_y.shape)
print(train_x.shape)

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 28 * 28 ])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y_pred = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y_true))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(0, len(train),100):
    batch_y = train_y[i:i+100]
    batch_x = train_x[i:i+100]
    train_step.run(feed_dict={x: batch_x, y_true: batch_y})

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

submission_path = '../input/sample_submission.csv'
pd.DataFrame(tf.argmax(y_pred,1).eval(feed_dict={x: test})).to_csv(submission_path)

