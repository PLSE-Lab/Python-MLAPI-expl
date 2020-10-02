import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

import numpy as np
import tensorflow as tf
import functools
import time

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('keep_prob', 0.5, 'keep probability')
flags.DEFINE_integer('epochs', 10, 'training epochs')

def logs(func):
    @functools.wraps(func)
    def wraper(*args, **kw):
        start = time.time()
        result = func(*args, **kw)
        duration = time.time() - start
        print('time duration is %.3f' % duration)
        return result
    return wraper

@logs
def one_hot(labels, count):
    labels = labels.astype(dtype=np.int32)
    sparse_labels = np.zeros(shape=[labels.shape[0], count], dtype=np.int32)
    sparse_labels[np.arange(labels.shape[0]), labels] = 1
    return sparse_labels

if __name__  == '__main__':
    train_data_url = '../input/train.csv'
    test_data_url = '../input/test.csv'
    train_data = np.genfromtxt(train_data_url, delimiter=',')
    test_data = np.genfromtxt(test_data_url, delimiter=',')
    train_data = train_data[1:]
    test_data = test_data[1:]
    train_label = train_data[:, 0]
    train_data = train_data[:, 1:]
    train_data = train_data/255.
    test_data = test_data/255.
    labels = one_hot(train_label, 10)

    x_train = train_data
    x_test = test_data
    #data_perprocessing
    mean = np.mean(train_data)
    x_train = (train_data - mean)
    x_test = (test_data - mean)
    #build model
    input_image = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='input_image')
    train_labels = tf.placeholder(shape=[None, 10], dtype=tf.int32, name='input_labels')
    probs = tf.placeholder(shape=[], dtype=tf.float32, name='probs')
    w1 = tf.Variable(tf.truncated_normal(shape=[784, 300], dtype=tf.float32, stddev=0.1), name='layer1_weights')
    b1 = tf.Variable(tf.zeros(shape=[300]), name='layer1_bias')
    w2 = tf.Variable(tf.truncated_normal(shape=[300, 10], dtype=tf.float32, stddev=0.1), name='layer2_weights')
    b2 = tf.Variable(tf.zeros(shape=[10]), name='layer2_bias')
    layer1_output = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(input_image, w1), b1)), keep_prob=probs)
    layer2_output = tf.nn.bias_add(tf.matmul(layer1_output, w2), b2)
    softmax_output = tf.nn.softmax(layer2_output)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=layer2_output))
    #train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    equals = tf.equal(tf.argmax(softmax_output, dimension=1), tf.argmax(train_labels, dimension=1))
    predict = tf.argmax(softmax_output, dimension=1)
    accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))
    batch_size = 128
    iters = x_train.shape[0]//batch_size
    a = np.arange(x_train.shape[0])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for _ in range(FLAGS.epochs):
        np.random.shuffle(a)
        for i in range(iters):
            data_batch = x_train[a[i*batch_size:(i+1)*batch_size]]
            label_batch = labels[a[i*batch_size:(i+1)*batch_size]]
            _, loss_val = sess.run([train_op, loss], feed_dict={input_image: data_batch, train_labels: label_batch, probs:FLAGS.keep_prob})
            if i % 100 == 0:
                print('training loss is %.3f' % loss_val)
            if i % 500 == 0:
                acc, train_pred, train_softmax = sess.run([accuracy, predict, softmax_output], feed_dict={input_image: data_batch, train_labels: label_batch, probs:1.0})
                print('training accuracy ===================================== %.3f' % acc)
    results = sess.run(predict, feed_dict={input_image: x_test, probs:1.0})
    print(results[0:100])
    index = np.arange(results.shape[0])
    index = index.reshape([index.shape[0], -1])
    results = results.reshape([results.shape[0], -1])
    test_predict = np.hstack((index, results))
    np.savetxt('results.csv', test_predict, delimiter=', ')


