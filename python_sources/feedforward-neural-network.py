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

class Network(object):
    def __init__(self, xdim = 28*28, hidnum = 500, ydim = 10):
        W_hx = tf.Variable(tf.random_normal(stddev=1.0/xdim, shape=[xdim, hidnum]))
        b_h  = tf.Variable(tf.random_normal(stddev=1.0/xdim, shape=[hidnum]))
        W_yh = tf.Variable(tf.zeros(shape=[hidnum, ydim]))
        b_y  = tf.Variable(tf.zeros(ydim))
        self.params = [W_hx, b_h, W_yh, b_y]

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, xdim])
        self.hid = tf.tanh( tf.matmul( self.x, W_hx) + b_h)
        self.py = tf.nn.softmax( tf.matmul( self.hid, W_yh)+b_y)
        self.y_predict = tf.to_int32(tf.argmax( self.py, axis=1))
        
class Training(object):
    def __init__(self, model, masked=False, mask = None):
        self.x = model.x
        self.model = model
        if not masked:
            self.mask_ph = None
        else:
            self.mask_ph = model.mask
        self.mask = mask

        self.ytrue = tf.placeholder(dtype=tf.int32, shape=[None])
        batchsize = tf.shape(self.ytrue)[0]
        self.lrate = tf.Variable(0.0)
        self.cost = -tf.reduce_mean( tf.log( tf.gather_nd(model.py, tf.stack([tf.range( batchsize), self.ytrue],axis=1))))
        self.error = tf.reduce_mean(tf.to_float(tf.not_equal( model.y_predict, self.ytrue)))

        self.optimize = tf.train.GradientDescentOptimizer(self.lrate).minimize( self.cost, var_list=model.params)

    def set_data(self, dataset):
        self.train_data = dataset['train']
        self.test_data = dataset['test']

    def error_rate(self, sess, data_x, data_y, batchsize=500):
        num_batch = data_y.size//batchsize
        error_arr = np.empty(num_batch)
        for i in range(num_batch):
            beg = i*batchsize
            end = (i+1)*batchsize
            feed_dict={self.x : data_x[beg:end],
                self.ytrue : data_y[beg:end],
                }
            if self.mask_ph!=None:
                feed_dict[self.mask_ph] = self.mask
            error_arr[i] = sess.run(self.error, feed_dict)
        return error_arr.mean()

    def train(self, sess, batchsize = 20, epochs = 10, learning_rate = 0.01):
        sess.run(self.lrate.assign(learning_rate))
        for epoch in range(epochs):
            for i in range(self.train_data.size//batchsize):
                begin = i*batchsize
                end = (i+1)*batchsize
                feed_dict = {self.x : self.train_data.x[begin:end], 
                    self.ytrue : self.train_data.y[begin:end],
                    }
                if self.mask_ph!=None:
                    feed_dict[self.mask_ph] = self.mask
                self.optimize.run(feed_dict)
            self.print_err(sess, epoch)

    def train_once(self, sess, batchsize = 20, learning_rate = 0.01):
        sess.run(self.lrate.assign(learning_rate))
        for i in range(self.train_data.size//batchsize):
            begin = i*batchsize
            end = (i+1)*batchsize
            feed_dict = {self.x : self.train_data.x[begin:end], 
                self.ytrue : self.train_data.y[begin:end],
                }
            if self.mask_ph!=None:
                feed_dict[self.mask_ph] = self.mask
            self.optimize.run(feed_dict)
    
    def print_err(self, sess, epoch):
            print ("epoch: ", epoch+1, ", tarin error = ", self.error_rate(sess, self.train_data.x, self.train_data.y)*100, '%')

class Dataset(object):
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y
        self.size = _x.shape[0]

##load data
def load_mnist():
    data = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    data_dict = {}
    data_dict['train'] = Dataset(np.asarray(data.values[:,1:], dtype=np.float32)/255, data.values[:,0])
    data_dict['test']  = Dataset(np.asarray(test.values, dtype=np.float32)/255, None)
    return data_dict

n=Network()
t=Training(n)
data_dict = load_mnist()
t.set_data(data_dict)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    t.train(sess, epochs=100, batchsize=200, learning_rate=0.1)
    predict = sess.run(n.y_predict, feed_dict = {n.x:data_dict['test'].x})
output_data = {'ImageId':range(1, predict.size+1), "Label":predict}
output_df = pd.DataFrame(data=output_data)
output_df.to_csv("output.csv", index=False)