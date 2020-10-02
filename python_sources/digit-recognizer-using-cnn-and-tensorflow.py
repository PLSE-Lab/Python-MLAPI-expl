#!/usr/bin/env python
# coding: utf-8

# 
# #**Introduction** 
# 
# The goal in this competition is to take an image of a handwritten single digit, and determine what that digit is.  
# 
# The data for this competition were taken from the MNIST dataset. The MNIST ("Modified National Institute of Standards and Technology") dataset is a classic within the Machine Learning community that has been extensively studied.  More detail about the dataset, including Machine Learning algorithms that have been tried on it and their levels of success, can be found at http://yann.lecun.com/exdb/mnist/index.html.
# ![enter image description here][3]
# 
# #**Methode**
# ###Convolutional Neural Network using Tensorflow
# 
# Convolutional neural networks (CNNs) consist of multiple layers of receptive fields. These are small neuron collections which process portions of the input image. The outputs of these collections are then tiled so that their input regions overlap, to obtain a better representation of the original image; this is repeated for every such layer. Tiling allows CNNs to tolerate translation of the input image.
# 
# Convolutional networks may include local or global pooling layers, which combine the outputs of neuron clusters. They also consist of various combinations of convolutional and fully connected layers, with pointwise nonlinearity applied at the end of or after each layer. A convolution operation on small regions of input is introduced to reduce the number of free parameters and improve generalization. One major advantage of convolutional networks is the use of shared weight in convolutional layers, which means that the same filter (weights bank) is used for each pixel in the layer; this both reduces memory footprint and improves performance. [Wiki][1]
# 
# ![enter image description here][2]
# 
# 
# 
# 
#   [1]: https://en.wikipedia.org/wiki/Convolutional_neural_network
#   [2]: http://www.pyimagesearch.com/wp-content/uploads/2014/06/cnn_architecture.jpg
#   [3]: http://covartech.github.io/images/testBlog_01.png

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.utils import shuffle
import time

print('Tensorflow Version:', tf.__version__)


# Initializing of weigth and bias for each hidden layer, the filters for each convolutional layer and the y-indicator. 

# In[ ]:


def error_rate(targets, predictions):
    return np.mean(targets != predictions)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    print('Y indicator.shape:', ind.shape)
    return ind

def init_weight_and_bias(M1, M2):
    W = np.random.randn(int(M1), int(M2)) / np.sqrt(int(M1) + int(M2))
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32) 

def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32)


# **Class for the Hidden Layer**

# In[ ]:


class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]
        print('Hiddenlayer ', self.id, ': M1: ', self.M1, ', M2: ', self.M2, ', W.shape: ', W.shape, ', b.shape: ', b.shape)

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


# **Class for the Convolutional Layer**
#  

# In[ ]:


class ConvPoolLayer(object):
    def __init__(self, mi, mo, fw=5, fh=5, conv_id=0, poolsz=(2, 2)):
        # mi = input feature map size
        # mo = output feature map size
        self.conv_id = conv_id 
        sz = (fw, fh, mi, mo)
        W0 = init_filter(sz, poolsz)
        self.W = tf.Variable(W0)
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = tf.Variable(b0)
        self.poolsz = poolsz
        self.params = [self.W, self.b]
        print('ConvPoolLayer ', self.conv_id, ': filter-sz: ', sz, ', pool-sz: ', poolsz)


    def forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return tf.tanh(pool_out)


# **Class of the Convolutional Neural Network**
# 
# In this class the convolutional layer and the hidden layer as well as the logistic regression output layer are initialized. Subsequently, the training set is divided into the corresponding batch size. The RMSPropOptimizer is used as the optimization algorithm. This can also be replaced by a gradient descent algorithm. The method 'predict' is used for validation to check the learning behavior of the algorithm after N iterations. The 'predictImage' method is used to recognize one or more images.

# In[ ]:


class CNN(object):
    def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes
        
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        self.session = tf.Session(config=config)
        tf.logging.set_verbosity(tf.logging.ERROR)
    
    def fit(self, X, Y, lr=10e-4, mu=0.99, reg=10e-4, decay=0.99999, eps=10e-3, batch_sz=30, epochs=3, show_fig=True):
        lr = np.float32(lr)
        mu = np.float32(mu)
        reg = np.float32(reg)
        decay = np.float32(decay)
        eps = np.float32(eps)
        K = len(set(Y))

        # make a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)

        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        Yvalid_flat = np.argmax(Yvalid, axis=1) # for calculating error rate

        # initialize convpool layers
        N, d, d, c = X.shape
        mi = c
        outw = d
        outh = d
        self.convpool_layers = []
        conv_id = 1
        for mo, fw, fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi, mo, fw, fh, conv_id)
            self.convpool_layers.append(layer)
            outw = outw / 2
            outh = outh / 2
            mi = mo
            conv_id += 1

        # initialize mlp layers
        self.hidden_layers = []
        M1 = self.convpool_layer_sizes[-1][0]*outw*outh 
        hidd_id = 1
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, hidd_id)
            self.hidden_layers.append(h)
            M1 = M2
            hidd_id += 1

        # logistic regression layer
        W, b = init_weight_and_bias(M1, K)
        self.W = tf.Variable(W, 'W_logreg')
        self.b = tf.Variable(b, 'b_logreg')
        print('Logistic-Regression Layer: ', 'W.shape: ', W.shape, ', b.shape: ', b.shape, '\n')

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.convpool_layers:
            self.params += h.params
        for h in self.hidden_layers:
            self.params += h.params

        # set up tensorflow functions and variables
        tfX = tf.placeholder(tf.float32, shape=(None, d, d, c), name='X')
        tfY = tf.placeholder(tf.float32, shape=(None, K), name='Y')
        act = self.forward(tfX)

        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(act, tfY)) + rcost
        prediction = self.predict(tfX)

        train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)

        n_batches = N / batch_sz
        costs = []
        errors = []
        count = 0
        
        init = tf.initialize_all_variables()
        self.session.run(init)
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(int(n_batches)):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                self.session.run(train_op, feed_dict={tfX: Xbatch, tfY: Ybatch})

                if j % int(n_batches / 5) == 0:
                    c = self.session.run(cost, feed_dict={tfX: Xvalid, tfY: Yvalid})
                    costs.append(c)

                    p = self.session.run(prediction, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        
                    e = error_rate(Yvalid_flat, p)
                    errors.append(e)
                    print("epochs:", i, "batch:", j, "batch size:", int(n_batches), "cost:", c, "error rate:", e)
                        
        if show_fig:
            plt.ion()
            plt.xlabel('Number of Iterations x ' + str(int(n_batches / 5)))
            plt.ylabel('Error Rate')
            plt.plot(errors, linestyle='solid', c='Red')

    def predictImage(self, X):
        if X.ndim == 3:
            w, h, c = X.shape
            X = X.reshape(1, w, h, c)
        else:
            N, w, h, c = X.shape
        tfX = tf.placeholder(tf.float32, shape=(None, w, h, c), name='X')
        prediction = self.predict(tfX)
        Y_flat = self.session.run(prediction, feed_dict={tfX: X.astype(np.float32)})
        return Y_flat
        
    def forward(self, X):
        Z = X
        for c in self.convpool_layers:
            Z = c.forward(Z)
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        pY = self.forward(X)
        return tf.argmax(pY, 1)


# **Loading Data**
# 
# Loading the training and test images from the *.csv file and convert it from an unnormalized 1D vector to an normalized 2D image matrix. After the loading we convert it to an Tensorflow format N x width x heigth, colors.

# In[ ]:


def getData(file, with_label=True):
    Y = []
    X = []
    first = True
    for line in open(file):
        if first:
            first = False
        elif with_label:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1:]])
        else:
            row = line.split(',') 
            Y.append(-1)
            X.append([int(p) for p in row[:]])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y

def getImageData(file, set='Train'):
    if set == 'Train':
        X, Y = getData(file, with_label=True)
    else:
        X, Y = getData(file, with_label=False)
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

Xtrain, Ytrain = getImageData('../input/train.csv', set='Train')
Xtest, Ytest = getImageData('../input/test.csv', set='Test')

# reshape X for tf: N x w x h x c
Xtrain = Xtrain.transpose((0, 2, 3, 1))
print("Xtrain.shape:", Xtrain.shape)
    
Xtest = Xtest.transpose((0, 2, 3, 1))
print("Xtest.shape:", Xtest.shape)


# **Create the CNN-Model**

# In[ ]:


model = CNN(
        convpool_layer_sizes=[(20, 5, 5), (20, 5, 5)],
        hidden_layer_sizes=[500, 300],
    )


# **Training**
# 
# Training the model with the following parameter:
# 
# Convolutional Layer = 2, 
# Convpool layer sizes = [(20, 5, 5), (20, 5, 5)],
# Hidden layer sizes = [500, 300],
# Learning Rate = 10e-4,
# Momentum = 0.99,
# Regularization =10e-4,
# Decay = 0.99999,
# Eps = 10e-3, 
# Batch size = 30, 
# Epochs= 5
# 
# Note: To get a better learning result it is better to increase the epochs to 10. But on kaggle I get an server timeout.    

# In[ ]:


model.fit(Xtrain, Ytrain, epochs=5, batch_sz=30)


# #**Results**
# 
# In order to prove the correctness of the CNN algorithm the accuracy of the whole test set was checked by Kaggle. 
# 
# ###**Results on Test-Set**
# 
# **Score = 0.98986**
# 
# **Accuracy = 98,986%**
# 

# To evaluate the CNN-model I used the whole trainings set. 
# 
# ###Results Training-Set:
# **Error-Rate = 0.00416**
# 
# **Accuracy = 99.583%**

# The following pictures show randomly selected images from the test set which were classified by the CNN-model.

# In[ ]:


def showImage(X, index):
    N, w, h, c = X.shape
    grid = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            grid[i,j] = X[index,i,j,0]
    plt.rcParams["figure.figsize"] = [1.5,1.5]
    plt.imshow(grid, cmap='gray')
    plt.ion()
    plt.show()
    
for i in range(10):    
    index  = np.random.choice(Xtest.shape[0])
    Ytest = model.predictImage((Xtest[index]))
    print('Prediction:', Ytest)
    showImage(Xtest, index)

