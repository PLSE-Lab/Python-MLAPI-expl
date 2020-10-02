# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import matplotlib.pyplot as plt

#data preprocessing
# prosthetic hand EMG sensor files
df0 = pd.read_csv('../input/0.csv', header = None)
df1 = pd.read_csv('../input/1.csv', header = None)
df2 = pd.read_csv('../input/2.csv', header = None)
df3 = pd.read_csv('../input/3.csv', header = None)

df = pd.concat([df0, df1, df2, df3])

D = 64 # number of input features
M1 = 34 # first layer number of nodes, relatively arbitrarily chosen
M2 = 17 # second hidden layer number of nodes, relatively arbitrarily chosen
K = 4 # output layer nodes or number of classes

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2)

N = len(Ytrain)
T = np.zeros((N, K)) 
for i in range(N):
  T[i, Ytrain[i]] = 1 # this creates an indicator/dummy variable matrix for the output layer. We need to do this for
# two reasons. 1) it creates an NxK matrix that will be broadcastable with the predictions generated from the forward
# function and used in the cost function. 2) when we argmax the predictions, it will turn into a matrix NxK of values only
# either 1 or 0 which can directly be compared with T to test the accuracy

def initialize_weights_and_biases(shape):
  return tf.Variable(tf.random_normal(shape, stddev=0.01))

# using two hidden layers
def feed_forward(W3, W2, W1, b3, b2, b1, X):
  Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
  Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
  return tf.matmul(Z2, W3) + b3

tfX = tf.placeholder(tf.float32, [None, D]) # creates placeholder variables without actually assigning values to them yet
tfY = tf.placeholder(tf.float32, [None, K]) # None means it can take any size N total number of instances


W1 = initialize_weights_and_biases([D, M1])
W2 = initialize_weights_and_biases([M1, M2])
W3 = initialize_weights_and_biases([M2, K])
b1 = initialize_weights_and_biases([M1])
b2 = initialize_weights_and_biases([M2])
b3 = initialize_weights_and_biases([K])

pY_given_X = feed_forward(W3, W2, W1, b3, b2, b1, tfX)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
       labels = tfY, logits = pY_given_X))

train_model = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
predict_output = tf.argmax(pY_given_X, 1) # 1 refers to axis = 1, meaning it does argmax on each instance n

session = tf.Session()
initializer = tf.global_variables_initializer()
session.run(initializer)


for i in range(2000):
  session.run(train_model, feed_dict = {tfX: Xtrain, tfY: T})
  pred = session.run(predict_output, feed_dict = {tfX:Xtrain, tfY:T})
  if i % 250 == 0:
    print("classification_rate: {}".format(np.mean(Ytrain == pred)))
    

# Test Set evaluation
Ntest = len(Ytest)
Ttest = np.zeros((Ntest, K)) # test set indicator matrix
for i in range(Ntest):
  Ttest[i, Ytest[i]] = 1
  
predtest = session.run(predict_output, feed_dict = {tfX: Xtest, tfY: Ttest})
print("Test Set classification rate: {}".format(np.mean(Ytest == predtest))) # evaluates boolean as either 1 or 0 then 
# takes mean