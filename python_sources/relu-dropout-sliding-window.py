# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import glob
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import matplotlib.pyplot as plt

NUM_READS = 8 # each row contains 8 reads
NUM_SENSORS = 8 # each read has 8 sensors(channels)

D = NUM_SENSORS * NUM_READS # number of input features
M1 = 40 # first layer number of nodes, relatively arbitrarily chosen
M2 = 25 # second hidden layer number of nodes, relatively arbitrarily chosen
M3 = 20 # third hidden layer number of nodes, relatively arbitrarily chosen
K = 4 # output layer nodes or number of classes

def read_data():
    allFiles = glob.glob("../input/*.csv")
    print(allFiles)

    list = []
    for file in allFiles:
        read = pd.read_csv(file, header = None)
        list.append(read)
    df = pd.concat(list)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.05)
    return Xtrain, Xtest, Ytrain, Ytest

def read_data_sliding():
    print('Using ', D, ' features.')
    allFiles = glob.glob("../input/*.csv")
    list = []
    print('----------------')
    for file in allFiles:
        df = pd.read_csv(file)
        readX = df.iloc[:, :-1].values
        aClass = int(df.iloc[0,-1])
        print('Input X', readX.shape)  
        print('Class', aClass)
        roll = readX
        listRolls = []
        for i in range(1, NUM_READS - 1):
            roll = np.roll(readX, -NUM_SENSORS)
            listRolls.append(roll)
            
        for listRoll in listRolls:
            #insert result class into the last column
            rollWithClass = np.insert(listRoll, D, values=aClass, axis=1)
            list.append(rollWithClass)
        #insert result class into the last column
        readMatrix = np.insert(readX, D, values=aClass, axis=1)
        print('Loaded samples: ', readMatrix.shape)
        print('Loaded class:', aClass)
        print('----------------')
        list.append(readMatrix)
    data = np.concatenate(list)
    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.05)
    return Xtrain, Xtest, Ytrain, Ytest

Xtrain, Xtest, Ytrain, Ytest = read_data_sliding()

N = len(Ytrain)
T = np.zeros((N, K)) 
for i in range(N):
  T[i, Ytrain[i]] = 1 # this creates an indicator/dummy variable matrix for the output layer. We need to do this for
# two reasons. 1) it creates an NxK matrix that will be broadcastable with the predictions generated from the forward
# function and used in the cost function. 2) when we argmax the predictions, it will turn into a matrix NxK of values only
# either 1 or 0 which can directly be compared with T to test the accuracy

def initialize_weights_and_biases(shape):
  return tf.Variable(tf.random_normal(shape, stddev=0.01))

def feed_forward(W4, W3, W2, W1, b4, b3, b2, b1, X):
  Z1 = tf.matmul(X, W1)
  Z1 = tf.nn.dropout(Z1, 0.9)
  Z1 = tf.nn.relu(Z1 + b1)
  
  Z2 = tf.matmul(Z1, W2)
  Z2 = tf.nn.dropout(Z2, 0.9)
  Z2 = tf.nn.relu(Z2 + b2)
  
  Z3 = tf.nn.relu(tf.matmul(Z2, W3) + b3)
  return tf.matmul(Z3, W4) + b4

tfX = tf.placeholder(tf.float32, [None, D]) # creates placeholder variables without actually assigning values to them yet
tfY = tf.placeholder(tf.float32, [None, K]) # None means it can take any size N total number of instances


W1 = initialize_weights_and_biases([D, M1])
W2 = initialize_weights_and_biases([M1, M2])
W3 = initialize_weights_and_biases([M2, M3])
W4 = initialize_weights_and_biases([M3, K])
b1 = initialize_weights_and_biases([M1])
b2 = initialize_weights_and_biases([M2])
b3 = initialize_weights_and_biases([M3])
b4 = initialize_weights_and_biases([K])

pYGivenX = feed_forward(W4, W3, W2, W1, b4, b3, b2, b1, tfX)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
       labels = tfY, logits = pYGivenX))

trainModel = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).minimize(cost)
predictOutput = tf.argmax(pYGivenX, 1) # 1 refers to axis = 1, meaning it does argmax on each instance n

session = tf.Session()
initializer = tf.global_variables_initializer()
session.run(initializer)


for i in range(4000):
  session.run(trainModel, feed_dict = {tfX: Xtrain, tfY: T})
  pred = session.run(predictOutput, feed_dict = {tfX:Xtrain, tfY:T})
  if i % 500 == 0:
    print("classification_rate: {}".format(np.mean(Ytrain == pred)))
    

# Test Set evaluation
Ntest = len(Ytest)
Ttest = np.zeros((Ntest, K)) # test set indicator matrix
for i in range(Ntest):
  Ttest[i, Ytest[i]] = 1
  
predtest = session.run(predictOutput, feed_dict = {tfX: Xtest, tfY: Ttest})
print("Test Set classification rate: {}".format(np.mean(Ytest == predtest))) # evaluates boolean as either 1 or 0 then 
# takes mean