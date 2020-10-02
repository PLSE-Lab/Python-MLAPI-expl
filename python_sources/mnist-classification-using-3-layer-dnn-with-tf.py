# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

tf.reset_default_graph()

def mini_batches(x, y, seed, batch_size = 64,):
    np.random.seed(seed)
    m = x.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = x[:, permutation]
    shuffled_Y = y[:, permutation]
    num_complete_minibatches = math.floor(m/batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * batch_size:(k + 1) * batch_size]
        mini_batch_Y = shuffled_Y[:,k * batch_size:(k + 1) * batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def one_hot(y, C):
    C = tf.constant(C , name ="C")
    one_hot_matrix = tf.one_hot(indices = y, depth = C, axis = 0)
    sess = tf.Session()
    encode = sess.run(one_hot_matrix)
    sess.close()
    return encode


def forward_propagation(X, parameters, is_training):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    keep_prob = 0.7
    if not is_training:
        keep_prob = 1
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    A1 = tf.contrib.layers.dropout(A1, keep_prob = keep_prob)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    A2 = tf.contrib.layers.dropout(A2, keep_prob = keep_prob)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

def model_costs(Z3, Y):
    z3 = tf.transpose(Z3)
    y = tf.transpose(Y)
    cost = cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = z3))
    return cost


def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [512, 784], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [512, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [256, 512], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [256, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [10, 256], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [10, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
y_train  = train.iloc[:,0].values
X_train  = train.iloc[:,1:].values
X_test  = test.iloc[:,:].values
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

# plt.imshow(X_train[1].reshape((28, 28)),  cmap=plt.cm.gray, interpolation='nearest')
# plt.show()

X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T
X_train = X_train/255
X_test = X_test/255
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


y_train = one_hot(y_train, 10)
print(y_train.shape)

lr = 0.005
num_epochs = 100
batch_size = 64

X = tf.placeholder(tf.float32, [784, None], name = "X")
Y = tf.placeholder(tf.float32, [10, None], name = "Y")
print("X = " + str(X))
print("Y = " + str(Y))
is_training = True

parameters = initialize_parameters()
Z3 = forward_propagation(X, parameters, is_training)
cost = model_costs(Z3, Y)
optimize = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon = 1e-08).minimize(cost)
init = tf.global_variables_initializer()
costs = []
seed = 1
with tf.Session() as sess:
    sess.run(init)
    for epochs in range(num_epochs):
        epoch_cost = 0
        num_batchs = int(X_train.shape[1]/batch_size)
        batchs = mini_batches(X_train, y_train, seed, batch_size)
        seed = seed + 1
        for mini_batch in batchs:
            (minibatch_X, minibatch_Y) = mini_batch
            _ , minibatch_cost = sess.run([optimize, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
            epoch_cost += minibatch_cost / num_batchs
        if epochs % 10 == 0:
            print ("Cost after epoch %i: %f" % (epochs, epoch_cost))
        if epochs % 10 == 0:
            costs.append(epoch_cost)
    parameters = sess.run(parameters)
    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Train Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(lr))
    plt.show()
    
    # Predictions
    is_training = False
    Z3 = forward_propagation(X, parameters, is_training)
    answer = sess.run(Z3, feed_dict={X: X_test})
    index_result = np.argmax(answer, axis = 0)
    # Testing the prediction
    plt.imshow(X_test[:,2].reshape((28, 28)),  cmap=plt.cm.gray, interpolation='nearest')
    plt.title("Predicted as "+ str(index_result[2]))
    plt.show()
    
    # Saving the predictions as CSV
    numbers = list(range(1, 28001))
    data = pd.DataFrame({"ImageId":numbers, "Label":index_result})
    data.shift(periods=1)
    data.dropna(inplace= True)
    print(data.head())
    data.to_csv('submission.csv', index=False)
    

    
    
