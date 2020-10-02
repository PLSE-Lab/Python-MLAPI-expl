#!/usr/bin/env python
# coding: utf-8

# #### Deep Neural Network written in Python for the MNSIT handwritten dataset from scratch without using any deep learning frameworks. I have implemented Droupout technique for regularisation and Mini-batch, Adams optimisation for optimising the gradient descent and Sigmoid is used as an activation function.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams["figure.figsize"] = 10,7

warnings.filterwarnings('ignore') # filter all warnings

# set a seed so that the results are consistent
np.random.seed(0)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input/digit-recognizer/'):
    for filename in filenames:
        print(filename)

# Any results you write to the current directory are saved as output.


# ## 1. Load the Train and Test dataset

# In[ ]:


test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


range_class = np.arange(10)

y = np.asfarray(train.iloc[:,0])
train_x = train.iloc[:,1:].values

train_x, dev_x, train_y, dev_y = train_test_split(train_x, y, test_size=0.2, random_state=42)

dev_ch_y = np.array([(range_class==label).astype(np.float) for label in dev_y])
train_ch_y = np.array([(range_class==label).astype(np.float) for label in train_y])


# In[ ]:


test_x = test.values


# ## 2. Data Visualize

# In[ ]:


y = train.iloc[:,0].value_counts()
x = range(len(y))
plt.bar(x, y, color='rgbymc')
plt.xticks(x, x)
plt.ylabel('no. of images w.r.t labels')
plt.xlabel('Lables between 0-9')
plt.grid()


# In[ ]:


# Creating a figure to display images in rows and columns pattern (1x 10)
figure = plt.figure()

# Manually setting the figure width and height
figure.set_size_inches(20.5, 8.5)

# Setting up an image in each figure with a title of image label
for itr in range(1, 10):
    plt.subplot(1, 10, itr)
    label = train.loc[itr,'label']
    pixels = train.iloc[itr,1:].values.reshape((28,28))
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')

# Displaying all image present in figure
plt.show()


# ## 3. Normalising the dataset

# In[ ]:


train_x = train_x / 255.
test_x  = test_x  / 255.


# ## 4. Dataset

# In[ ]:


shape_x = train_x.shape
shape_y = train_y.shape

shape_dev_x = dev_x.shape
shape_dev_y = dev_y.shape

m = train_y.shape[0]

print ('The shape of Train X is: %s' % str(shape_x))
print ('The shape of Train Y is: %s\n' % str(shape_y))
print ('The shape of Dev X is: %s' % str(shape_dev_x))
print ('The shape of Dev Y is: %s\n' % str(shape_dev_y))
print ('I have m = %d training examples! \n' % (m))
print ('I have m = %d dev examples!' % (shape_dev_x[0]))


# ## 5. Defining the Neural Network

# In[ ]:


def layer_size(X, Y):
    
    n_x = X.shape[1]
    n_h = 4
    n_y = Y.shape[1]
    
    return (n_x, n_h, n_y)


# In[ ]:


def initialise_parameter(n_x, n_h, n_y):
    
    np.random.seed(0)
    
    W1 = np.random.randn(n_h[0], n_x) * np.sqrt(1. / n_x)
    b1 = np.zeros(shape=(n_h[0], 1))
    
    W2 = np.random.randn(n_h[1], n_h[0]) * np.sqrt(1. / n_h[0])
    b2 = np.zeros(shape=(n_h[1], 1))
    
    W3 = np.random.randn(n_y, n_h[1]) * np.sqrt(1. / n_h[1])
    b3 = np.zeros(shape=(n_y, 1))
    
    assert(W1.shape == (n_h[0], n_x))
    assert(b1.shape == (n_h[0], 1))

    assert(W2.shape == (n_h[1], n_h[0]))
    assert(b2.shape == (n_h[1], 1))
    
    assert(W3.shape == (n_y, n_h[1]))
    assert(b3.shape == (n_y, 1))
    
    parameters = {"W1": W1, 
                  "b1": b1, 
                  "W2": W2, 
                  "b2": b2, 
                  "W3": W3, 
                  "b3": b3
                 }
    
    return parameters


# ## 6. Initialise Adam parameters 

# In[ ]:


def initialise_adam(parameters):
    
    v = {}
    s = {}
    
    v["dW1"] = np.zeros((parameters["W1"].shape[0],parameters["W1"].shape[1]))
    v["db1"] = np.zeros((parameters["b1"].shape[0],parameters["b1"].shape[1]))
    s["dW1"] = np.zeros((parameters["W1"].shape[0],parameters["W1"].shape[1]))
    s["db1"] = np.zeros((parameters["b1"].shape[0],parameters["b1"].shape[1]))
    
    v["dW2"] = np.zeros((parameters["W2"].shape[0],parameters["W2"].shape[1]))
    v["db2"] = np.zeros((parameters["b2"].shape[0],parameters["b2"].shape[1]))
    s["dW2"] = np.zeros((parameters["W2"].shape[0],parameters["W2"].shape[1]))
    s["db2"] = np.zeros((parameters["b2"].shape[0],parameters["b2"].shape[1]))
    
    v["dW3"] = np.zeros((parameters["W3"].shape[0],parameters["W3"].shape[1]))
    v["db3"] = np.zeros((parameters["b3"].shape[0],parameters["b3"].shape[1]))
    s["dW3"] = np.zeros((parameters["W3"].shape[0],parameters["W3"].shape[1]))
    s["db3"] = np.zeros((parameters["b3"].shape[0],parameters["b3"].shape[1]))
    
    
    return v, s


# ## 7. Activation Function

# In[ ]:


# Sigmoid Function Defination
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


# In[ ]:


_x = np.linspace(-5, 5, 40)
plt.plot(sigmoid(_x))
plt.plot(sigmoid_derivative(sigmoid(_x)))
plt.grid()


# ## 8. Forward Propagation with Dropout

# In[ ]:


def forward_propagation(X, parameters, keep_prob):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = (np.dot(W1, X.T) + b1).T
    A1 = sigmoid(Z1)
    
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = (D1 < keep_prob).astype(int)
    A1 = A1 * D1
    A1 = A1 / keep_prob
    
    Z2 = (np.dot(W2, A1.T) + b2).T
    A2 = sigmoid(Z2)
    
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = (D2 < keep_prob).astype(int)
    A2 = A2 * D2
    A2 = A2 / keep_prob
    
    Z3 = (np.dot(W3, A2.T) + b3).T
    A3 = sigmoid(Z3)
    
    assert(A3.shape == (X.shape[0], 10))
    
    cache = {
        "Z1" : Z1,
        "A1" : A1,
        "Z2" : Z2,
        "A2" : A2,
        "Z3" : Z3,
        "A3" : A3,
        "D1" : D1,
        "D2" : D2
    }

    return A3, cache


# ## 9. Cost Function

# In[ ]:


def compute_cost(A3, Y):
    
    m = Y.shape[0] # number of example
    
    logprobs = np.multiply(Y, np.log(A3)) + np.multiply((1 - Y), np.log(1 - A3))
    cost = - np.sum(logprobs) / m
    
    cost = float(np.squeeze(cost))
    
    assert(isinstance(cost, float))
    
    return cost


# ## 10. Backward Propagation with Dropout

# In[ ]:


def backward_propagation(parameters, cache, X, Y, keep_prob):
    
    m = Y.shape[0]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    
    D1 = cache["D1"]
    D2 = cache["D2"]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    dZ3 = (A3 - Y)
    dW3 = (1 / m) * np.dot(dZ3.T, A2)
    db3 = (1 / m) * np.sum(dZ3, keepdims=True)

    dA2 = np.dot(dZ3, W3)
    dA2 = dA2 * D2
    dA2 = dA2 / keep_prob
    
    dZ2 = np.multiply(dA2, sigmoid_derivative(A2))
    dW2 = (1 / m) * np.dot(dZ2.T, A1)
    db2 = (1 / m) * np.sum(dZ2, keepdims=True)
    
    dA1 = np.dot(dZ2, W2)
    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob
    
    dZ1 = np.multiply(dA1, sigmoid_derivative(A1))
    dW1 = (1 / m) * np.dot(dZ1.T, X)
    db1 = (1 / m) * np.sum(dZ1, keepdims=True)
    
    grads = {"dW1": dW1, 
             "db1": db1, 
             "dW2": dW2, 
             "db2": db2, 
             "dW3": dW3, 
             "db3": db3
            }
    
    return grads
    


# ## 11. Update Parameters

# In[ ]:


def update_parameters(parameters, grads, learning_rate):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"]
    
    W1 = W1 - (learning_rate * dW1)
    b1 = b1 - (learning_rate * db1)
    W2 = W2 - (learning_rate * dW2)
    b2 = b2 - (learning_rate * db2)
    W3 = W3 - (learning_rate * dW3)
    b3 = b3 - (learning_rate * db3)
    
    parameters = {"W1": W1, 
                  "b1": b1, 
                  "W2": W2, 
                  "b2": b2, 
                  "W3": W3, 
                  "b3": b3
                 }
    
    return parameters


# ## 12. Update parameter with Adam

# In[ ]:


def update_parameters_with_adam(parameters, grads, v, s, learning_rate, t = 2, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    
    v_corrected = {}
    s_corrected = {}
    L = len(parameters) // 2
    
    for l in range(L):

        v["dW" + str(l+1)] = (beta1 * v["dW" + str(l+1)]) + ((1 - beta1) * grads["dW" + str(l+1)])
        v["db" + str(l+1)] = (beta1 * v["db" + str(l+1)]) + ((1 - beta1) * grads["db" + str(l+1)])

        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - np.power(beta1, t))

        s["dW" + str(l+1)] = (beta2 * s["dW" + str(l+1)]) + ((1 - beta2) * np.power(grads["dW" + str(l+1)], 2))
        s["db" + str(l+1)] = (beta2 * s["db" + str(l+1)]) + ((1 - beta2) * np.power(grads["db" + str(l+1)], 2))

        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - np.power(beta2, t))

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate *                                         (v_corrected["dW" + str(l+1)]/ np.sqrt(s_corrected["dW" + str(l+1)] + epsilon))
        
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate *                                         (v_corrected["db" + str(l+1)]/ np.sqrt(s_corrected["db" + str(l+1)] + epsilon))

    return parameters, v, s


# ## 12. Predict

# In[ ]:


def predict(X, parameters, keep_prob):
    
    m = X.shape[0]
    
    A3, cache = forward_propagation(X, parameters, keep_prob = 1.0)
    
    return A3


# ## 13. Split Dataset in Batches

# In[ ]:


def split_in_mini_batches(X, Y, mini_batch_size = 128):
    
    m = X.shape[0]

    mini_batches = []

    len = int(X.shape[0]/mini_batch_size)

    for k in range(0, len):
        mini_batch_x = X[mini_batch_size * k : mini_batch_size * (k + 1), :]
        mini_batch_y = Y[mini_batch_size * k : mini_batch_size * (k + 1), :]

        assert(mini_batch_x.shape == (mini_batch_size, 784))
        assert(mini_batch_y.shape == (mini_batch_size, 10))

        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_x = X[mini_batch_size * (k + 1) : m, :]
        mini_batch_y = Y[mini_batch_size * (k + 1) : m, :]
        
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
        
    return mini_batches


# ## 14. Neural Network Model

# In[ ]:


def nn_model(X, Y, n_h, learning_rate, num_iterations, keep_prob, mini_batch_size, print_cost=False):

    np.random.seed(3)

    cost_per_iter = []
    
    dev_accuracy_arr = []
    train_accuracy_arr = []
    
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]
    
    # Initialize parameters
    parameters = initialise_parameter(n_x, n_h, n_y)
    v, s = initialise_adam(parameters)
    
    mini_batches = split_in_mini_batches(X, Y, mini_batch_size)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        
        for m_x, m_y in mini_batches:
        
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A3, cache = forward_propagation(m_x, parameters, keep_prob)

            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = compute_cost(A3, m_y)
            
            cost_per_iter.append(cost)

            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = backward_propagation(parameters, cache, m_x, m_y, keep_prob)

            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, learning_rate)
            
            # Print the cost every 1000 iterations
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

            train_prediction = (train_y != np.array(predict(train_x, parameters, keep_prob).argmax(axis=1)).T).astype(int)
            dev_prediction = (dev_y != np.array(predict(dev_x, parameters, keep_prob).argmax(axis=1)).T).astype(int)

            dev_accuracy_arr.append(100 - np.mean(dev_prediction) * 100)
            train_accuracy_arr.append(100 - np.mean(train_prediction) * 100)
    
    test_prediction = np.vstack((np.arange(1,28001), predict(test_x, parameters, keep_prob = 1.0).argmax(axis=1).T)).T
    data_to_submit = pd.DataFrame(test_prediction, columns = ['ImageId','Label']) 
    
    output = {
        "cost" : cost_per_iter[-1],
        "parameters" : parameters,
        "cost_per_iter" : cost_per_iter,
        "train_accuracy_arr" : train_accuracy_arr,
        "dev_accuracy_arr" : dev_accuracy_arr,
        "data_to_submit" : data_to_submit
    }
    
    return output


# In[ ]:


models = {}
learning_rates = [0.001]

for i in learning_rates:
    print ("learning rate is: " + str(i))
    
    models[str(i)] = nn_model(train_x, train_ch_y, n_h = [500, 50], learning_rate = i,                               num_iterations = 20, keep_prob = 0.8, mini_batch_size = 64, print_cost=False)
    
    print ("Cost is: " + str(models[str(i)]["cost"]))
    print("train accuracy: {} %".format(models[str(i)]["train_accuracy_arr"][-1]))
    print("dev accuracy: {} %".format(models[str(i)]["dev_accuracy_arr"][-1]))
    print ("-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["cost_per_iter"]), label= str(i))

plt.ylabel('cost')
plt.xlabel('iterations')
plt.legend(loc='upper right')
plt.grid()
plt.show()


# In[ ]:


plt.plot(np.squeeze(models["0.001"]["dev_accuracy_arr"]), label= "Dev Accuracy")
plt.plot(np.squeeze(models["0.001"]["train_accuracy_arr"]), label= "Train Accuracy")
plt.legend(loc='upper left')
plt.grid()
plt.show()


# In[ ]:


models["0.001"]["data_to_submit"].to_csv('csv_to_submit.csv', index = False)

