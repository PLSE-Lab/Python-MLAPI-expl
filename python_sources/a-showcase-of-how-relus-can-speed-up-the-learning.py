#!/usr/bin/env python
# coding: utf-8

# # Neural network and activation functions. A showcase of the differences between tanh, sigmoid and ReLU as activation functions.
# 
# ### In this notebook I am going to code a simple neural network entirely in numpy, note that normally you will never do this (the same way you will never code your own hash table), but It is an interesting way to understand what happens in neural networks.
# 
# ### It is worth noticing that both the training examples and the labels will be organized by COLUMNS, not rows, this means that each sample will be in one column, not in one row, the only reason to do so is to make calculations easier.
# 
# ### Also this is based in the notations of the course deeplearning.ai, by Andrew Ng, which I highly recommend.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_raw = pd.read_csv("../input/mnist_train.csv")
test_raw = pd.read_csv("../input/mnist_test.csv")


# In[ ]:


train_raw.head()


# ## A quick exploration of the dataset

# In[ ]:


train_raw.describe()


# ## For the purposes of this notebook, I am only going to write a network which can identify the digits 0 and 1, so I will drop every other example, this is done just to simplify calculations.
# 
# ## Also, I will NOT be using mini batches, instead I will feed all the training examples in each epoch, so I will only keep 1000 training examples and 500 test samples. No validation data will be used (again, this is just to showcase the activation functions).

# In[ ]:


train = train_raw[(train_raw["label"] == 0) | (train_raw["label"] == 1)]
test =  train_raw[(train_raw["label"] == 0) | (train_raw["label"] == 1)]

train = train.head(1000)
test = test.head(500)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


y_train = train["label"]
y_test = test["label"]
del train["label"]
del test["label"]
X_train = train
X_test = test


# ## We have to normalize our data, in the case of the MNIST dataset, this simply means dividing by 255, a more formal explanations is that 
# 
# ## $$X = \frac{X - min(x)}{max(x) - min(x)}$$

# In[ ]:


X_train /= 255
X_test /= 255


# In[ ]:


X_train.describe()


# ## Now we want to ensure that our y values have shape (number_of_examples, 1) instead of (number_of_examples, )

# In[ ]:


y_train = y_train.values
y_test = y_test.values
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)


# ## Remember, our columns will be our training examples, so we need to do the transpose of each matrix here

# In[ ]:


X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T


# ## We will use $m$ to denote our number of training examples

# In[ ]:


m = X_train.shape[1]


# ## And here is where the fun begins, we are going to define our three functions. 
# 
# ## The sigmoid function is defined as  
#  # $$sigmoid(z) = \frac{1}{1 + e ^ {-z}}$$
# ## The relu function is much simpler, as it is the maximum between the input and zero
# ## The tanh function is defined as  
# #  $$tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

# In[ ]:


def sigmoid(z):
    output = 1 / (1+np.exp(-z))
    return output

def relu(z):
    return np.maximum(z, 0)

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


# ## It would be good to display our functions... lets plot them

# In[ ]:


import matplotlib.pylab as plt
def plot_function(function, title="sigmoid"):
    x = np.arange(-7, 7, 0.01)
    y = function(x)
    
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()
    
plot_function(sigmoid, "sigmoid")    
plot_function(tanh, "tanh")
plot_function(relu, "relu")


# ## Now we will provide another function to compute the derivative, and here is the whole deal. 
# 
# ## The ReLU function will have normally quite a large derivative, because it is defined as 0 if x <= 0 and 1 otherwise... however in the case of the sigmoid function, the derivative is
# # $$ \frac{e^z - e^{-z}}{e^z + e^{-z}} * (1 - \frac{e^z - e^{-z}}{e^z + e^{-z}}) $$
# ## Or in simpler terms
# # $$ sigmoid(x) * (1-sigmoid(x)) $$
# ## Now, the problem with this is that the derivative of the sigmoid will never, ever be larger than 0.25... and this is going to make things... slow, because ultimately we will use the derivative to uptate our weights and biases, so if the derivative is small, the update will be small, and the learning will be slow.
# 
# ## Finally the tanh function derivative is 
# # $$ 1 - (\frac{e^z - e^{-z}}{e^z + e^{-z}})^ 2 $$
# ## Or in simpler terms
# # $$ 1 - tanh^2(x) $$
# 

# In[ ]:


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu_derivative(z):
    z[z<=0] = 0
    z[z>0] = 1
    return z

def tanh_derivative(z):
    return 1 - (np.power(tanh(z), 2))

def compute_derivative(z, function="sigmoid"):
    if function == "sigmoid":
        return sigmoid_derivative(z)
    elif function == "relu":
        return relu_derivative(z)
    elif function == "tanh":
        return tanh_derivative(z)


# ## Now, it would be VERY interesting to plot the derivative function of each of our activation functions. Ultimately we are interested in seeing how the derivatives change their values.
# 
# ## Pay close attention to the y axis... you will notice that the sigmoid never actually goes over 0.25 while tanh reaches 1.0 in certain areas... ReLU on the other end, has a larger derivative

# In[ ]:


plot_function(sigmoid_derivative, "sigmoid derivative")
plot_function(tanh_derivative, "tanh derivative")
plot_function(relu_derivative, "relu derivative")


# 

# ## Have a closer look at the implications of this. Remember that in gradient descent, we will update our weights and biases as
# 
# ## $$ W_l = W_l -  \alpha * \frac{\partial cost}{\partial W_l} $$
# ## $$ b_l = b_l -  \alpha * \frac{\partial cost}{\partial b_l} $$
# 
# ## Where $ l $ represents the layer number we want to calculate...  this essentially means that a small 
# ## $$ \frac{\partial cost}{\partial b_l} $$
# ## or a small 
# ## $$ \frac{\partial cost}{\partial W_l}  $$ 
# ## means a slow learning, conversely, a large value of any of those derivatives will mean a quick learning... at this point you should already have an intuition about why the sigmoid function is not great for learning.
# 
# 

# In[ ]:


def initialize_params():
    W1 = np.random.randn(256, 784) * 0.01
    b1 = np.zeros((256, 1))
    W2 = np.random.randn(1, 256) * 0.01
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2


# In[ ]:


W1, b1, W2, b2 = initialize_params()
print("y_train shape", y_train.shape)
print("W1 shape", W1.shape, "X_train (A1) shape", X_train.shape,  "b1 shape", b1.shape)
A2 = np.dot(W1, X_train) + b1
print("W2 shape", W2.shape, "Z2 shape", A2.shape, "b2 shape", b2.shape)
print("----------------------------------------------------------------------")
print("We will doo W1 x X_train = Z1 ", W1.shape, "x" , X_train.shape, "=", W1.shape[0],",", X_train.shape[1])
print("\tThen we apply a activation function to Z1, getting A1... A1 shape will be the same as Z1, so ", X_train.shape)
print("Finally we will do W2 x Z1", W2.shape, "x", A2.shape, "=", W2.shape[0], ",", A2.shape[1])
print("\tThen we apply the sigmoid function to Z2, and we will get A2, which is our y_hat")


# ## Now, we just need to implement the forward pass, which is pretty much a prediction. Notice the ```activation``` argument, so later we can control it.

# In[ ]:


def forward_pass(W1, b1, W2, b2, X, m, activation="sigmoid"):
    Z1 = np.dot(W1, X) + b1
    if activation == "sigmoid":
        A1 = sigmoid(Z1)
    elif activation == "relu":
        A1 = relu(Z1)
    elif activation == "tanh":
        A1 = tanh(Z1)
        
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    y_hat = A2

    return y_hat, Z1, A1


# ## We need to be able to calculate our cost over time, this will be done using the following formula
# # $$ \frac{-1}{m}  \sum\limits_{i = 0}^{m} [ y_i * log(\hat{y_i}) + (1-y_i) * log(1-\hat{y}_i)]$$

# In[ ]:


def calculate_cost(y, y_hat):
    m = y.shape[1]
    log_probs = np.dot(y, np.log(y_hat.T)) + np.dot((1-y), np.log(1-y_hat.T))
    cost = (-1/m) * np.sum(log_probs)
    cost = np.squeeze(cost)
    return cost


# In[ ]:


y_hat, Z1, A1 = forward_pass(W1, b1, W2, b2, X_train, m)
cost = calculate_cost(y_train, y_hat)
print("Cost", cost)


# ## Time to implement the backward pass, notice the ```activation``` function, of course this assumes that the ```activation``` argument will be the same one that was passed in the ```forward_pass``` function

# In[ ]:


def backward_pass(X, y, Z1, A1, W2, y_hat, activation="sigmoid"):
    m = y.shape[1]
    dZ2 = y_hat - y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * compute_derivative(Z1, function=activation)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2


# In[ ]:


def model(X, y, iterations=100, activation="sigmoid", lr=0.001):
    W1, b1, W2, b2 = initialize_params()
    costs = []
    for epoch in range(0, iterations+1):
        y_hat, Z1, A1 = forward_pass(W1, b1, W2, b2, X, m, activation=activation)
        
        
        
        dW1, db1, dW2, db2 = backward_pass(X, y, Z1, A1, W2, y_hat, activation=activation)
        
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2
        
        if epoch % 500 == 0:
            current_cost = calculate_cost(y, y_hat)
            costs.append(current_cost)
            print("Cost at epoch", epoch, "is", current_cost)
    return costs


# In[ ]:


sigmoid_costs = model(X_test, y_test, iterations = 6000, activation="sigmoid")


# In[ ]:


tanh_costs = model(X_test, y_test, iterations = 6000, activation="tanh")


# In[ ]:


relu_costs = model(X_test, y_test, iterations = 6000, activation="relu")


# ## Finally, we plot the evolution of our cost with the different activation functions. Of course we want our cost to go down quickly and in an stable manner. 

# In[ ]:


x = np.arange(0, 6001, 500)
plt.figure(figsize=(12, 7))
plt.plot(x, sigmoid_costs)
plt.plot(x, tanh_costs)
plt.plot(x, relu_costs)
plt.legend([ "Sigmoid cost", "Tanh cost", "ReLU cost"])
plt.show()


# In[ ]:




