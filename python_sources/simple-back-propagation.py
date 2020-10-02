#!/usr/bin/env python
# coding: utf-8

# ## In this notebook I will implement simple 2 layer neural network from scratch. 
# 
# ## Please UPVOTE if you like this kernel

# In[ ]:


import numpy as np
from tqdm import tqdm_notebook
from IPython.display import Image

import warnings
warnings.filterwarnings('ignore')


# ### Neural Network 

# In[ ]:


Image(filename='/kaggle/input/imagesforkernel/classification_kiank.png') 


# ### Forward Propagation and cost function

# **Mathematically**:
# 
# For one example $x^{(i)}$:
# $$z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1]}\tag{1}$$ 
# $$a^{[1] (i)} = \tanh(z^{[1] (i)})\tag{2}$$
# $$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2]}\tag{3}$$
# $$\hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)})\tag{4}$$
# $$y^{(i)}_{prediction} = \begin{cases} 1 & \mbox{if } a^{[2](i)} > 0.5 \\ 0 & \mbox{otherwise } \end{cases}\tag{5}$$
# 
# Given the predictions on all the examples, you can also compute the cost $J$ as follows: 
# $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$

# ### Back-Propagation Calculation

# In[ ]:


Image(filename='/kaggle/input/imagesforkernel/grad_summary.png') 


# - n_x: the size of the input layer
# - n_y: the size of the output layer
# 
# - X : Input data
# - h : hidden layer size
# - y : target

# In[ ]:


class NonLinearFunctions:
    @staticmethod
    def sigmoid(z):
        return 1.0/(1+np.exp(-z))
    
    @staticmethod
    def relu(z):
        return max(0,z)
    
    @staticmethod
    def leaky_relu(z,lb=0.01):
        return max(lb*z,z)
    
    @staticmethod
    def tanh_func(z):
        return np.tanh(z)


# In[ ]:


class NeuralNet(NonLinearFunctions):
    '''
    input: 
        X : input features
        y : targets
        h : number of hidden units
        lr : learning rate (default: 0.01)
    '''
    def __init__(self,X,y,h,lr = 1e-2):
        self.X = X
        self.y = y
        self.h = h
        self.lr = lr
        
    def layer_size(self):
        n_x = self.X.shape[0]
        n_h = self.h
        n_y = self.y.shape[0]
        return (n_x,n_h,n_y)
    
    def initialize_parameters(self):
        (n_x,n_h,n_y) = self.layer_size()
        W1 = np.random.randn(n_h,n_x)*1/np.sqrt(n_x ** 1)
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h)*1/np.sqrt(n_x ** 1)
        b2 = np.zeros((n_y,1))
        
        parameters = {'W1':W1,
                     'b1':b1,
                     'W2':W2,
                     'b2':b2}
        return parameters
    
    def forward(self,parameter):
        W1 = parameter['W1']
        b1 = parameter['b1']
        W2 = parameter['W2']
        b2 = parameter['b2']
        
        Z1 = np.dot(W1,self.X)+b1
        A1 = NonLinearFunctions.sigmoid(Z1)
        Z2 = np.dot(W2,A1)+b2
        A2 = NonLinearFunctions.sigmoid(Z2)
        
        cache = {'Z1':Z1,
                'A1':A1,
                'Z2':Z2,
                'A2':A2}
        
        return A2,cache
    
    def compute_cost(self,parameters):
        m = self.y.shape[1]
        A2,_ = self.forward(parameters)
        logprob = np.multiply(np.log(A2),self.y)+np.multiply(np.log(1-A2),(1-self.y))
        cost = -np.sum(logprob)/m
        cost = float(np.squeeze(cost))
        
        return cost
    
    def backward(self,parameter):
        m = self.X.shape[1]
        
        W1 = parameter['W1']
        b1 = parameter['b1']
        W2 = parameter['W2']
        b1 = parameter['b2']
        
        _,cache = self.forward(parameter)
        A1 = cache['A1']
        A2 = cache['A2']
        
        dZ2 = A2-self.y
        dW2 = 1/m*(np.dot(dZ2,A1.T))
        db2 = 1/m*(np.sum(dZ2,axis=1, keepdims=True))
        dZ1 = np.multiply(np.dot(W2.T,dZ2),(1-np.power(A1,2)))
        dW1 = 1/m*(np.dot(dZ1,self.X.T))
        db1 = 1/m*(np.sum(dZ1,axis=1, keepdims=True))
        
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
    
        return grads
    
    def update_parameters(self,grads):
        parameter = self.initialize_parameters()
        W1 = parameter['W1']
        b1 = parameter['b1']
        W2 = parameter['W2']
        b2 = parameter['b2']
                
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        
        W1 = W1 - self.lr*dW1
        b1 = b1 - self.lr*db1
        W2 = W2 - self.lr*dW2
        b2 = b2 - self.lr*db2
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters
    
    def train_model(self,iteration = 100,print_cost = True,print_interval=10):
        '''
        input: 
            iteration : number of iterations
            print_cost : True if you want to print cost function (default: True)
            print_interval : interval in which cost will be printed
        output: 
            cost : list of cost values
        '''
        costs = []
        parameters = self.initialize_parameters()
        for i in tqdm_notebook(range(1,iteration+1)):
            A2,cache = self.forward(parameters)
            cost = self.compute_cost(parameters)
            grads = self.backward(parameters)
            parameters = self.update_parameters(grads)
            costs.append(cost)
            if print_cost and i%print_interval ==0:
                print(f'Cost for iteration {i}: {cost}')
        return costs


# ### Test NeualNetwork

# In[ ]:


from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:


cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target


# In[ ]:


y = y.reshape(-1,1)


# In[ ]:


y -= y.mean()


# In[ ]:


Model = NeuralNet(X,y,128,lr=1e-5)


# In[ ]:


cost = Model.train_model(iteration=500)


# In[ ]:


X = range(len(cost))
import matplotlib.pyplot as plt


# In[ ]:


plt.plot(X,cost)
plt.title('Cost vs number of iteration')
plt.show()


# In[ ]:




