#!/usr/bin/env python
# coding: utf-8

# # Aim of the Project
# 
# In this project, we  experiment with two models to classify MNIST data. First, we train a logistic regression model which shows classification accuracy of around $90\%$ in the test dataset and $85\%$ in the submission dataset. Second, we train a four layer neural network model on MNIST training dataset. As of now, the classification accuracy on the training dataset is around $98\%$ and on the test dataset is around $95\%$.
# 
# This main purpose of the project is to understand the backpropagation algorithm. The neural network model has been coded from scratch and may be extremely inefficient in practice. I plan to do further experiments on hyperparameter tuning, regularization, effect of other activitation functions, cost functions and performance evaluation on other network architectures.
# 
# 
# 

# In[ ]:


#!conda env list


# In[ ]:


#import sys
#!conda install --yes --prefix {sys.prefix} scikit-learn


# # Packages needed

# In[ ]:


# PACKAGE
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import cbook
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#import seaborn as sns


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# # Kaggle dataset

# In[ ]:


# input methods
import os
print(os.listdir("../input"))


# In[ ]:


training_data = pd.read_csv("../input/train.csv")
training_data = training_data.values


# # Explanation of the dataset dimensions
# 
# <code>training_data</code> is a numpy array with columns representing features and rows as the instances of the training dataset. The first column of the <code>training_data</code> is the label column.

# # Dividing dataset into training, dev and test dataset.

# In[ ]:


length = training_data.shape[0]
length
training_length = round(0.7*(length))
#dev_length = round(0.8*(length))
x_train = training_data[:training_length,1:].T
y_train = training_data[:training_length,0]

#x_dev = training_data[training_length+1:dev_length,1:].T
#y_dev = training_data[training_length+1:dev_length,0]

x_test = training_data[training_length+1:,1:].T
y_test = training_data[training_length+1:,0]



y_train = y_train.reshape(1,len(y_train))
#y_dev = y_dev.reshape(1,len(y_dev))
y_test = y_test.reshape(1,len(y_test))


# # Data Rescaling and Standardization
# ## Rescaling

# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test) 


# ## Standardization
# 
# Data preprocessing such that the mean of the data is $0$ and standard deviation is $1$.

# In[ ]:


#scaler_train = StandardScaler().fit(x_train)
#scaler_test = StandardScaler().fit(x_test)
#x_train = scaler_train.transform(x_train)
#x_test = scaler_test.transform(x_test)


# The standardization has been commented out since this resulted in deterioration of the performance of the logistic regression.

# Here is an example of a training set image.

# In[ ]:



img = x_train[:,3]
img = img.reshape(28,28)
plt.imshow(img,cmap = 'gray')
plt.show()
#y_train[:,3]


# # Exploratory Data Analysis

# In[ ]:


count_in_labels = []
for i in range(10):
    idx = y_train == i
    count_in_labels.append(len(y_train[idx]))
    print("Count in label %d is %d"%(i,count_in_labels[i]))


# All the labels have approximately same amount of instances.

# # Experimenting with Logostic Regression

# ## Training the dataset

# In[ ]:


logisticregsr = LogisticRegression(multi_class='multinomial', solver = 'lbfgs')
y_train_l = y_train.ravel()
logisticregsr.fit(x_train.T,y_train_l)


# In[ ]:


pred = logisticregsr.predict(x_test.T)
y_test_check = y_test.ravel()
y_train_check = y_train.ravel()
#y_test_check.shape
score_train = logisticregsr.score(x_train.T,y_train_check)*100
print("The accuracy on train data is",score_train,"%")
score_test = logisticregsr.score(x_test.T,y_test_check)*100
print("The accuracy on test data is",score_test,"%")


# # Accuracy
# 
# The accuracy on the train data is around $94\%$ and on the test data is about $92\%$.
# 
# # Confusion Matrix
# 

# In[ ]:


c_matrix = confusion_matrix(y_test_check,pred)


# In[ ]:


print(c_matrix)


# The submission set resulted in an accuracy of around 85.2%

# # Experimenting with Neural Network
# 
# ## Transforming the output in the dataset
# 
# Each entry in <code>y_train</code> (or <code>y_test</code>) is a number in the range of 0 to 9 which is the label of the corresponding instance in x_train. The function <code>transform_output</code> takes such a label $y$ as input and outputs a binary column vector with $1$ in the $y^{th}$ position and $0$ in rest of the entries. 
# For example, if an instance has label $5$, then the corresponding transformed label vector is ${(0,0,0,0,0,1,0,0,0,0)}^T$.
# 
# ## Building the neural network
# 
# We define a network class which has the information of size of the layers and corresponding initializing weights and biases.
# 
# Take a proper look at the indices. It can get confusing. if there are three layers, then there will be two weight matrices and biases.
# 
# ### Cost function
# 
# Let $y$ be the transformed label vector of an arbitrary instance and $a$ be the output of the model. Then the cost function $C$ is defined as : 
# 
# $$ C(y,a) =  \frac{{||y-a||}^{2}}{2} $$
# 
# Over the whole training set, the (vectorized) cost function is an average of cost functions of all the instances,
# 
# $$ C(Y,A) = \frac{1}{2m}\sum\limits_{i=1}^{m}C(y^{(i)},a^{(i)})$$
# 
# where $Y =[y^{(1)}\ ...\ y^{(m)}]$, $A = [a^{(1)}\ ...\ a^{(m)}]$ and $y^{(i)}$ and $a^{(i)}$ are the transformed label vector and output of the model respectively. Observe that both $y^{(i)}$ and $a^{(i)}$ are column vectors.
# 
# One important thing to note that $Y$ is given and $A$ is computed by the model. Therefore, the actual variables in the cost function are actually the weights and biases of the neural networks. 
# 
# ### He Initialization
# 
# ### Activation function
# 
# We use RELU as the activation function for hidden layers and sigmoid for probability estimation for output layer.
# 
# ### Feedforward 
# 
# Given a neural network with its set of weights, biases and input vector $x$, feedforward computes the prediction vector $a$ (where value at each entry i is the probability that x belongs to the class i). The computation at layer $l$ is 
# $$ z^{[l]} = w^{[l]}.a^{[l-1]} +b^{[l]} $$
# $$ a^{[l]} = \sigma(z^{[l]})$$
# 
# I have defined two functions <code>feedforward</code> and <code>feedforward_output</code>. <code>feedforward</code>, along with computing $a$, also caches $z$ and $a$ so that they can be used in backpropagation. The other function just computes $a$.
# 
# ### Backpropagation
# 
# Backpropagation is used to compute $\frac{\partial C}{\partial w^{[l]}}$ and $\frac{\partial C}{\partial b^{[l]}}$ for all layers $l$ in the neural network. <code>backpropagation</code> returns two lists of the partial derivatives of all the weights and biases($dw$ and $db$).
# 
# Computation of the partial derivatives make use of the following formulae:
# 
# $$ \frac{\partial C}{\partial w^{[l]}} =  \frac{\partial C}{\partial z^{[l]}}. \frac{\partial z^{[l]}}{\partial w^{[l]}} $$
# 
# $$  \frac{\partial C}{\partial b^{[l]}} =  \frac{\partial C}{\partial z^{[l]}}$$
# 
# In the code, we represent $\frac{\partial C}{\partial w^{[l]}}$ as $dw$, $\frac{\partial C}{\partial b^{[l]}}$
# as db and  $\frac{\partial C}{\partial z^{[l]}}$ as $dz$.
# 
# In the vectorized form (i.e over all inputs of the batch), dw and db are computed as:
# 
# $$ dw^{[l]} =  \frac{1}{m}dz.{A^{[t]}}^T$$
# $$ db^{[l]} = \frac{1}{m}.\sum\limits_{i=1}^{m} dz^{[l]}[:,i]$$
# 
# ### compute_dZ function
# 
# $dz^{[l]}$ is computed using the compute_dZ function. 
# In the vectorized form, it is computed using the following equation: 
# 
# $$dz^{[L]} = \nabla_a C*\sigma'(z^{[L]}) $$
# $$dz^{[l]} = {w^{[l+1]}}^T.dz^{[l+1]}*\sigma'(z^{[l]})$$
# 
# Here $*$ is a broadcast operation.
# 
# Therefore, <code>compute_dZ</code> takes as arguments, training data (<code>batch_x,batch_y</code>), $l$ (current layer), $z$ values (stored in <code>Z_cache</code>), and $dz^{l+1}$ (as <code>dZ</code>).
# 
# ### Exponentially Weighted average
# 
# Given a set of points $\lbrace a_1,...,a_n\rbrace$, moving average $\lbrace v_1,...,v_n\rbrace$ is computed as follows:
# 
# <ul>
# <li> Initialize $v_1 = 0$.</li>
# <li> 
# For $i = 2$ to $n$:
# <ul>
#    <li>$ v_i = \beta v_{i-1} + (1 - \beta)a_i $</li>
# </ul>        
# </li>
# </ul> 
# 
# where $\beta\in (0,1)$.
# 
# The set $\lbrace v_1,v_2,...,v_n\rbrace$ is tolerant to noise in the original dataset.
# 
# ### Gradient Descent with Moment
# 
# We use the idea of exponentially weighted average to gradient descent. For every iteration, we use backpropagation to compute <code>dw</code> and <code>db</code> and then compute the exponentially weighted average using the computed gradients.
# <ul>
# <li> Compute <code>dw</code> and <code>db</code> using <code>backpropagation</code>.</li>
# <li> <code>vdw</code> $=\beta$<code>vdw</code> $+(1-\beta)$<code>dw</code></li>
# <li> <code>vdb</code> $=\beta$<code>vdb</code> $+(1-\beta)$<code>db</code></li>
# </ul>
# 
# Setting $\beta = 0.9$ works as a pretty robust value. Using Gradient Descent with moment leads to faster convergence.
# 
# ### Update function
# 
# 
# <code>update</code> function calls the backpropagation function to compute the gradients $dw$ and $db$, and updates the weights and biases of the neural network by taking exponentially weighted average.
# 
# 
# ### Minibatch Gradient Descent 
# 
# Minibatch Gradient Descent function is implemented as SGD in the code which takes in as arguments training data, batch_size, no_of_epoch and eta as parameters and perform the procedure outlined in the process outline. 
# 
# Process outline:
# 
# For no_of_epoch times, do the following :  
# <ul>
# <li>Randomly shuffle the training data. </li>
# <li> Seperate the training_data into <code>x_train_shuffled</code> and <code>y_train_shuffled</code>.</li>
# <li>Divide the shuffled training data (<code>x_train_shuffled</code> and <code>y_train_shuffled</code>) into batches of specified <code>batch_size</code> and store the batches in the lists <code>batches_x</code> and <code>batches_y</code>.</li>
# 
# <li>For each (<code>batch_x</code>,<code>batch_y</code>) in <code>zip(batches_x,batches_y)</code>, use the tuple to update the parameters (i.e call the update function which in turn performs the backpropagation and computes gradient descent with moment).</li>
# 
# <li>Completion of the whole dataset marks the end of one epoch.</li>
# </ul>
# 
# <code>batch_size, no_of_epoch</code> and <code>eta</code> are the hyperparameters of the algorithm.

# In[ ]:


def transform_output(y):
    l = y.shape[1]
    ty = np.zeros((10,l))
    for i in range(l):
        s = y[0,i].astype(int)
        # print(type(s))
        ty[s,i] = 1
    return ty


# In[ ]:


def function(z):
    return np.maximum(0.01*z,z)


# In[ ]:


def function_prime(z):
    a = np.greater(z,0.01*z)
    a = a.astype(int)
    a[a==0] = 0.01
    return a


# In[ ]:


def cost_function(y,a,m):
    b = y-a
    s = (1/(2*m))*(np.dot(b.T,b))
    l = s.shape[1]
    dsum = 0
    for i in range(l):
        dsum = dsum + s[i,i]
    return dsum


# In[ ]:


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# In[ ]:


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


# In[ ]:


def grad_a_cost_function(y,a,m):
    return -(1/m)*(y-a)


# # Defining the Neural Network

# In[ ]:


class Network:
    # atype is the activation type; whether it is a RELU or sigmoid
    def __init__(self,sizes,atype):
        self.num_layers = len(sizes)
        self.weights = []
        if atype=="sigmoid":
            self.biases = [np.random.randn(y,1) for y in sizes[1:]]
            #self.biases = [np.zeros((y,1)) for y in sizes[1:]]
            #self.weights = [np.zeros((sizes[1],sizes[0]))]
            #self.weights = [np.zeros((sizes[1],sizes[0]))
            for x in range(1,len(sizes)):
                (self.weights).append(np.random.randn(sizes[x],sizes[x-1]))
        else:
            self.biases = [np.zeros((y,1)) for y in sizes[1:]]
            #self.weights = [np.zeros((sizes[1],sizes[0]))]
            for x in range(1,len(sizes)):
                (self.weights).append(np.random.randn(sizes[x],sizes[x-1])*np.sqrt(2/sizes[x-1]))
                
            
    def feedforward(self, batch_x):
        length = len(self.weights)
        a_cache=[]
        a = batch_x
        a_cache.append(a)
        z_cache=[] 
        for i in range(length-1):
            z = np.dot(self.weights[i],a) + self.biases[i]
            z_cache.append(z)
            a = function(z)
            a_cache.append(a)
        i=i+1
        z = np.dot(self.weights[i],a) + self.biases[i]
        z_cache.append(z)
        a = sigmoid(z)
        a_cache.append(a)
        return a_cache,z_cache
    
    def feedforward_output(self, x_test):
        length = len(self.weights)
        a = x_test
        for i in range(length-1):
            z = np.dot(self.weights[i],a) + self.biases[i]
            a = function(z)
            #print("z[",i,"] = ",z)
        i = i + 1
        z = np.dot(self.weights[i],a) + self.biases[i]
        #print("z[",i,"] = ",z)
        a = sigmoid(z)
        return a
    
    def compute_dZ(self,batch_x,batch_y,l,Z_cache,dZ):
        m = batch_x.shape[1]
        n = batch_x.shape[0]
        #y = batch[n-1,:]
        #y = y.reshape(1,len(y))
        # print(type(y))
        #print(y.shape)
        #y = transform_output(y)
        if l== self.num_layers - 2:
            #a = function(Z_cache[l])
            a = sigmoid(Z_cache[l])
            #return grad_a_cost_function(batch_y,a,m)*function_prime(Z_cache[l]) 
            return grad_a_cost_function(batch_y,a,m)*sigmoid_prime(Z_cache[l])
        else:
            return np.dot(self.weights[l+1].T,dZ)*function_prime(Z_cache[l])
    
    def backpropagation(self,batch_x,batch_y):
        dw = []
        db = []
        m = batch_x.shape[1]
        A_cache,Z_cache = self.feedforward(batch_x)
        L = self.num_layers
        dZ = []
        for l in range(L-2,-1,-1):
            dZ = self.compute_dZ(batch_x,batch_y,l,Z_cache,dZ)
            db.append((1/m)*np.sum(dZ,axis = 1,keepdims = True))
            dw.append((1/m)*np.dot(dZ,A_cache[l].T))
        db.reverse()
        dw.reverse()
        return dw,db 
        
    def update(self,batch_x,batch_y,eta,vdw,vdb):
        beta = 0.9
        dw,db = self.backpropagation(batch_x,batch_y)
        
        vdw = [beta*x +(1-beta)*y for (x,y) in zip(vdw,dw)]
        vdb = [beta*x +(1-beta)*y for (x,y) in zip(vdb,db)]
        for i in range(self.num_layers-1):
            #check(dw[i])
            self.weights[i] = self.weights[i] - eta*vdw[i]
            self.biases[i] = self.biases[i] - eta*vdb[i]
        
        return vdw,vdb

            
    # we assume that in the training data, rows represent the features of the training set and columns denote the 
    # instances of the training set.
    
    def SGD(self,x_train,y_train,batch_size,no_of_epoch,eta,x_test=None,y_test=None):
        training_data = x_train
        print(training_data.shape)
        print(y_train.shape)
        training_data = np.append(training_data,y_train,axis = 0)
        print(training_data.shape)
        # In the previous two lines, we have put x_train and y_train into one matrix: training data;
        #training_data = [[x_train],
        #                 [ytrain]]
        n = training_data.shape[0]
        training_size = training_data.shape[1]
        vdw = [np.zeros(y.shape) for y in self.weights]
        vdb = [np.zeros(y.shape) for y in self.biases]
        for i in range(no_of_epoch):
            np.random.shuffle(training_data.T)
            x_train_shuffled = training_data[:-1,:]
            y_train_shuffled = training_data[n-1,:]
            y_train_shuffled = y_train_shuffled.reshape(1,len(y_train_shuffled ))
            # print(type(y_train_shuffled))
            y_train_shuffled_transformed = transform_output(y_train_shuffled)
            batches_x = [x_train_shuffled[:,k:k + batch_size] for k in range(0,training_size,batch_size)]
            batches_y = [y_train_shuffled_transformed[:,k:k + batch_size] for k in range(0,training_size,batch_size)]
            for (batch_x,batch_y) in zip(batches_x,batches_y):
                vdw,vdb = self.update(batch_x,batch_y,eta,vdw,vdb)
            a = self.feedforward_output(x_train[:,1:10])
            m = a.shape[1]
            y = transform_output(y_train[:,1:10])
            cost = cost_function(y,a,m)
            print("The cost after epoch no.",i," is ",cost)
            #a = np.argmax(a,axis = 0)
            #s = score(y_train,a)
            #print(s)
        
        


# # Initialization of hyperparameters

# In[ ]:


n = Network([784,30,30,10],"RELU")
n.SGD(x_train,y_train,10,50,0.5)


# In[ ]:


img = x_test[:,7]


# In[ ]:


img = img.reshape(len(img),1)
A_test= n.feedforward_output(img)
A_test = np.argmax(A_test,axis=0)
print(img.shape)
img = img.reshape(28,28)
plt.imshow(img,cmap = 'gray')
plt.show()
print("The predicted output is", A_test)


# # Score

# In[ ]:


A_train = n.feedforward_output(x_train)
A_train = np.argmax(A_train,axis = 0)
#A.shape
A_train = A_train.reshape(1,len(A_train))


# In[ ]:


A_test = n.feedforward_output(x_test)
A_test = np.argmax(A_test,axis = 0)
A_test.shape
A_test = A_test.reshape(1,len(A_test))


# In[ ]:


def score(y,a):
    l = y.shape[1]
    return (1 - (np.count_nonzero(y-a)/l))*100


# In[ ]:


score_train = score(y_train,A_train)
score_test = score(y_test,A_test)
print("Accuracy on train data is ",score_train, "% and on test data is",score_test,"%")


# # References
# <ul>
#  <li>neuralnetworksanddeeplearning.com </li>
#   <li>deeplearning.ai Coursera Specialization</li>  
# </ul>

# In[ ]:




