#!/usr/bin/env python
# coding: utf-8

# # Why we are racing for smartest AI? 
# 
# #### As we know, we live in community, with every individuals facing a weakness. To specify, there is some kind of disorder named brain problem, or maybe in another term, some diseases affect the brain. 
# 
# #### In the middle school, I am the one of the most stupid students in class, ranked 34th out of 36. Got exam score of 10/100, which means I am far too inferior from the simple machine learning, even my in-class exercise who had 50% accuracy is far smarter than me. In the world full of intelligent system, it would not be colorful if all of them are smart.[](http://)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Here is a kind of AI that facing some brain problem!

# #### We will start modeling the deep learning using simple ANN. 
# 
# #### Showed skipped memory if we use the wrong approach, simulating a problem in the robot's brain. Then, "stupid" model will be the outcome.
# 
# #### Here is my modified Deep Learning course (Math 4072) exercise notebook, but instead of competing for best score, I am heading for worst score.

# ## Loading the necessary library
# 
# Yea, I think the worst is for not doing anything, but I wont replicate my real world stupidity. 

# In[ ]:


import struct
import numpy as np
from numpy.random import randn, seed
import matplotlib.pyplot as plt


# In[ ]:


def read_idx(filename):
    with open(filename, "rb") as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
#dont worry i wont swap the test and train    
x_train_ = read_idx("/kaggle/input/fashionmnist/train-images-idx3-ubyte")
y_train_ = read_idx("/kaggle/input/fashionmnist/train-labels-idx1-ubyte")
x_test_  = read_idx("/kaggle/input/fashionmnist/t10k-images-idx3-ubyte")
y_test_  = read_idx("/kaggle/input/fashionmnist/t10k-labels-idx1-ubyte")


# In[ ]:


print(x_train_.shape)  #data gambar untuk training
print(y_train_.shape)  #data label untuk training sebelum one-hot encoding


# ### Flatten, turn 2D into 1D array

# In[ ]:


x_train = []
for x in x_train_:
    x_train.append(x.flatten())


x_test = []
for x in x_test_:
    x_test.append(x.flatten())

print("x_train shape:",np.array(x_train).shape)
print("x_test shape:",np.array(x_test).shape)


# ### One hot encoding

# In[ ]:


y_train = np.eye(y_train_.max()+1)[y_train_]
y_test = np.eye(y_test_.max()+1)[y_test_]


# In[ ]:


print("y_train shape:",np.array(y_train).shape)
print("y_test shape:",np.array(y_test).shape)


# In[ ]:


# Dictionary of labels
label = {0: 'T-shirt/top',
         1: 'Trouser',
         2: 'Pullover',
         3: 'Dress',
         4: 'Coat',
         5: 'Sandal',
         6: 'Shirt',
         7: 'Sneaker',
         8: 'Bag',
         9: 'Ankle boot'}


# ## The most interesting Part
# 
# There is some function, you know $softmax$?
# 
# ![Source: wikipedia](https://wikimedia.org/api/rest_v1/media/math/render/svg/bdc1f8eaa8064d15893f1ba6426f20ff8e7149c5)
# Source: wikipedia
# 
# 
# 
# We will use $sinh$ for hidden layer, and $kepler's \  third \ law$ function for output layer. Lets see
# 

# In[ ]:


def softmax(s):
    exps = np.exp(s - np.max(s, axis = 1, keepdims = True))
    return exps/np.sum(exps, axis = 1, keepdims = True)

def sigmoid(s):
    return 1/(1+np.exp(-s))

def tanh(s):
    return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))

def linear(s):
    return s

def quadratic(s):
    return s**2

def cubic(s):
    return s**3

def newton_2nd(a):
    m = 1000
    return m*a

def kepler_3rd_law(s):
    g_constant = 6.67e-11
    solar_mass = 2e30
    earth_solar_distance = 150e8
    p = np.sqrt(((4*(np.pi**2))/(g_constant*(solar_mass+s)))*earth_solar_distance**3)
    return p/(365*24*3600)

def sinh(s):
    return (np.exp(s)-np.exp(-s))/2

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples



class NeuralNetwork:
    def __init__(self, x, y):
        self.maxx  = np.max(x)
        self.meanx = np.mean(x)
        self.stdx  = np.std(x)
        
        self.X  = x 
        self.y = y
        
        seed(1)
        H  = 784
        Ni = self.X.shape[1]
        No = self.y.shape[1]
    
        self.w0 = randn(Ni,H)
        self.b0 = randn(1,H)
        self.w1 = randn(H,H)
        self.b1 = randn(1,H)
        self.w2 = randn(H,No)
        self.b2 = randn(1,No)
        
    def forward(self):
        self.A0 = self.X
        Z1      = self.A0@self.w0 + self.b0   
        self.A1 = kepler_3rd_law(Z1)
        Z2      = self.A1@self.w1 + self.b1 
        self.A2 = kepler_3rd_law(Z2)
        Z3      = self.A2@self.w2 + self.b2   
        self.A3 = sinh(Z3)  
        
    def backward(self):
        alpha    =  1
        e        = self.y - self.A3
        delta3   = -e/e.shape[0]  # delta_w2
        delta2   = delta3@self.w2.T*self.A2*(1-self.A2) # delta_w1
        delta1   = delta2@self.w1.T*self.A1*(1-self.A1) # delta_w0
        self.w2 -= alpha*self.A2.T@delta3
        self.b2 -= alpha*sum(delta3)
        self.w1 -= alpha*self.A1.T@delta2
        self.b1 -= alpha*sum(delta2)
        self.w0 -= alpha*self.A0.T@delta1
        self.b0 -= alpha*sum(delta1)
        

    def predict(self,xs):
        self.X   = xs
        
        self.forward()
        predict = np.argmax(self.A3, axis = 1)
        return predict


# In[ ]:


sigmoid(3)
#wth why this one was here


# In[ ]:


from datetime import datetime

this = datetime.now()

X   = np.array(x_train)
y   = np.array(y_train)
ann = NeuralNetwork(X, y)

epochs = 5

for x in range(epochs):
    print("Epoches: ", x, '/',epochs)
    ann.forward()
    ann.backward()
    
then = datetime.now()
print('Time elapsed: ', then - this)


# In[ ]:


pred_train = ann.predict(X)
match_train    = sum((pred_train == np.argmax(y, axis = 1))*1)
train_acc  = match_train/len(y)*100

print('Train Accuracy:', train_acc, '%')


# Its okay, low enough for now. But how could you achieve it with kepler's third law?
# 
# Okay. the result is so big and it aims for the 0th label.

# In[ ]:


pred_test = ann.predict(x_test)
match_test    = sum((pred_test == np.argmax(y_test, axis = 1))*1)
test_acc  = match_test/len(y_test)*100

print('Test Accuracy:', test_acc, '%')


# # Here is your meme material!

# In[ ]:


plt.matshow(x_train_[100])
plt.show()

print('What your NN think: ',label[pred_train[100]])
print('The reality: ', label[np.argmax(y[100])])


# In[ ]:




