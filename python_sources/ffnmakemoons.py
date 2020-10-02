#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
from tqdm import tqdm_notebook


# In[ ]:


my_map = matplotlib.colors.LinearSegmentedColormap.from_list('',['red','green','blue'])


# In[ ]:


data,label = make_moons(n_samples = 1000,noise = 0.2,random_state = 0)


# In[ ]:


plt.scatter(data[:,0],data[:,1],c = label,cmap = my_map)
plt.show()


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(data,label,stratify = label,random_state = 0)


# In[ ]:


class FFN:
    
    def __init__(self,input_size = 2,hidden = [2]):
        self.nx = input_size
        self.ny = 1
        self.nh = len(hidden)
        self.size = [self.nx] + hidden + [self.ny]
        self.W = {}
        self.B = {}
        for i in range(self.nh + 1):
            self.W[i + 1] = np.random.randn(self.size[i],self.size[i + 1])
            self.B[i + 1] = np.zeros((1,self.size[i + 1]))
    
    def sigmoid(self,x):
        return 1.0/(1.0 + np.exp(-x))
    
    def grad_sigmoid(self,x):
        return x*(1 - x)
    
    def forwardpass(self,x):
        self.A = {}
        self.H = {}
        self.H[0] = x.reshape(1,-1)
        for i in range(self.nh + 1):
            self.A[i + 1] = np.matmul(self.H[i],self.W[i + 1]) + self.B[i + 1]
            self.H[i + 1] = self.sigmoid(self.A[i + 1])
        return self.H[self.nh + 1]
    
    def predict(self,X):
        y_pred = []
        for x in X:
            y_pred.append(self.forwardpass(x))
        return np.array(y_pred).ravel()
    
    def grad(self,x,y):
        self.forwardpass(x)
        self.dW = {}
        self.dB = {}
        self.dH = {}
        self.dA = {}
        L = self.nh + 1
        self.dA[L] = (self.H[L] - y)
        for i in range(L,0,-1):
            self.dW[i] = np.matmul(self.H[i - 1].T,self.dA[i])
            self.dB[i] = self.dA[i]
            self.dH[i - 1] = np.matmul(self.dA[i],self.W[i].T)
            self.dA[i - 1] = np.multiply(self.dH[i - 1],self.grad_sigmoid(self.H[i - 1]))
    
    def fit(self,X,Y,epochs = 1,lr = 1,loss_plt = True,initialise = True):
        if initialise:
            self.W = {}
            self.B = {}
            for i in range(self.nh + 1):
                self.W[i + 1] = np.random.randn(self.size[i],self.size[i + 1])
                self.B[i + 1] = np.zeros((1,self.size[i + 1]))
        
        if loss_plt:
            loss = {}
        
        for epoch in tqdm_notebook(range(epochs), total = epochs,unit = 'epoch'):
            dW = {}
            dB = {}
            for i in range(self.nh + 1):
                dW[i + 1] = np.zeros((self.size[i],self.size[i + 1]))
                dB[i + 1] = np.zeros((1,self.size[i + 1]))
            for x,y in zip(X,Y):
                self.grad(x,y)
                for i in range(self.nh + 1):
                    dW[i + 1] += self.dW[i + 1]
                    dB[i + 1] += self.dB[i + 1]
            m = X.shape[1]
            for i in range(self.nh + 1):
                self.W[i + 1] = self.W[i + 1] - lr*dW[i + 1]/m
                self.B[i + 1] = self.B[i + 1] - lr*dB[i + 1]/m
            if loss_plt:
                loss[epoch] = mean_squared_error(self.predict(X),Y)
        if loss_plt:
            plt.plot(np.array(list(loss.values())))
            plt.xlabel('Epoch')
            plt.ylabel('Mean Squared Error')
            plt.show()


# In[ ]:


Brain = FFN(2,[6,6])


# In[ ]:


Brain.fit(X_train,Y_train,epochs = 3500,lr =  0.001)


# In[ ]:


y_pred = (Brain.predict(X_train) > 0.5).astype(np.int).ravel()
accuracy_train = accuracy_score(y_pred,Y_train)
print(accuracy_train)


# In[ ]:


y_pred = (Brain.predict(X_test) > 0.5).astype(np.int).ravel()
accuracy_test = accuracy_score(y_pred,Y_test)
print(accuracy_test)


# In[ ]:




