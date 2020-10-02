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
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:53:13 2020

@author: Ishal
"""

import numpy as np

class neural_network:
    
    def split(self,x,y,size):
        index = int(x.shape[1]*size)
        x_train = x[:,0:index]
        x_test = x[:,index:]
        y_train = y[:,0:index]
        y_test = y[:,index:]
        return x_train,y_train,x_test,y_test
    
    def sigmoid(self,A):
        Z = 1 / (1 + np.exp(-A))
        return Z
    
    def normalize(self,x):
        for i in range(x.shape[1]):
            x[:,i] = (x[:,i] - np.mean(x[:,i])) / (np.std(x[:,1]))
            
        return x
    
    def parameters_initialize(self,X,layer):
        parameters = {}
        parameters["W1"] = np.random.randn(layer[0],X.shape[0])*(np.sqrt(2/X.shape[0]))
        parameters["b1"] = np.zeros((layer[0],1))
        for i in range(len(layer)-2):
            parameters["W"+str(i+2)] = np.random.randn(layer[i+1],layer[i])*(np.sqrt(2/layer[i]))
            parameters["b"+str(i+2)] = np.zeros((layer[i+1],1))
        i+=1
        parameters["W"+str(i+2)] = np.random.randn(layer[i+1],layer[i])*(np.sqrt(1/layer[i]))
        parameters["b"+str(i+2)] = np.zeros((layer[i+1],1))
        return parameters
    
    def relu(self,A):
        Z = np.maximum(A,0.001*A)
        return Z
    
    def forward_activation(self,X,parameters):
        caches = {}
        caches["Z1"] = np.dot(parameters["W1"],X) + parameters["b1"]
        caches["A1"] = self.relu(caches["Z1"])
        for i in range((len(parameters)//2)-2):
            caches["Z"+str(i+2)] = np.dot(parameters["W"+str(i+2)],caches["A"+str(i+1)]) + parameters["b"+str(i+2)]
            caches["A"+str(i+2)] = self.relu(caches["Z"+str(i+2)])
        i+=1
        caches["Z"+str(i+2)] = np.dot(parameters["W"+str(i+2)],caches["A"+str(i+1)]) + parameters["b"+str(i+2)]
        caches["A"+str(i+2)] = self.relu(caches["Z"+str(i+2)])
        return caches
    
    def backward_prop(self,X,Y,parameters,caches):
        caches_d = {}
        l = len(parameters)//2
        m = X.shape[1]
        caches_d["dZ"+str(l)] = caches["A"+str(l)] - Y
        caches_d["dW"+str(l)] = np.dot(caches_d["dZ"+str(l)],np.transpose(caches["A"+str(l-1)]))/m
        caches_d["db"+str(l)] = np.sum(caches_d["dZ"+str(l)],axis=1,keepdims=True)/m
        for p in range(l,2,-1):
            t = np.zeros(caches["A"+str(p-1)].shape)
            for i in range(caches["A"+str(p-1)].shape[0]):
                for j in range(caches["A"+str(p-1)].shape[1]):
                    if(caches["A"+str(p-1)][i][j]>0):
                        t[i][j] = 1
            caches_d["dZ"+str(p-1)] = np.dot(np.transpose(parameters["W"+str(p)]),caches_d["dZ"+str(p)])*t
            caches_d["dW"+str(p-1)] = np.dot(caches_d["dZ"+str(p-1)],np.transpose(caches["A"+str(p-2)]))/m
            caches_d["db"+str(p-1)] = np.sum(caches_d["dZ"+str(p-1)],axis=1,keepdims=True)/m
        p = 2
        t = np.zeros(caches["A"+str(p-1)].shape)
        for i in range(caches["A"+str(p-1)].shape[0]):
            for j in range(caches["A"+str(p-1)].shape[1]):
                if(caches["A"+str(p-1)][i][j]>0):
                    t[i][j] = 1
        caches_d["dZ"+str(p-1)] = np.dot(np.transpose(parameters["W"+str(p)]),caches_d["dZ"+str(p)])*t
        caches_d["dW"+str(p-1)] = np.dot(caches_d["dZ"+str(p-1)],np.transpose(X))/m
        caches_d["db"+str(p-1)] = np.sum(caches_d["dZ"+str(p-1)],axis=1,keepdims=True)/m
        return caches_d
    
    def cost(self,AL,Y):
        m = Y.shape[1]
        cost = -(np.sum((Y*np.log(AL)) + ((1-Y)*np.log(1-AL))))/m
        return cost
    
    def updation(self,caches_d,parameters,learning_rate=0.3):
        for i in range(len(parameters)//2):
            parameters["W"+str(i+1)]-=learning_rate*caches_d["dW"+str(i+1)]
            parameters["b"+str(i+1)]-=learning_rate*caches_d["db"+str(i+1)]
        return parameters