#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.optimize as opt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
#print(os.listdir("../input"))
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
X_train = train_data.iloc[:25000,1:].astype('float')
X_val =  train_data.iloc[32000:,1:].astype('float')
y_train = train_data.iloc[:25000,0]
y_val = train_data.iloc[32000:,0]
X_test = test_data.iloc[:,:].astype('float')
m_train = len(X_train)
m_test = len(X_test)
m_val = len(X_val)
y_train.dtype
# Any results you write to the current directory are saved as output.


# In[ ]:


ones = np.ones((m_train,1), dtype = 'float')
X_train = np.hstack((ones,X_train))
ones = np.ones((m_test,1), dtype = 'float')
X_test = np.hstack((ones,X_test))
ones = np.ones((m_val,1), dtype = 'float')
X_val = np.hstack((ones,X_val))
(a,b) = X_train.shape
X_test.shape


# In[ ]:



# sigmoid Function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


# In[ ]:


# cost function
def cost(theta, X, y):
        prediction = sigmoid(X_train @ theta)
        prediction[prediction == 1.0] = 0.999
        prediction[prediction == 0.0] = 0.001
        J = (-1/m_train)*np.sum(np.multiply(y,np.log(prediction))
                       + (np.multiply(1-y,np.log(1- prediction))))
        return J
theta = np.zeros(X_train.shape[1])
print(cost(theta,X_train,y_train))


# In[ ]:


# Gradient funtion
def gradient(theta, X, y):
    return ((1/m_train) * (np.dot(X.T , (sigmoid(X @ theta) - y))))


# In[ ]:


# fitting the parameters
theta_optimized = np.zeros([10,b])
for i in range (0,10):
    label = (y_train == i).astype(int)
    initial_theta = np.zeros(X_train.shape[1])
    theta_optimized[i,:] = opt.fmin_cg(cost, initial_theta, gradient, (X_train, label))


# In[ ]:


# probabilies for x_val
tmp = sigmoid(X_train @ theta_optimized.T)
predictions = tmp.argmax(axis=1)
print("Training accuracy:", str(100 * np.mean(predictions == y_train)) + "%")


# In[ ]:


# probability of each no 0 to 10 (28000*10)
tmp = sigmoid(X_test @ theta_optimized.T)
predictions = tmp.argmax(axis=1)
ID = list(range(1,28001))
my_submission = pd.DataFrame({'ImageId': ID, 'Label': predictions})
my_submission.to_csv('submission.csv', index=False)
my_submission.head()

