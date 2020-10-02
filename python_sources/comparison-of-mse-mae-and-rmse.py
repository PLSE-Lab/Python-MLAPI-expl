#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#here are some made-up numbers to start with
target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]
error = []

for i in range(len(target)):
    error.append(target[i] - prediction[i])
print ("Errors:" ,error)


# In[ ]:


# Calculate the squared error and absolute value of errors
sqdError = []
absError = []
for j in error:
    sqdError.append(j*j)
    absError.append(abs(j))
    
print ("Absolute Error:", absError)
print ("Squared Error:", sqdError)


# In[ ]:


# Calculate and print Mean Squared Error (MSE)
MSE = sum(sqdError)/len(sqdError)
print ("MSE = ", MSE )


# In[ ]:


from  math import sqrt
# Calculate and print RMSE
RMSE = sqrt(MSE)
print ("RMSE = ", RMSE)


# In[ ]:


# Calculate MAE
MAE = sum(absError)/len(absError)
print ("MAE = ", MAE)


# In[ ]:


# Compare MSE to target variable
targetDeviation = []
targetMean = sum(target)/len(target)
for k in target:
    targetDeviation.append((k - targetMean)* (k - targetMean))
targetVariance = sum(targetDeviation)/len(targetDeviation)
print ("Target Variance = ", targetVariance)


# In[ ]:


targetStdDeviation = sqrt(sum(targetDeviation)/len(targetDeviation))
print ("Target Standard Deviation = ", targetStdDeviation)


# In[ ]:




