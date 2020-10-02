#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
trainData = pd.read_csv("/kaggle/input/restaurant-revenue-prediction/train.csv")
testData = pd.read_csv("/kaggle/input/restaurant-revenue-prediction/test.csv")
m=pd.read_csv("/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv")

# Any results you write to the current directory are saved as output.


# In[ ]:


print(trainData.head)


# In[ ]:


print(testData.head)


# In[ ]:


testData.shape


# In[ ]:


trainData.shape


# In[ ]:


y = trainData['revenue']
trainData = trainData.drop('Id',axis=1)
trainData = trainData.drop('Open Date',axis=1)
trainData = trainData.drop('City',axis=1)
trainData = trainData.drop('City Group',axis=1)
trainData = trainData.drop('Type',axis=1)
trainData = trainData.drop('revenue',axis=1)

testData = testData.drop('Id',axis=1)
testData = testData.drop('Open Date',axis=1)
testData = testData.drop('City',axis=1)
testData = testData.drop('City Group',axis=1)
testData = testData.drop('Type',axis=1)



# In[ ]:





# In[ ]:


testData


# In[ ]:


X=trainData


# In[ ]:


X
print(trainData.head)


# In[ ]:


plt.plot(X,y)


# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,y)


# In[ ]:


pred=lm.predict(testData)
print(pred)
print(pred.shape)
#print(trainData.shape)


# In[ ]:


sc=lm.score(X,y)
print(sc)


# In[ ]:


#Accuracy is too low because data is very less in trainData


# In[ ]:


testData = pd.read_csv("/kaggle/input/restaurant-revenue-prediction/test.csv")
submission = pd.DataFrame({
        "Id": testData["Id"],
        "Prediction": pred
    })
submission.to_csv('LinearRegressionSimple.csv',header=True, index=False)


# In[ ]:




