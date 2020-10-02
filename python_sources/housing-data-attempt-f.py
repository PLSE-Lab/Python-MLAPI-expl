#!/usr/bin/env python
# coding: utf-8

# This algorithm will output a set of answers, based on standardized data fed into a KNN. NaN's are replaced with -1.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Define all of the required data frames upfront
train = pd.read_csv("../input/train.csv")
output = pd.read_csv("../input/sample_submission.csv")
test = pd.read_csv("../input/test.csv")

trainX = train.drop('SalePrice', axis=1)
testX = test

trainY = train['SalePrice']

trainProcess = pd.get_dummies(trainX)
testProcess = pd.get_dummies(testX)

# get the columns in train that are not in test
col_to_add = np.setdiff1d(trainProcess.columns, testProcess.columns)

# add these columns to test, setting them equal to zero
for c in col_to_add:
    testProcess[c] = 0

# select and reorder the test columns using the train columns
testX = testProcess[trainProcess.columns]
trainX = trainProcess

print(testX.shape)
print(trainX.shape)
#plt.scatter(train['YrSold'], train['SalePrice'])


# In[ ]:


#Clean NaN
trainX = trainX.fillna(value = "-1")
testX = testX.fillna(value = '-1')

#Standardize the data
sX = preprocessing.scale(trainX)
trainX = pd.DataFrame(sX) #Scaled array of all values of the training X

sX = preprocessing.scale(testX)
testX = pd.DataFrame(sX)

#scaledTest = preprocessing.scale(Test)
#NTest = pd.DataFrame(scaledTest)


# In[ ]:


#Make a model using a Random Forest Classifier for classification
rfc = RandomForestClassifier()
rfc.fit(trainX, trainY)


#Make a model using a KNN for regression
#knn =  KNeighborsRegressor(n_neighbors = 3)
#knn.fit(trainX, trainY)


# In[ ]:


#Make predictions based on said model for the test data
answers = rfc.predict(testX)
#answers = knn.predict(testX)


# In[ ]:


output['SalePrice'] = answers


# In[ ]:


output.to_csv('Predictions.csv', index=False)


# In[ ]:




