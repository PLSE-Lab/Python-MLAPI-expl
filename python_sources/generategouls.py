#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition.kernel_pca import KernelPCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn import ensemble


# In[2]:


trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")


# In[3]:


trainData.isnull().sum()


# In[4]:


testData.isnull().sum()


# In[5]:


np.unique(trainData[['type']].values)


# In[6]:


np.unique(trainData[['color']].values)


# In[7]:


np.unique(testData[['color']].values)


# In[8]:


trainData.head()


# In[9]:


lbl = preprocessing.LabelEncoder()
lbl.fit(list(trainData['color'].values)) 
trainData['color'] = lbl.transform(list(trainData['color'].values))

lbl = preprocessing.LabelEncoder()
lbl.fit(list(trainData['type'].values)) 
trainData['type'] = lbl.transform(list(trainData['type'].values))


# In[10]:


trainData.head()


# In[11]:


yTrain = trainData['type'].values
xTrain = trainData.drop(["id", "type"], axis=1)
xTrain.head()


# In[12]:


model = ensemble.RandomForestClassifier(n_estimators=170)
model.fit(xTrain, yTrain)


# In[13]:


model.score(xTrain,yTrain)


# In[14]:


lbl = preprocessing.LabelEncoder()
lbl.fit(list(testData['color'].values)) 
testData['color'] = lbl.transform(list(testData['color'].values))


# In[15]:


testData.head()


# In[16]:


yTest = testData['id'].values
xTest = testData.drop(["id"], axis=1)
xTest.head()


# In[17]:


pred = model.predict(xTest)
my_submission = pd.DataFrame({'ID': yTest, 'y': pred})


# In[18]:


predic = pd.read_csv('../input/sample_submission.csv')


# In[19]:


my_submission_new = []
i = 0
for row in my_submission.iterrows():
    my = {}
    my['id'] = predic.id[i]
    if(row[1]['y'] ==0):
        my['type'] = 'Ghost'
    elif(row[1]['y'] ==1):
        my['type'] = 'Ghoul'
    else:
        my['type'] = 'Goblin'
    my_submission_new.append(my)
    i = i+1


# In[20]:


df = pd.DataFrame(my_submission_new, columns=["id","type"])


# In[21]:


df.to_csv('submission.csv', index=False)


# In[ ]:




