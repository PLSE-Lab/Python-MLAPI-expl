#!/usr/bin/env python
# coding: utf-8

# In[22]:


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


# In[23]:


import pandas as pd


# In[24]:


data=pd.read_csv("../input/winequality-red.csv")


# In[25]:


data.head()


# In[26]:


data.shape


# In[27]:


#description about the dataset
data.describe()


# In[28]:


#infromation about the dataset
data.info()


# In[29]:


from matplotlib import pyplot as plt


# In[30]:


data['quality'].value_counts().plot.bar()
plt.show()


# In[31]:


data['quality'] = data['quality'].map({
        3 : 0,
        4 : 0,
        5 : 0,
        6 : 0,
        7 : 1,
        8 : 1         
})


# In[32]:


import seaborn as sns


# In[33]:


from sklearn.tree import DecisionTreeClassifier


# In[34]:


from sklearn.neighbors import KNeighborsClassifier


# In[35]:


X = data[['fixed acidity','citric acid','residual sugar','sulphates','alcohol']]
Y = data[['quality']]
print(X.shape)
print(Y.shape)


# In[36]:


norm=(X-X.min())/(X.max()-X.min())
norm.head()


# In[37]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(norm,Y)


# In[38]:


#prediction
knnpred= classifier.predict(norm)
knnpred


# In[39]:


#confusion matrix
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Y, knnpred))


# In[40]:


#accuracy
from sklearn.metrics import accuracy_score 
Accuracy_Score = accuracy_score(Y, knnpred)
Accuracy_Score


# In[41]:


#decision tree
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(norm, Y)


# In[42]:


dtpred=classifier.predict(norm)
dtpred


# In[43]:


#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y,dtpred)
cm


# In[44]:


#accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y,dtpred)
accuracy


# In[ ]:




