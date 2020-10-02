#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("../input/Iris.csv" )


# In[4]:


df.head()


# In[5]:


sns.pairplot(df,palette='husl',hue = 'Species')


# In[8]:


#using Svm to figure out the Species
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[9]:


#importing the gridsearch
from sklearn.grid_search import GridSearchCV


# In[77]:


#spliting traing and testing data
X = df.drop('Species',axis = 1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# #**Using Grid Search method and SVM estimator******

# In[11]:


#defining parameter for grid search
param = {'C': [0.1,1,10,100,1000], "gamma":[1,0.1,0.01,0.001, 0.0001],'kernel' : ['linear', 'rbf','sigmoid']}
grd_search = GridSearchCV(SVC(),param, verbose=2)


# In[12]:


#fitting the Traing data on grid search model
grd_search.fit(X_train,y_train)


# In[13]:


#Getting the best parameter
grd_search.best_params_


# In[98]:


#prediction
pred = grd_search.predict(X_test)


# In[101]:


#grd_search.best_estimator_


# In[93]:


df.columns


# In[113]:


a = {"Id":999,'SepalLengthCm':3.5,'SepalWidthCm':3.8,'PetalLengthCm':4.5,'PetalWidthCm':.6}
pd.Series(a)
X_test.append(a,ignore_index=True)


# In[114]:


#grd_search.predict(newly added data)
grd_search.predict(X_test[-1:])


# In[65]:


len(X_test)


# In[ ]:





# In[ ]:


print(classification_report(pred,y_test))


# In[ ]:


print(confusion_matrix(pred,y_test))


# **Using SVM methode**

# In[ ]:


svm = SVC(C= 1,gamma = 1, kernel = 'linear')


# In[ ]:


svm.fit(X_train,y_train)


# In[ ]:


p = svm.predict(X_test)


# In[ ]:


print (classification_report(p,y_test))


# In[ ]:


print (confusion_matrix(p,y_test))


# In[ ]:




