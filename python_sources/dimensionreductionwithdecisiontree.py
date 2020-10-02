#!/usr/bin/env python
# coding: utf-8

# In[51]:


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


# In[52]:


breast_cancer=pd.read_csv('../input/Breast_cancer_data.csv')


# In[53]:


breast_cancer.tail()
x=breast_cancer.iloc[:,0:5]
y=breast_cancer.iloc[:,5:]


# In[54]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(max_features=3,criterion="entropy")


# In[55]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=1999)


# In[56]:


dtc.fit(x_train,y_train)


# In[57]:


ypr=dtc.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(accuracy_score(y_test,ypr))
print(confusion_matrix(y_test,ypr))


# In[58]:


dtc.tree_.compute_feature_importances()


# Last 3 feature is the most important in information theory entropy

# In[59]:


breast_cancer.tail()


# In[60]:


reducted=breast_cancer.iloc[:,2:5]
del reducted["mean_area"]


# In[61]:


from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)
ypr=svm.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(accuracy_score(y_test,ypr))
print(confusion_matrix(y_test,ypr))


# In[62]:


x_train, x_test, y_train, y_test=train_test_split(reducted,y,test_size=0.20,random_state=1999)
svm=SVC()
svm.fit(x_train,y_train)
ypr=svm.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(accuracy_score(y_test,ypr))
print(confusion_matrix(y_test,ypr))


# Boost accuracy to %66 to %86 with only reduced dimensions. 

# In[ ]:





# In[ ]:




