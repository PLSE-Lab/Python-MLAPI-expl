#!/usr/bin/env python
# coding: utf-8

# # Basic Introduction to PCA - Principal Component Analysis

# ![PCA](https://lh3.googleusercontent.com/proxy/2mLDGKfDqC81Bf_8DT2ndcl637heYbRSGqk8OZALn9N1OdgY1P2PYlzoAH6sDhKRBr8RyGQyUJ7O0VuNVmdKetg2xogJYmcmUrSuBl65BO1NFPFiDZ5YCnrTZPCOnnV5l7lQuXil_utOfUbH61irT3J-vsRZmtDhKqeDr5ZJuG8UzWHh3TZT5t6XDaZdkwVBpCOGcg00ewZlaCLguz6V8IyypTGZUQ)

# Hey Guys , like my first kaggle this is also a hyper basic notebook that is based on helping you understand a fundamental concept in ML.
# Without further ado let us get started...
# 

# # What is PCA

# PCA  is a **Dimensionality Reduction** technique meaning it helps reduce a higher dimensionaity matrix to a lower dimensionality one....'duh'.
# In easy terms it uses fancy math to reduce complication of data in order to reduce the number of columns we have to deal with as in complicated ML datasets
# it becomes very difficult to differentiate the useful from the garbage.
# I will not be going through complicated math as i  do not find it useful under this context.
# 
# 

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


df= pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


df.head()


# In[ ]:


# taking care of categorical values
data = pd.get_dummies(df)


# In[ ]:


data.info()


# In[ ]:


# using pca on this dataset
data = data.drop('Attrition_Yes',axis =1)
from sklearn.decomposition import PCA
pca= PCA()
g = pca.fit(data)
#print(g.explained_variance_ratio_)
np.cumsum(g.explained_variance_ratio_)


# In[ ]:


data_X = data.drop('Attrition_No',axis=1)
data_Y = data['Attrition_No']


# In[ ]:


data_X = pca.fit_transform(data)


# In[ ]:


print(len(data))


# In[ ]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(data_X,data_Y,test_size = 0.3)


# In[ ]:


# classification
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
from sklearn import metrics
print(metrics.accuracy_score(y_pred,y_test))


# In[ ]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
g = lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
metrics.r2_score(y_pred,y_test)


# In[ ]:


#checking what would happen if pca was not applied
data2 = pd.get_dummies(df)


# In[ ]:


data2 = data2.drop('Attrition_Yes',axis =1 )
data2.head()


# In[ ]:


data2X = data2.drop('Attrition_No',axis=1)
data2Y = data2['Attrition_No']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(data2X,data2Y,test_size = 0.3)


# In[ ]:


# classification
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
from sklearn import metrics
print(metrics.accuracy_score(y_pred,y_test))


# In[ ]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
g = lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
metrics.r2_score(y_pred,y_test)

