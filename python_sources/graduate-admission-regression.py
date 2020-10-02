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
import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.


# In[ ]:


admission = pd.read_csv('../input/Admission_Predict.csv')


# In[ ]:


admission.head()


# In[ ]:


admission.info()


# In[ ]:


admission.isnull().sum()


# In[ ]:


target = admission['Chance of Admit ']
target.head()


# In[ ]:


df = admission.copy()
df.head()


# In[ ]:


df.nunique()


# In[ ]:


sns.scatterplot(df['GRE Score'][:100],df['Chance of Admit '][:100],s=25,data=df);


# In[ ]:


df.corr()['Chance of Admit ']


# In[ ]:


df.drop(columns='Chance of Admit ',axis=1,inplace=True)
df.head()


# In[ ]:


df.set_index('Serial No.',inplace=True)
df.head()


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid = train_test_split(df,target,random_state=0)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

log_clf = LinearRegression().fit(X_train,y_train)
lnr_clf = LinearRegression().fit(X_train,y_train)


# In[ ]:


print('Accuracy of Logistic regression classifier on test set: {:.3f}'.format(log_clf.score(X_train,y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.3f}'.format(log_clf.score(X_valid,y_valid)))
print()
print('Accuracy of Linear regression classifier on test set: {:.3f}'.format(lnr_clf.score(X_train,y_train)))
print('Accuracy of Linear regression classifier on test set: {:.3f}'.format(lnr_clf.score(X_valid,y_valid)))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

dt_clf = DecisionTreeRegressor().fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on test set: {:.3f}'.format(dt_clf.score(X_train,y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.3f}'.format(dt_clf.score(X_valid,y_valid)))


# By default the `n_neighbors` value in KNN is 5

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

knn_clf = KNeighborsRegressor().fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on test set: {:.3f}'.format(knn_clf.score(X_train,y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.3f}'.format(knn_clf.score(X_valid,y_valid)))


# The above putput is an example of overfitting

# By default the `n_neighbors` value to 8 and we can crealy see that we have avoided overfitting to some extent

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

knn_clf = KNeighborsRegressor(n_neighbors=8).fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on test set: {:.3f}'.format(knn_clf.score(X_train,y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.3f}'.format(knn_clf.score(X_valid,y_valid)))


# In[ ]:


from sklearn.svm import SVR

svc_reg = SVR(gamma='auto').fit(X_train,y_train)

print('Accuracy of Logistic regression classifier on test set: {:.3f}'.format(svc_reg.score(X_train,y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.3f}'.format(svc_reg.score(X_valid,y_valid)))

