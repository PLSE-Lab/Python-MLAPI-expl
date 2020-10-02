#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/pga9-classification-hackathon/hackathon_train.csv')
test = pd.read_csv('/kaggle/input/pga9-classification-hackathon/hackathon_test.csv')
sample_submission = pd.read_csv('/kaggle/input/pga9-classification-hackathon/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


sns.heatmap(train.isnull())


# In[ ]:


sns.pairplot(train,hue = 'default.payment.next.month',vars = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE'])


# In[ ]:


sns.countplot(train['default.payment.next.month'])


# In[ ]:


train.columns


# In[ ]:


sns.scatterplot(x = 'BILL_AMT1',y= 'BILL_AMT2',hue = 'default.payment.next.month',data=train)


# In[ ]:


plt.figure(figsize=(20,15))
sns.heatmap(train.corr(),annot = True)


# In[ ]:


train['SEX'].value_counts()


# In[ ]:


# train.drop('ID',inplace = True,axis = 1)
# test.drop('ID',inplace = True,axis = 1)


# In[ ]:


X = train.drop('default.payment.next.month',axis = 1)


# In[ ]:


y = train['default.payment.next.month']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test , y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


# In[ ]:


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# **SUPPORT VECTORE MACHINE**

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc_model = SVC()


# In[ ]:


svc_model.fit(X_train,y_train)


# In[ ]:


y_pred = svc_model.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report , confusion_matrix


# In[ ]:


cm = confusion_matrix(y_test,y_pred)


# In[ ]:


cr = classification_report(y_test,y_pred)


# In[ ]:


print(cr)


# In[ ]:


sns.heatmap(cm,annot = True,fmt='d')


# In[ ]:


#use the normalization to increase the accuracy
min_train = X_train.min()
print(min_train)


# In[ ]:


range_train = (X_train - min_train).max()
print(range_train)


# In[ ]:


X_train_scaled = (X_train - min_train)/range_train


# In[ ]:


X_train_scaled.head()


# In[ ]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[ ]:


svc_model.fit(X_train_scaled,y_train)


# In[ ]:


y_pred = svc_model.predict(X_test_scaled)


# In[ ]:


cm = confusion_matrix(y_test,y_pred)


# In[ ]:


sns.heatmap(cm,annot = True,fmt = 'd')


# In[ ]:


cr = classification_report(y_test,y_pred)


# In[ ]:


print(cr)


# **LOGISTIC REGRESSION**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logistic_regression = LogisticRegression()


# In[ ]:


logistic_regression.fit(X_train,y_train)


# In[ ]:


y_pred = logistic_regression.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test,y_pred)


# In[ ]:


sns.heatmap(cm,annot = True,fmt = 'd')


# In[ ]:


cr = classification_report(y_test,y_pred)


# In[ ]:


print(cr)


# In[ ]:


logistic_regression.fit(X_train_scaled,y_train)


# In[ ]:





# In[ ]:


param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}


# In[ ]:


from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(),param_grid=param_grid,refit=True,verbose=4)


# In[ ]:


grid.fit(X_train_scaled,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid_prediction = grid.predict(X_test_scaled)


# In[ ]:


cm = confusion_matrix(y_test,grid_prediction)


# In[ ]:


sns.heatmap(cm,annot = True,fmt = 'd')


# In[ ]:


cr = classification_report(y_test,grid_prediction)


# In[ ]:


print(cr)

