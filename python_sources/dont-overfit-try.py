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


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
submission=pd.read_csv("../input/sample_submission.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


submission.head()


# In[ ]:


y_train=train["target"]
print(y_train.head())
x_train=train
del x_train["target"]


# In[ ]:


test_id=test["id"]
x_test=test
del x_test["id"]
del x_train["id"]


# In[ ]:


print(x_train.head())
print(x_test.head())


# In[ ]:


#Shapes
print(x_train.shape)
print(x_test.shape)


# In[ ]:


#Divding my train data to see whetther my data overfit
from sklearn import model_selection
x_train_train,x_train_test,y_train_train,y_train_test=model_selection.train_test_split(x_train,y_train)


# In[ ]:


#linear Regressor
from sklearn.linear_model import LinearRegression


# In[ ]:


clf=LinearRegression()
clf.fit(x_train_train,y_train_train)


# In[ ]:


print(clf.score(x_train_test,y_train_test))
print("Linear Model will fail badly")


# In[ ]:


#Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(solver='saga',max_iter=1000,C=0.05)
clf.fit(x_train_train,y_train_train)


# In[ ]:


print(clf.score(x_train_test,y_train_test))
print("Logistic Regression is giving a accuracy of 73% with test data created fom testing data")
y_predict=clf.predict(x_test)


# In[ ]:


predictions=pd.DataFrame({"id":test_id,"target":y_predict})


# In[ ]:


predictions.to_csv("Dont_overfit.csv")


# In[ ]:


#Random_Forest
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,n_jobs=8,max_depth=50,min_samples_split=2,random_state=1)
clf.fit(x_train_train,y_train_train)


# In[ ]:


clf.score(x_train_test,y_train_test)


# In[ ]:


from sklearn.svm import SVC
clf=SVC(random_state=1,degree=3,gamma='scale')
clf.fit(x_train_train,y_train_train)


# In[ ]:


clf.score(x_train_test,y_train_test)


# In[ ]:




