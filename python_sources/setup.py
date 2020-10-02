#!/usr/bin/env python
# coding: utf-8

# In[21]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# Import training and testing sets

# In[22]:


train = pd.read_csv('../input/bank-train.csv')
test = pd.read_csv('../input/bank-test.csv')


# In[23]:


train.describe()


# In[24]:


sb.heatmap(train.corr())  


# Choosing several highly correlated features with y (the output): pdays, previous, and nr.employed in logistic regression model

# In[25]:


x_train = train[['pdays','previous','nr.employed']]
y_train = train.y
# create and fit model
LogReg = LogisticRegression()
LogReg.fit(x_train, y_train)


# In[26]:


# metrics on training data (do NOT use this as a reliable estimate)
y_train_pred = LogReg.predict(x_train)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_train, y_train_pred)
confusion_matrix


# Initial Results: Confusion Matrix and Classification Report

# In[27]:


print(classification_report(y_train, y_train_pred))


# Make predictions on test set

# In[29]:


x_test = test[['pdays','previous','nr.employed']]
predictions = LogReg.predict(x_test)


# Create submission file

# In[30]:


submission = pd.concat([test.id, pd.Series(predictions)], axis = 1)
submission.columns = ['id', 'Predicted']
submission.to_csv('submission.csv', index=False)


# This is a baseline model that didn't do much analysis overall. 
# Suggestions: Look for missing data, do more exploration, better select features, use some categorical features, apply cross validation, optimize for F1 score
