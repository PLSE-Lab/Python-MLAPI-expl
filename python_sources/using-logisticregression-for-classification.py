#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns


# Load the credit data and let get some information on the data.

# In[ ]:


credit_df = pd.read_csv("/kaggle/input/credit/Default.csv",index_col=0)
credit_df.head()


# In[ ]:


credit_df.info()


# Balance by Gender

# In[ ]:


sns.boxplot(y="balance",x="student",data=credit_df,width=0.8)


# In[ ]:


sns.pairplot(credit_df,hue="student")


# Let has a look at the Correlation Matrix as a heatmap

# In[ ]:


credit_df[credit_df["default"] == "Yes"].head()


# In[ ]:


sns.heatmap(data=credit_df.corr(), annot=True, linewidths=.5, cmap="coolwarm")


# Okay lets predict default using Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

modl = LogisticRegression(random_state=10)

# Create feature set
X = credit_df[['balance','income']]
X["student"] = pd.get_dummies(credit_df["student"],drop_first=True)
y = credit_df["default"]

# create a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

modl.fit(X_train,y_train)
y_predict = modl.predict(X_test)


# Great our Model is ready so how accurate is it?

# In[ ]:


print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))


# Looks like based on the data we can predict the likelyhood of not defaulting with 96% accuracy. From the confusion matrix above, we only has 1 also false positive (in this case a positive reading is the likelyhood of not defaulting). Great lets try it out.
# 
# So a user with a credit balance of 1500 and an income of 25k who is not a student. I wonder what if this user will defualt

# In[ ]:


modl.predict_proba(np.array([1500,25000,0]).reshape(1,-1))


# According to the model, Chances of a "No" (i.e. will not default) 93.3% and for  Yes (i.e will default) is 6.6%. The Nos have it. Increase that guys limit :-)
