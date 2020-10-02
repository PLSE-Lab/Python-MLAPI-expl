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
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/train_real.csv')
test_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/test_real.csv')


# In[ ]:


train_df


# Your EDA (Exploratory Data Analysis) goes here. Get a good feel of the data, look out for stuff that might help later.

# In[ ]:


sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(14,8)
plt.show()


# # Train-Test split
# 
# You are also expected to split train_df into train and validation dataframes (or else choose a cross validation scheme)
# 

# In[ ]:


train_df.dtypes


# In[ ]:


X_train = train_df.drop(['Id', 'label','Soil','Wilderness_Area_C'], axis=1)
Y_train = train_df['label']


# In[ ]:


X_test,X_val,Y_test,Y_val = train_test_split(X_train,Y_train,random_state=26)
X_val


# Dropping 'Soil' column for convenience. You should try to think of ways to generate features from these columns. Try seeing kernels from other Kaggle (Tabular data) competitions for inspirations for Feature Engineering.

# In[ ]:


X_train.head()


# In[ ]:


Y_train.head()


# In[ ]:





# In[ ]:


X_tester = test_df.drop(['Id','Soil','Wilderness_Area_C'], axis=1)
X_tester.head()


# # Basic Binary Logistic regression
# 
# You should obviously see the limitations of a model and familiarise yourselves with other models in sci-kit learn.

# In[ ]:


# we can try to fit the base model 
# we can try logistic regression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn import metrics 
from sklearn.metrics import roc_auc_score

clf=LogisticRegressionCV(cv=5, max_iter = 1000).fit(X_train, Y_train)


# In[ ]:


train_res=clf.predict(X_train)
train_res


# In[ ]:


test_res = clf.predict(X_tester)
test_res


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100,oob_score=True,max_features=10)


random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_tester)


# # Make a submission
# 

# In[ ]:


submission_df = pd.DataFrame()
submission_df['Id'] = test_df['Id']
submission2_df = pd.DataFrame()
submission2_df['Id'] = test_df['Id']


# In[ ]:


submission_df['Predicted'] = test_res.tolist()
submission2_df['Predicted'] = Y_pred.tolist()


# In[ ]:


submission_df.tail()


# In[ ]:


submission2_df.tail()


# In[ ]:


submission_df.to_csv('vanilla_logistic_submission.csv',index=False)
submission2_df.to_csv('vanilla_logistic_submission2.csv',index=False)


# In[ ]:


get_ipython().system('ls')


# In[ ]:





# In[ ]:




