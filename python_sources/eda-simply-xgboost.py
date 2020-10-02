#!/usr/bin/env python
# coding: utf-8

# # Beginning Practice on Kaggle

# Some of my process may not be appropriate. Hope to get some suggestions from kaggle.

# # 0. Import Data.

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


#input the data
train=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv("/kaggle/input/titanic/test.csv")
#extract the result of training data and keep it for latter use
train_result=train.Survived
print(train_result.shape)


# In[ ]:


#merge the train and test data together so that it will be more convenient for us to do the transformation.
train_and_test=pd.concat([train.drop(["Survived"],axis=1),test])
print(train_and_test.shape)


# # 1. Feature engineering 
# ## 1.1.Adding a feature: Title, to replace Name

# In[ ]:


train_and_test['Title'] = train_and_test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_and_test=train_and_test.drop(["Name"],axis=1)
train_and_test.head(2)


# In[ ]:


#Let's see how many kinds of titles do we have, and count the amount of each type.
from collections import Counter
Counter(train_and_test["Title"])


# Some rare type like "Don","Mme" only has 1. And we can recognize that some of them are typos, such as "Mme". 

# In[ ]:


#Change the titles.
train_and_test['Title'] = train_and_test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_and_test['Title'] = train_and_test['Title'].replace('Mlle', 'Miss')
train_and_test['Title'] = train_and_test['Title'].replace('Ms', 'Miss')
train_and_test['Title'] = train_and_test['Title'].replace('Mme', 'Mrs')


# In[ ]:


#Check the count again to see whether the count is reasonable
Counter(train_and_test["Title"])


# ## 1.2.Dealing with Age 

# In[ ]:


#use the median to fill the na
train_and_test["Age"]=train_and_test["Age"].fillna(train_and_test.groupby("Title")["Age"].transform("median"))
train_and_test.describe()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # setting seaborn default for plots


# In[ ]:


#Check the distribution of age
plt.figure(figsize = (5,3))
sns.distplot(train_and_test["Age"])


# Seems to be normal. Thus, it should be OK.

# ## 1.3.Dealing with "Embarked"

# In[ ]:


#Fill the NA with Unknown
train_and_test["Embarked"]=train_and_test["Embarked"].fillna("Unknown")
train_and_test.head(2)


# ## 1.4.Dealing with "Fare"

# In[ ]:


#Still, the first step is to fill the NA.
train_and_test["Fare"]=train_and_test["Fare"].fillna(train_and_test["Fare"].median())
train_and_test["Fare"].describe()


# In[ ]:


from scipy import stats
from scipy.stats import norm, skew
stats.boxcox(train.Fare+1)[1]


# In[ ]:


# We use log should be OK.
plt.figure(figsize = (5,3))
sns.distplot(np.log(train_and_test["Fare"]+1))
plt.xlim()


# In[ ]:


train_and_test["Fare"]=np.log(train_and_test["Fare"]+1)


# ## 1.5.Dealing with "Cabin"

# In[ ]:


train_and_test["Cabin"].sample(3)


# In[ ]:


train_and_test["Cabin"]=train_and_test["Cabin"].str[:1]
train_and_test["Cabin"]=train_and_test["Cabin"].fillna("Unknown")
train_and_test.head(2)


# ## 1.6.Drop Useless Columns

# In[ ]:


train_and_test.sample(3)


# In[ ]:


train_and_test=train_and_test.drop(["PassengerId","Ticket"],axis=1)


# ## 1.7.Out put the dataset which is ready to be put into the model

# In[ ]:


train_and_test.to_csv("train_and_test.csv",index=False)


# In[ ]:


import xgboost as XGB


# In[ ]:


train_and_test=pd.read_csv("train_and_test.csv")
ttdata=pd.get_dummies(train_and_test)
all_mat=ttdata.iloc[:,:].values


# In[ ]:


from sklearn import preprocessing,model_selection


# In[ ]:


stand = preprocessing.StandardScaler()
stand.fit(all_mat)
all_mat_st = stand.transform(all_mat)


# # 2. Fitting Models

# In[ ]:


X=all_mat[:891,:]
y=train_result.iloc[:].values
n,p = X.shape


# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.25, random_state = 10, stratify = y)


# In[ ]:


#After tuning the parameters:
xgb=XGB.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.2,
              learning_rate=0.01, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=0.5, verbosity=1)
xgb.fit(X_train,y_train)


# In[ ]:


xgb.score(X_test,y_test)


# In[ ]:


#fit the model again, using all we know
xgb.fit(X,y)


# In[ ]:


PassengerId = test["PassengerId"]
test_mat=all_mat[891:,:]
test_pred=xgb.predict(test_mat)
output=pd.DataFrame({"PassengerId":PassengerId, "Survived":test_pred})
output.to_csv("submission.csv",index=False)
print(output.sample(3))


# In[ ]:





# In[ ]:





# In[ ]:




