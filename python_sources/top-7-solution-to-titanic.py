#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


titanic_train = pd.read_csv('../input/titanic/train.csv')
titanic_test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


titanic_train.head(10)


# In[ ]:


titanic_train.shape


# In[ ]:


titanic_test.shape


# In[ ]:


titanic_train.nunique()


# In[ ]:


len_train= len(titanic_train)
new_df = pd.concat([titanic_train, titanic_test], sort=True)


# In[ ]:


new_df.isnull().sum()


# In[ ]:


new_df.fillna(-999, inplace=True)


# In[ ]:


new_df.isnull().sum()


# In[ ]:


train = new_df[:len_train]
test = new_df[len_train:].reset_index(drop=True)
test.drop("Survived", axis=1, inplace=True)


# In[ ]:


target= train["Survived"]
train =train.drop(["PassengerId", "Survived"], axis=1)
test_id = test["PassengerId"]
test= test.drop(["PassengerId"], axis=1)


# In[ ]:


X = train
y= target


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, random_state=1234)


# In[ ]:


categorical_features_indices = np.where(X.dtypes != np.float)[0]


# In[ ]:


#importing library and building model
model=CatBoostClassifier(
    one_hot_max_size=4,
    iterations=100,
    random_seed=10,
    verbose=False,
    eval_metric='Accuracy'
)
model.fit(X_train, y_train,cat_features=categorical_features_indices,plot=True)


# In[ ]:


Cat_pred = model.predict(test)


# In[ ]:


Cat_pred.astype(int)


# In[ ]:


#SUBMISSION 
submission = pd.DataFrame({"PassenderId": test_id, "Survived": Cat_pred})
(submission["Survived"]==1).value_counts()


# In[ ]:


#SUBMISSION 
submission = pd.DataFrame({"PassengerId":test_id, "Survived": Cat_pred})
(submission["Survived"]==1).value_counts()


# In[ ]:


submission.Survived = submission.Survived.apply(int)
submission.head()


# In[ ]:


submission.to_csv("Submit.csv",  index=False)

