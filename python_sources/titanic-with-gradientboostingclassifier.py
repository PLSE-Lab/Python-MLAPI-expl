#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import copy
from sklearn.model_selection import train_test_split


# In[53]:


train = pd.read_csv('../input/train.csv')


# In[54]:


train.head()


# In[55]:


for i in train.columns:
    print(i, len(set(train[i])))

# so PassengerId and Name is useless


# In[56]:


print(train.isnull().any())
print("")
for i in ['Age', 'Cabin', 'Embarked']:
    print(i, len(train[train[i].isnull()]))


# In[57]:


train[train['Embarked'].isnull()]


# In[58]:


train[~train['Embarked'].isnull()].groupby(['Embarked','Pclass']).agg({'Embarked' : {'count'}})


# In[59]:


train.iloc[[61,829],11] = 'C'


# In[60]:


train.iloc[train[train['Age'].isnull()].index,5] = -1


# In[61]:


cabin_train = train[~train['Cabin'].isnull()].copy()


# In[62]:


cabin_train['Cabin'] = cabin_train['Cabin'].apply(lambda x : x[0])


# In[63]:


cabin_train[~cabin_train['Cabin'].isnull()].groupby(['Cabin']).agg({'Fare' : {'count','min','max','mean'}})


# In[64]:


useful_train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


# In[65]:


y_train = useful_train['Survived']
x_train = useful_train.drop(labels = ['Survived'], axis = 1)


# In[66]:


x_train = pd.get_dummies(x_train, prefix=['Sex', 'Embarked'])


# In[67]:


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=1024)


# In[70]:


parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[8,10],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.8, 0.9,1.0],
    "n_estimators":[50,80]
    }

estimator = GradientBoostingClassifier(random_state = 1024)
model = GridSearchCV(estimator = estimator, param_grid = parameters, cv=5, scoring = 'accuracy')


# In[ ]:


model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
print(model.best_params_)


# In[ ]:


result = pd.read_csv('../input/test.csv')


# In[ ]:


result.isnull().any()


# In[ ]:


useful_result = result[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


# In[ ]:


y_result = useful_result['Survived']
x_result = useful_result.drop(labels = ['Survived'], axis = 1)


# In[ ]:


x_result = pd.get_dummies(x_result, prefix=['Sex', 'Embarked'])


# In[ ]:


predict = model.predict(x_result)


# In[ ]:


submission = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


submission['Survived'] = predict


# In[ ]:


submission.to_csv('./gender_submission.csv', index = False)


# In[ ]:




