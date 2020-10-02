#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.describe(include='all')


# In[ ]:


train.head()


# In[ ]:


train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)


# In[ ]:


train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)


# In[ ]:


train = train.drop(['Embarked'], axis = 1)
test = test.drop(['Embarked'], axis = 1)


# In[ ]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# In[ ]:


corr = train.corr()
fig, ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot=True, linewidths=1.5 , fmt='.2f',ax=ax)
plt.show()


# In[ ]:


sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# In[ ]:


print(pd.isnull(test).sum())


# In[ ]:


print(pd.isnull(train).sum())


# In[ ]:


train['Age'].fillna((train['Age'].mean()), inplace=True)
test['Age'].fillna((test['Age'].mean()), inplace=True)


# In[ ]:


train['Fare'].fillna((train['Fare'].mean()), inplace=True)
test['Fare'].fillna((test['Fare'].mean()), inplace=True)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train[['Fare']] = scaler.fit_transform(train[['Fare']])
test[['Fare']] = scaler.fit_transform(test[['Fare']])


# In[ ]:


train[['Age']] = scaler.fit_transform(train[['Age']])
test[['Age']] = scaler.fit_transform(test[['Age']])


# In[ ]:


train.head()


# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)


# In[ ]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[ ]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = gbk.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

