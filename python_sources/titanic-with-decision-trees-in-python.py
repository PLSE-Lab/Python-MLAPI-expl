#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder


# In[7]:


data = pd.read_csv("../input/train.csv")
submit = pd.read_csv("../input/test.csv")


# In[8]:


#train['Sex'] = train.Sex.astype('category')
lb = LabelEncoder()
data['Embarked'] = lb.fit_transform(data['Embarked'].astype(str))
data['Sex'] = lb.fit_transform(data['Sex'].astype(str))
data['Cabin'] = lb.fit_transform(data['Cabin'].astype(str))
data = data.fillna(data.median())


# In[9]:


train = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']]
target = data[['Survived']]
submit = submit[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']]


# In[11]:


decisiontree = DecisionTreeClassifier(random_state = 10)
cross_val_score(decisiontree, train, target, cv=30).mean()


# In[ ]:





# In[ ]:




