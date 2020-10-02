#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
training = pd.read_csv('../input/train.csv')
training.head()


# In[ ]:


training.info()


# In[ ]:


training.describe()


# In[ ]:


training['Sex'].value_counts()


# In[ ]:


training['Ticket'].value_counts()


# In[ ]:


training['Cabin'].value_counts()


# In[ ]:


training['Embarked'].value_counts()


# %matplotlib inline
# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
training.hist(bins=50, figsize=(20,15))
plt.show()


# In[ ]:


corr_matrix = training.corr()
corr_matrix['Survived'].sort_values(ascending=False)


# In[ ]:


from pandas.tools.plotting import scatter_matrix
attributes = ['Survived', 'Fare', 'Pclass']
scatter_matrix(training[attributes], figsize=(12,8))


# In[ ]:


# there are only two rows with 'Embarked' null so we can just drop those rows
training = training.dropna(subset=['Embarked'])
# The majority of rows are missing the 'Cabin' attribute so for now I might just drop this
training = training.drop('Cabin', axis=1)
# Quite a few rows are missing age, let's just replace it with the median
median = training['Age'].median()
training['Age'] = training['Age'].fillna(median)
training.info()


# In[ ]:


# for simplicity I'm going to get rid of some features for now
training = training.drop(['PassengerId', 'Ticket', 'Name', 'Embarked'], axis=1)
# Let's split up the targets and features
targets = training['Survived']
features = training.drop('Survived', axis=1)


# In[ ]:


features.head()


# In[ ]:


# We have to convert categorical text attributes into integer categories
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

cat_attribs = ['Sex']
cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                         ('label_binarizer', LabelBinarizer()),
                         ])

num_attribs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                         ('std_scaler', StandardScaler()),
                        ])

full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),
                                               ("cat_pipeline", cat_pipeline),
                                              ])


# In[ ]:


features.head()


# In[ ]:


features_prepared = full_pipeline.fit_transform(features)


# In[ ]:


features_prepared


# In[ ]:




