#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))


# In[20]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[21]:


train_data.head()


# Getting a bit of more info.

# In[22]:


train_data.info()


# * Some of the `Age` values are missing.
# * Most of the `Cabin` values are missing.
# * Only 2 of the `Embarked` values are missing.

# The `Pclass` attribute can be interesting. Can it be that most of the elite class peopel survived? We may dive into that later.

# Taking a look at the numerical attributes.

# In[23]:


train_data.describe()


# Let us see the survived and not-survived numbers.

# In[24]:


train_data['Survived'].value_counts()


# Looks like more than half the people did not survive.

# Let us build a preprocess pipeline to use categorical and numerical attributes effectively.

# In[25]:


from sklearn.base import BaseEstimator, TransformerMixin

class AttributeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attributes_names]


# We should build a pipeline for the numerical attributes. We do not want to miss out on important features due to missing values and NaNs.

# In[26]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numerical_pipeline = Pipeline([
    ('select_numeric', AttributeSelector(['Age', 'SibSp', 'Parch'])),
    ('imputer', SimpleImputer(strategy='median')) # Replacing with median values
])


# In[27]:


# Using the `numerical_pipeline`
numerical_pipeline.fit_transform(train_data)


# What about the missing values in the string categorical attributes? We will need a different approach for that.

# In[28]:


class CategoricalImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[att].value_counts().index[0] for att in X],
                                       index=X.columns)
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# **Pclass** can be an interesting attribute. Another categorical attribute is the **Embarked** attribute. We should not be ignoring the categorical attributes. Also, we would be using `OneHotEncoder` to creat dummy variables.

# In[29]:


from sklearn.preprocessing import OneHotEncoder


# Building the categorical attributes' pipeline.

# In[30]:


categorical_pipeline = Pipeline([
    ('select_cat', AttributeSelector(['Pclass', 'Sex', 'Embarked'])),
    ('imputer', CategoricalImputer()),
    ('cat_encoder', OneHotEncoder(sparse=False)),
])


# In[31]:


categorical_pipeline.fit_transform(train_data)


# In[32]:


from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[
    ('numerical_pipeline', numerical_pipeline),
    ('categorical_pipeline', categorical_pipeline),
])


# In[33]:


X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data['Survived']


# **Trying `RandomForestClassifier`**:

# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict

rnd_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_forest_clf.fit(X_train, y_train)
rnd_forest_scores = cross_val_score(rnd_forest_clf, X_train, y_train, cv=10)
rnd_forest_scores.mean()


# In[35]:


X_test = preprocess_pipeline.transform(test_data)
y_pred = rnd_forest_clf.predict(X_test)
y_pred


# In[36]:


submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




