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


train_data = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')


# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


for col in train_data.columns:
    print(col, train_data[col].nunique())


# Drop columns with high cardinality

# In[ ]:


categorical_columns = [col for col in train_data.columns if train_data[col].dtype == object]
columns_with_high_cardinality = [col for col in categorical_columns if train_data[col].nunique() > 30]

train_data.drop(columns_with_high_cardinality, axis=1, inplace=True)


# In[ ]:


train_data


# In[ ]:


from sklearn.model_selection import train_test_split

y = train_data['target']
train_data.drop(['target'], axis=1, inplace=True)
train_X, valid_X, train_y, valid_y = train_test_split(train_data, y, train_size=0.8, test_size=0.2, random_state=0)

Impute missing values
# In[ ]:


for col in train_X.columns:
    mode = train_X[col].mode()[0]
    train_X[col].fillna(mode, inplace=True)
    valid_X[col].fillna(mode, inplace=True)


# In[ ]:


train_X


# OneHot-encode columns with low cardinality

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

columns_oh_encode = [col for col in train_X.columns if train_X[col].dtype == object]

oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

column_transformer = ColumnTransformer(transformers=[
    ('onehot', oh_encoder, columns_oh_encode)
])


# XGBClassifier

# In[ ]:


from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

model = XGBClassifier(n_estimators=100, random_state=0, n_jobs=-1)

pipeline = Pipeline(steps=[
    ('transformer', column_transformer),
    ('model', model)
])

pipeline.fit(train_X, train_y)
prediction = pipeline.predict_proba(valid_X)


# In[ ]:


from sklearn.metrics import roc_auc_score

score = roc_auc_score(valid_y, prediction[:,1])
print(score)

Predict the test data.
# In[ ]:


test_data = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')

test_data.drop(columns_with_high_cardinality, axis=1, inplace=True)

#
for col in train_data.columns:
    mode = train_data[col].mode()[0]
    train_data[col].fillna(mode, inplace=True)
    test_data[col].fillna(mode, inplace=True)

#
pipeline.fit(train_data, y)
prediction_test = pipeline.predict_proba(test_data)


# In[ ]:


output = pd.DataFrame({
    'id': test_data.index,
    'target': prediction_test[:,1]
})
output.to_csv('submission.csv', index=False)

