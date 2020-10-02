#!/usr/bin/env python
# coding: utf-8

# # Introduction

# There're more ways how to approache the problem. *OHE will fail* in this case on Kaggle bacause of some bug in a recent update (as to fall 2019) which makes spare matrix output parameter inactive. We can *CatBoots* which would be probably the most straight-forward way how to get good results. But I wanted to take the oppurtinity here to show how simple *target (also known as mean or likely-hood) encoding* can be.
# 
# An interesting thing to do would be to use other encoding methods from *category_encoders* library to compare them in between.

# # Libraries

# In[ ]:


# Basic
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Tools
from category_encoders import TargetEncoder
#import category_encoders as ce # You can import whole library and play around with that

# Dataset
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load data

# In[ ]:


# Train
train_df = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')
train_df.shape


# In[ ]:


# Test
test_df = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
test_df.shape


# # EDA

# In[ ]:


# Have a look
train_df.head()


# In[ ]:


# Data types
train_df.dtypes


# In[ ]:


# Number of null values
train_df.isnull().sum().sum()


# # Encoding

# In[ ]:


# Differentiate
features = train_df.drop('target',axis=1)
target = train_df.target


# In[ ]:


# Select categorical features
cols = train_df.columns
nums = train_df._get_numeric_data().columns
cats = list(set(cols) - set(nums))


# In[ ]:


# Define target encoder
enc = TargetEncoder(cols=cats).fit(features, target)

# Encode
train_enc = enc.transform(features, target)
test_enc = enc.transform(test_df)


# In[ ]:


# This will fail because of some bug in pandas library (fall 2019)
'''# One Hot Encoding
df_ohe = pd.get_dummies(df, columns=cats, drop_first=True)
df_ohe.shape

from scipy.sparse import csr_matrix

df_ohe = csr_matrix(df_ohe.values)
df_ohe.memory_usage().sum()'''


# # Model

# In[ ]:


# Get training features and the target
X = train_enc
y = target


# In[ ]:


# Make a traning and validation dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=42)


# In[ ]:


# Define model
from sklearn import linear_model

lr = linear_model.LogisticRegression(
    solver='lbfgs', 
    max_iter=5000, 
    fit_intercept=True,
    random_state=42, 
    penalty='none', 
    verbose=0)

# Train model
lr.fit(X_train, y_train)


# In[ ]:


# Validate the model
from sklearn.metrics import accuracy_score

y_pre = lr.predict(X_test)
print('Accuracy : ',accuracy_score(y_test, y_pre))


# # Submission

# In[ ]:


# Predict test values
pred = lr.predict(test_enc).astype(np.int)
sub = pd.DataFrame({'id':test_df['id'], 'target':pred})


# In[ ]:


# Have a look
sub.head()


# In[ ]:


# Make a submission file
sub.to_csv('submission.csv',index=False)


# **If that helps, please up-vote!**
