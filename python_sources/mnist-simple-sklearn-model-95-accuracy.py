#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt


# In[3]:


import os
print(os.listdir("../input"))


# In[4]:


# Set our train and test date
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[5]:


train_df.head()


# In[6]:


# data size
train_df.shape


# In[7]:


# Set features and label for showing
digits = train_df.drop(['label'], 1).values
digits = digits / 255
label = train_df['label'].values


# In[8]:


# Show 25 digits of data
fig, axis = plt.subplots(5, 5, figsize=(22, 20))

for i, ax in enumerate(axis.flat):
    ax.imshow(digits[i].reshape(28, 28), cmap='binary')
    ax.set(title = "Real digit is {}".format(label[i]))


# # 2. Machine Learning

# In[9]:


# Machine Learning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[10]:


digits.shape


# In[11]:


# Set X, y for fiting
X = digits
y = label
#X_test = test_df.values # file data


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## 2.1 Random Forest Classifier Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Seting our model
model = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test) # predict our file test data


# In[ ]:


print("Model accuracy is: {0:.3f}%".format(accuracy_score(y_test, y_pred) * 100))


# In[ ]:


# Compare our result
fig, axis = plt.subplots(5, 5, figsize=(18, 20))

for i, ax in enumerate(axis.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='binary')
    ax.set(title = "Predicted digit {0}\nTrue digit {1}".format(y_pred[i], y_test[i]))


# In[ ]:


test_X = test_df.values
rfc_pred = model.predict(test_X)


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub.head()


# In[ ]:


# Make submission file
sub['Label'] = rfc_pred
sub.to_csv('submission.csv', index=False)


# In[ ]:


# Show our submission file
sub.head(10)


# In[ ]:




