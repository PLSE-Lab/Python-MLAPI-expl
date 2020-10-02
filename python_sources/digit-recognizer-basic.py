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


# **Loading data using Pandas DF.**

# In[ ]:


# Load data using Pandas
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

y = train_data['label'] # Fetches only the 'label' column information
X = train_data[train_data.columns.difference(['label'])] # Fetches entire data excluding the 'label' column

X_test = test_data


# **Splitting the data into Train Set and Validation Set**

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.33, random_state=42)


# In[ ]:


X_train = X_train.to_numpy()


# In[ ]:


y_train = y_train.to_numpy()


# In[ ]:


X_valid = X_valid.to_numpy()


# In[ ]:


y_valid = y_valid.to_numpy()


# In[ ]:


X_test = X_test.to_numpy()


# In[ ]:


# Data Normalization
X_train = X_train/255
y_valid = y_valid/255
X_test = X_test/255
print(X_train.shape)


# In[ ]:


from sklearn.linear_model import SGDClassifier
finalModel = SGDClassifier(loss="hinge", penalty="l2")
finalModel.fit(X_train, y_train)


# In[ ]:


values = finalModel.predict(X_valid)
values


# In[ ]:


def submissions(finalModel):
    values = finalModel.predict(X_test)
    indexes = [index+1 for index, value in enumerate(values)]
    dataFrame = pd.DataFrame(list(zip(indexes, values)), columns=['ImageId', 'Label'])
    return dataFrame


# In[ ]:


dataFrame = submissions(finalModel)
dataFrame.to_csv('submissions.csv', index=False)


# In[ ]:


dataFrame.shape

