#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
df


# # Define y, X, model

# ## y

# In[ ]:


y = df.species


# In[ ]:


df.columns


# ## X

# In[ ]:


features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features]
X


# In[ ]:


X = df.drop(columns=['species'])
X


# In[ ]:


df


# In[ ]:


from sklearn.model_selection import train_test_split 
train_X, test_X, train_y, test_y = train_test_split(X, y)


# In[ ]:


train_X


# In[ ]:


train_y


# In[ ]:


test_X


# In[ ]:


test_X.shape


# ## Model

# In[ ]:


from sklearn.neighbors import NearestCentroid
model = NearestCentroid()


# # train model 

# In[ ]:


model.fit(train_X, train_y)


# # prediction

# In[ ]:


preds = model.predict(test_X)


# In[ ]:


preds[:5]


# In[ ]:


test_y[:5]


# In[ ]:


from sklearn.metrics import accuracy_score 
print(accuracy_score(y_true=test_y, y_pred=preds))


# In[ ]:


df.columns


# In[ ]:


test_example = pd.DataFrame(data={
    'sepal_length': [4.9],
    'sepal_width': [1.4], 
    'petal_length': [5.3], 
    'petal_width': [1.1]
})


# In[ ]:


model.predict(test_example)

