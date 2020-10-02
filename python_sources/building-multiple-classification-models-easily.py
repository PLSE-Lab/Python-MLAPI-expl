#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set()


# In[ ]:


data = pd.read_csv('/kaggle/input/gender-classification/Transformed Data Set - Sheet1.csv')
data.head()


# In[ ]:


X = data.drop(['Gender'], axis = 1)
y = data.Gender
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Lazypredict a python library being built by me. You can read the documentation on how to use it here:
# https://lazypredict.readthedocs.io/en/latest/usage.html

# In[ ]:


get_ipython().system('pip install lazypredict')


# In[ ]:


from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)


# In[ ]:


models.to_csv('models.csv')
predictions.to_csv('predictions.csv')


# In[ ]:


models

