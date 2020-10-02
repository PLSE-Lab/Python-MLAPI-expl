#!/usr/bin/env python
# coding: utf-8

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html

# In[ ]:


import numpy as np
from sklearn.model_selection import ShuffleSplit


# In[ ]:


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
y = np.array([1, 2, 1, 2, 1, 2])


# In[ ]:


cv = ShuffleSplit(n_splits=5, train_size=0.5, test_size=.25, random_state=0)
for train_index, test_index in cv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)


# In[ ]:




