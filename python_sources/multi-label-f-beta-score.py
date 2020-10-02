#!/usr/bin/env python
# coding: utf-8

# # Calculate F score on multi-label classification task with scikit-learn and scipy.sparse
# ---

# In[ ]:


import numpy as np
import pandas as pd
import os

from scipy.sparse import lil_matrix
from sklearn.metrics import fbeta_score


# ## single-label
# example from [scikit-learn fbeta_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)

# In[ ]:


y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]


# In[ ]:


fbeta_score(y_true, y_pred, average='macro', beta=0.5)


# ## multi-label

# In[ ]:


y_true = [[0, 1], [1], [1, 2], [0], [1], [0, 2]]
y_pred = [[0], [0, 2], [1, 2], [2], [0, 1], [1, 2]]


# In[ ]:


# fbeta_score(y_true, y_pred, average='macro', beta=0.5)
# -> ValueError: You appear to be using a legacy multi-label data representation. Sequence of sequences are no longer supported; use a binary array or sparse matrix instead.


# ### Convert into sparse matrix

# In[ ]:


def label_to_sm(labels, n_classes):
    sm = lil_matrix((len(labels), n_classes))
    for i, label in enumerate(labels):
        sm[i, label] = 1
    return sm


# In[ ]:


y_true_sm = label_to_sm(labels=y_true, n_classes=3)
y_true_sm.toarray()


# In[ ]:


y_pred_sm = label_to_sm(labels=y_pred, n_classes=3)
y_pred_sm.toarray()


# In[ ]:


fbeta_score(y_true_sm, y_pred_sm, average='macro', beta=0.5)


# yay!
