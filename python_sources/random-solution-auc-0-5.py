#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


y = train['target']
X = train.drop(['target', 'ID_code'], axis = 1)


# In[ ]:


pred_y = pd.Series(0, index = y.index)
np.random.seed(1)
for i in y.index:
    if( np.random.uniform(low = 0, high = 1) >= 0.90 ):
        pred_y[i] = 1

pred_y[pred_y == 1].count()


# In[ ]:


confusion_matrix(y, pred_y)


# In[ ]:


accuracy_score(y, pred_y)


# In[ ]:


roc_auc_score(y, pred_y)

