#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import BaggingClassifier
from keras.callbacks import LearningRateScheduler


# In[ ]:


train_x = pd.read_csv('../input/train_x.csv', index_col=0, header=None)
train_y = pd.read_csv('../input/train_y.csv', index_col=0)
test_x = pd.read_csv('../input/test_x.csv', index_col=0, header=None)


# In[ ]:


mappping_type = {'Bird': 0, 'Airplane': 1}
train_y = train_y.replace({"target": mappping_type})


# In[ ]:


# _train_x = train_x.values.reshape(7200,32,32,3)
# _test_x = test_x.values.reshape(4800,32,32,3)


# In[ ]:


train_feature_matrix, test_feature_matrix = train_x/255, test_x/255


# In[ ]:


clf = SGDClassifier(max_iter=1000, tol=1e-3, loss='log', penalty='l2')
clf.fit(train_feature_matrix, np.ravel(train_y))


# In[ ]:


accuracy_score(clf.predict(train_feature_matrix), np.ravel(train_y))


# In[ ]:


predict_y = clf.predict_proba(test_x)


# In[ ]:


sample = pd.DataFrame(np.array([[i, x.argmax()] for i, x in enumerate(predict_y)]), columns=['id', 'target'])

mappping_type_inv = {0: 'Bird', 1: 'Airplane'}
sample = sample.replace({'target': mappping_type_inv})


# In[ ]:


sample.to_csv('submit.csv', index=False)

