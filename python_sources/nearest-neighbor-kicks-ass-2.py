#!/usr/bin/env python
# coding: utf-8

# I think this is so cool, though probably well-known.  So I just had to share it.  The idea is to use KNN to correct some of its own errors.  First we copy the KNN model from [Nearest Neighbor kicks ass](https://www.kaggle.com/chrisfreiling/nearest-neighbor-kicks-ass).  This first model uses just one neighbor.  Then we see if it can self-improve by applying a second KNN on the train and test set together.  In training this second model, the labels on the test set are taken from the first model.  This second model uses lots of neighbors (because it is trained on lots more data).  The purpose of the second model is to smooth out the decision boundary from the first model.  It seems to work!  The accuracy went from .79817 to .80611.  Okay, not a huge improvement, but I still think it's pretty cool!  There is lots more to explore.  For example, what happens if we do it again?  Can we applly this to correct other models? etc.

# In[ ]:


import os
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Read the data
X_full = pd.read_csv('../input/learn-together/train.csv', index_col='Id')
X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id')

#Drop a couple of useless columns
X_full.drop('Soil_Type7', axis=1, inplace=True)
X_test.drop('Soil_Type7',axis=1, inplace=True)
X_full.drop('Soil_Type15', axis=1, inplace=True)
X_test.drop('Soil_Type15',axis=1, inplace=True)

# Separate target from predictors 
y_full = X_full.Cover_Type
X_full.drop(['Cover_Type'], axis=1, inplace=True)


# Here is the list of weights from [Nearest Neighbor kicks ass](https://www.kaggle.com/chrisfreiling/nearest-neighbor-kicks-ass).   The first ten are for the first ten features. The eleventh one is for the four "Wilderness_Area" features, and the last one is for all of the "Soil_Type" features.  If you are interested, see [Nearest Neighbor kicks ass](https://www.kaggle.com/chrisfreiling/nearest-neighbor-kicks-ass) for the code used to get this list. These weights need to be optimized better!  I believe a lot of improvement can be made here. 

# In[ ]:


weights = [
3.3860658430898054,
0.4163438499758126,
7.35783588470092,
1.4635508470705287,
2.512455585483701,
0.7879386244955993,
2.3361452772106412,
4.509437549105931,
1.2565844481748276,
0.8105744594321818,
357.62840785739945,
195.87206818235353]


# In[ ]:


# Apply weights to copies of data
X_full_copy = X_full.copy()
X_test_copy = X_test.copy()

for i in range(10):
    c = X_full.columns[i]
    X_full_copy[c] = weights[i]*X_full_copy[c]
    X_test_copy[c] = weights[i]*X_test_copy[c]
for i in range(10,14):
    c = X_full.columns[i]
    X_full_copy[c] = weights[10]*X_full_copy[c]
    X_test_copy[c] = weights[10]*X_test_copy[c]
for i in range(14,len(X_full.columns)):
    c = X_full.columns[i]
    X_full_copy[c] = weights[11]*X_full_copy[c]
    X_test_copy[c] = weights[11]*X_test_copy[c]


# In[ ]:


# Use first KNN model with n_neighbors=1
model = KNeighborsClassifier(n_neighbors=1, p=1)
model.fit(X_full_copy, y_full)
preds_full = model.predict(X_full_copy)
y_test = model.predict(X_test_copy)


# In[ ]:


# Put (X_full, y_full) and (X_test,y_test) together into big list (X_all, y_all)
X_all = X_full_copy.append(X_test_copy)
y_test = pd.Series(y_test)
y_all = y_full.append(y_test)


# In[ ]:


# Use a second KNN model with lots of neighbors on big set
model2 = KNeighborsClassifier(n_neighbors=101, p=1)
model2.fit(X_all, y_all)
preds_all = model2.predict(X_all)


# In[ ]:


# Recover just the test portion
n_full = len(y_full)
preds_test = preds_all[n_full:]


# In[ ]:


# Make the submission file
output = pd.DataFrame({'Id': X_test_copy.index,'Cover_type': preds_test})
output.to_csv('submission.csv', index=False)

