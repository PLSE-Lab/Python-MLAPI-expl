#!/usr/bin/env python
# coding: utf-8

# # MNIST Predicting handwritten symbols
# 
# 
# 
# ## Import Libraries

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV


# ## Load Data
#     Each row is a separate image
#     785 columns
#     First column = class_label (see mappings.txt for class label definitions)
#     Each column after represents one pixel value (784 total for a 28 x 28 image)

# In[3]:


train = pd.read_csv("../input/emnist-balanced-train.csv",delimiter = ',')
test = pd.read_csv("../input/emnist-balanced-test.csv", delimiter = ',')

#print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)


# In[4]:


train_x = train.iloc[:,1:]
train_y = train.iloc[:,0]
del train

test_x = test.iloc[:,1:]
test_y = test.iloc[:,0]
del test


# In[5]:


print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)


# ## Visualization

# In[6]:


img1 = [[0]*28]*28
img1 = np.array(img1)
for i in range(28):
    img1[:,i]=train_x.iloc[1,i*28:i*28+28]

fig,ax = plt.subplots()
im = ax.imshow(img1,cmap='Greys')


# ## Model
# 
# ### KNN
# 
# KNN is a non-generalizing method but it has good result in some problems.
# For this problem the pixels positions are the most important trait and theres not much overlapping so KNN is a good model for this problem.

# In[7]:


n_neighbors = 2
weights = 'uniform'

model = neighbors.KNeighborsClassifier(n_neighbors, n_jobs = -1, weights=weights)
model.fit(train_x,train_y)


# In[8]:


print(model.score(test_x,test_y))


# KNN gets a 76% accuracy but this can be improved with some hyperparamter tuning on the number of neighbors and the distance function:

# In[9]:


parameters = {"n_neighbors": np.arange(1,35,2), "metric": ["euclidean","cityblock"]}

tuned_model = GridSearchCV(model,parameters)
tuned_model.fit(train_x[:10000],train_y[0:10000])
#tuned_model.score(test_x,test_y)

bestparams = tuned_model.best_params_
print(bestparams)


# In[10]:


model2 = neighbors.KNeighborsClassifier(n_neighbors = bestparams['n_neighbors'], n_jobs = -1, weights=weights, metric = bestparams['metric'])
model2.fit(train_x,train_y)
print(model2.score(test_x,test_y))


# With some hyperparameter tuning we increased the accuracy to 78% using the K nearest negihbor model
