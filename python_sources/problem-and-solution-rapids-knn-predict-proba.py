#!/usr/bin/env python
# coding: utf-8

# ## The purpose of this notebook
# 
# I think most of us are familiar with [this great kernel](https://www.kaggle.com/cdeotte/rapids-knn-30-seconds-0-938) by Chris Deotte showing how to build a very promissing KNN model on the competition data using RAPIDS. A few days ago, I decided to play with this model and discovered a very annoying bug in the RAPIDS implementation of the KNN algorithm: its `predict_proba()` method returns a binary outcome of zeros and ones instead of actual probabilities. The good new is that Chris has already submitted a bug report, so hopefully it will be fixed soon. The bad news is that fixing it might take a while and the competition might be over by then. Fortunately, Chris suggested a clever workaround which I am going to share with you in this notebook. 
# 
# I will start by generating a simple data set for classification. Then I will train a RAPIDS KNN model on these data and try to make predictions using the `predict()` method first and the `predict_proba()` method second. The former will successfully return the class labels but the latter will only return binary outcomes. Then I will show you how to get actual KNN probabilities using the `kneighbors()` method of the RAPIDS `NearestNeighbors()` class. 
# 
# ## Loading RAPIDS

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport sys\n!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null\nsys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\nsys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path\nsys.path = ["/opt/conda/envs/rapids/lib"] + sys.path \n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


from cuml.neighbors import KNeighborsClassifier, NearestNeighbors
import cuml; cuml.__version__


# ## Generate data
# 
# Generate data for KNN classifier; then train the classifier on these data.

# In[ ]:


from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=100, 
                  centers=5,
                  cluster_std=5.0,
                  n_features=4)

knn = KNeighborsClassifier(n_neighbors=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

knn.fit(X_train, y_train)


# Here is a gilimpse of our test data:

# In[ ]:


X_test.shape, y_test.shape


# In[ ]:


y_test


# ## Predictions
# 
# Let's try to predict classes (labels):

# In[ ]:


knn.predict(X_test)


# Predicting the labels seems to be working just fine. Let's try predicting probabilities using the `predict_proba` method.

# In[ ]:


knn.predict_proba(X_test)


# Predicting probabilities fails: we only get binary zeros and ones, not the actual probabilities (in the case at hands we got only zeros -- it is possible to change the parameters of the `make_blobs()` function to get both zeros and ones).
# 
# ## Workaround
# 
# The idea of the workaround is very simple: let's find the train set indecies of the nearest neighbours for all points in the test set. Here is how it can be done: 

# In[ ]:


import numpy as np

KNN=10
batch=5

clf = NearestNeighbors(n_neighbors=KNN)

clf.fit(X_train)

distances, indices = clf.kneighbors(X_test)

ct = indices.shape[0]

pred = np.zeros((ct, KNN),dtype=np.int8)

probabilities = np.zeros((ct, len(np.unique(y_train))),dtype=np.float32)

it = ct//batch + int(ct%batch!=0)

for k in range(it):
    
    a = batch*k; b = batch*(k+1); b = min(ct,b)
    pred[a:b,:] = y_train[ indices[a:b].astype(int) ]
    
    for j in np.unique(y_train):
        probabilities[a:b,j] = np.sum(pred[a:b,]==j,axis=1)/KNN


# In[ ]:


probabilities


# We have successfully computed the probabilities! This method works pretty fast on the competition data -- it is possible to build it in a 5-fold cross-validation algorithm.
