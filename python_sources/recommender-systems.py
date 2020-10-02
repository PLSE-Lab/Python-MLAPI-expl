#!/usr/bin/env python
# coding: utf-8

# # Based on [recommender system post](https://machinelearningmedium.com/2018/05/11/recommender-systems/)
# 
# ### Full Workbook - https://github.com/shams-sam/CourseraMachineLearningAndrewNg/blob/master/RecommenderSystem.ipynb

# In[1]:


import numpy as np


# # Dummy Rating Matrix

# In[2]:


y = np.array(
    [
        [3. , 0. , 4.5, 4. , 2. ],
        [3. , 4. , 3.5, 5. , 3. ],
        [0. , 0. , 3. , 5. , 3. ],
        [4. , 0. , 3. , 0. , 0. ],
        [0. , 0. , 5. , 5. , 3.5],
        [0. , 0. , 5. , 4. , 3.5],
        [0. , 5. , 5. , 5. , 4.5],
        [4. , 4. , 2.5, 5. , 0. ],
        [0.5, 0. , 4. , 0. , 2.5],
        [0. , 0. , 0. , 4. , 0. ]
    ]
)
r = np.where(y > 0, 1, 0)


# # Using Iterative Algorithm

# In[4]:


def estimate_x_v2(y, max_k=2, x=None, theta=None,
               _alpha = 0.01, _lambda=0.001, _tolerance = 0.001):
    r = np.where(y > 0, 1, 0)
    converged = False
    max_i, max_j = y.shape
    if type(x) != np.array:
        x = np.random.randn(max_i, max_k)
    if type(theta) != np.array:
        theta = np.random.randn(max_j, max_k+1)
    while not converged:
        update_x = np.zeros(x.shape)
        update_x = _alpha * (
            np.matmul(
                (
                    np.matmul(
                        np.hstack((np.ones((x.shape[0], 1)),x)), 
                        theta.transpose()
                    ) - y
                ) * r, 
                theta
            )[:, 1:] + _lambda * x
        )
        x = x - update_x
        if np.max(abs(update_x)) < _tolerance:
            converged = True
    return theta, x

def estimate_theta_v2(y, max_k=2, x=None, theta=None,
               _alpha = 0.01, _lambda=0.001, _tolerance = 0.001):
    r = np.where(y > 0, 1, 0)
    converged = False
    max_i, max_j = y.shape
    if type(x) != np.array:
        x = np.random.randn(max_i, max_k)
    if type(theta) != np.array:
        theta = np.random.randn(max_j, max_k+1)
    while not converged:
        update_theta = np.zeros(theta.shape)
        update_theta = _alpha * (
            np.matmul(
                np.hstack((np.ones((x.shape[0], 1)),x)).transpose(),
                (
                    np.matmul(
                        np.hstack((np.ones((x.shape[0], 1)),x)), 
                        theta.transpose()
                    ) - y
                ) * r, 
            ).transpose() + _lambda * theta
        )
        theta = theta - update_theta
        if np.max(abs(update_theta)) < _tolerance:
            converged = True
    return theta, x


# In[6]:


tolerance=0.001
max_k=50
theta, x = estimate_x_v2(y, _tolerance=tolerance, max_k=max_k)
for _ in range(2):
    theta, x = estimate_theta_v2(y, x=x, theta=theta, _tolerance=tolerance, max_k=max_k)
    theta, x = estimate_x_v2(y, x=x, theta=theta, _tolerance=tolerance, max_k=max_k)


# In[7]:


y


# In[8]:


np.matmul(np.hstack((np.ones((10, 1)), x)), theta.transpose()).round(decimals=2)


# # Using Collaborative Filtering Algorithm

# In[9]:


def colaborative_filtering_v2(y, max_k=2,
             _alpha=0.01, _lambda=0.001, _tolerance=0.001, r=None):
    if type(r) != np.ndarray:
        r = np.where(y>0, 1, 0)
    converged = False
    max_i, max_j = y.shape
    x = np.random.rand(max_i, max_k)
    theta = np.random.rand(max_j, max_k)
    
    while not converged:
        update_x = np.zeros(x.shape)
        update_theta = np.zeros(theta.shape)
        update_x = _alpha * (
            np.matmul(
                (np.matmul(x, theta.transpose()) - y) * r, 
                theta
            ) + _lambda * x
        )
        update_theta = _alpha * (
            np.matmul(
                x.transpose(),
                (np.matmul(x, theta.transpose()) - y) * r, 
            ).transpose() + _lambda * theta
        )
        x = x - update_x
        theta = theta - update_theta
        if max(np.max(abs(update_x)), np.max(abs(update_theta))) < _tolerance:
            converged = True
    return theta, x


# In[10]:


theta, x = colaborative_filtering_v2(y, max_k=max_k)


# In[11]:


y


# In[13]:


np.matmul(x, theta.transpose()).round(decimals=2)


# # Case of User with No Ratings
# - if a user has no ratings the algorithm will predict all low ratings by that user.

# In[17]:


y = np.hstack((y, np.zeros((y.shape[0], 1))))


# In[18]:


max_k = 5
tolerance = 0.0000001
theta, x = colaborative_filtering_v2(y, max_k=max_k, _tolerance=tolerance)


# In[19]:


y


# In[20]:


np.matmul(x, theta.transpose()).round(decimals=2)


# # Mean Normalization issue of user with no ratings

# In[21]:


def normalized(y, max_k=2,
             _alpha=0.01, _lambda=0.001, _tolerance=0.001):
    r = np.where(y>0, 1, 0)
    y_sum = y.sum(axis=1)
    r_sum = r.sum(axis=1)
    y_mean = np.atleast_2d(y_sum/r_sum).transpose()
    y_norm = y - y_mean
    theta, x = colaborative_filtering_v2(y_norm, max_k, _alpha, _lambda, _tolerance, r)
    return theta, x, y_mean


# In[22]:


theta, x, y_mean = normalized(y, max_k=max_k, _tolerance=tolerance)


# In[23]:


y


# In[24]:


(np.matmul(x, theta.transpose()) + y_mean).round(decimals=2)


# # The End
