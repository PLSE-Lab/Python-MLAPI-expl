#!/usr/bin/env python
# coding: utf-8

# # Generative Modelling for Classification
# 
# Given an observation $x_i$, we are searching for the class $j$ that belongs to.
# 
# $$P(Y=j|X=x_i)=\frac{P(Y=j)P(X=x_i|Y=j)}{P(X=x_i)}=\frac{\pi_jP_j(X=x_i)}{P(X=x_i)}$$
# Optimal prediction: the class $j$ with.png largest $\pi_j P_j(X=x_i)$.
# ![image.png](attachment:image.png)

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, multivariate_normal


# # Reading data

# In[ ]:


data = pd.read_csv('../input/iris.csv')


# In[ ]:


data.head()


# # List classes

# In[ ]:


# frequency of each class
data['class'].value_counts()


# In[ ]:


# Encoding classes
class_encoder = LabelEncoder()
data['class'] = class_encoder.fit_transform(data['class'].values)


# In[ ]:


data.head()


# In[ ]:


# list of class names
list(class_encoder.classes_)


# # Splitting data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data[['lsepal','wsepal','lpetal','wpetal']].values, data['class'].values, test_size=0.33, random_state=42)


# In[ ]:


v, freq = np.unique(y_train, return_counts = True)


# calculating
# :$$P(Y=j)=\pi_j$$

# In[ ]:


pi = freq/freq.sum()


# In[ ]:


pi


# # Abording the classification problem feature by feature
# 
# the probability that given a class $j$ the observation $x_i$ belongs to is approximated to a Gaussian
# 
# $$P(X=x_i|Y=j)=P_j(X=x_i)$$

# In[ ]:


def mean_var(X_train, y_train, idx_feature):
    mu = [X_train[np.argwhere(y_train == i)].reshape((-1,4))[:,idx_feature].mean() for i in range(len(list(class_encoder.classes_)))]
    vr = [X_train[np.argwhere(y_train == i)].reshape((-1,4))[:,idx_feature].var() for i in range(len(list(class_encoder.classes_)))]
    return mu, vr


# In[ ]:


mu, vr = mean_var(X_train, y_train, idx_feature=0)


# In[ ]:


# The mean for each class
mu


# In[ ]:


# The variance for each class
vr


# let's calculate the product of probability $\pi_jP_j(X=x_i)$ for each class and each observation of the test set and get the class that belongs to with the highest probability.

# In[ ]:


def predict_class(x, pi, mu, vr):
    prb = [pi[i] * norm.pdf(x, loc=mu[i], scale=vr[i]) for i in range(len(mu))]
    return np.argmax(prb)


# Calculate the accuracy

# In[ ]:


def test_accuracy(X_test, y_test, pi, mu, vr):
    cnt = 0
    for k, x in enumerate(X_test[:,0]):
        if predict_class(x, pi, mu, vr) == y_test[k]:
            cnt += 1
    return cnt/len(y_test)


# Calculate the accuracy for each given feature

# In[ ]:


feature_name = ['lsepal','wsepal','lpetal','wpetal']
for idx_feature in range(4):
    mu, vr = mean_var(X_train, y_train, idx_feature)
    print('The feature ',feature_name[idx_feature], ' has an accuracy of : ', test_accuracy(X_test, y_test, pi, mu, vr)*100, '%')


# # Mutlivarate Gaussian estimation
# 
# Now we use a multivariate gaussian distribution that use the fourth features to estimate the class

# In[ ]:


def multivariate_mean_var(X_train, y_train):
    mu = [X_train[np.argwhere(y_train == i)].reshape((-1,4)).mean(axis=0) for i in range(len(list(class_encoder.classes_)))]
    vr = [np.cov(X_train[np.argwhere(y_train == i)].reshape((-1,4)).T) for i in range(len(list(class_encoder.classes_)))]
    return mu, vr


# In[ ]:


def multivariate_predict_class(x, pi, mu, vr):
    prb = [pi[i] * multivariate_normal.pdf(x, mean=mu[i], cov=vr[i]) for i in range(len(mu))]
    return np.argmax(prb)


# In[ ]:


def multivariate_test_accuracy(X_test, y_test, pi, mu, vr):
    cnt = 0
    for k, x in enumerate(X_test):
        if multivariate_predict_class(x, pi, mu, vr) == y_test[k]:
            cnt += 1
    return cnt/len(y_test)


# In[ ]:


mu, vr = multivariate_mean_var(X_train, y_train)
print('The multivariate Gaussian estimation has an accuracy of : ', multivariate_test_accuracy(X_test, y_test, pi, mu, vr)*100, '%')

