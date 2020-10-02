#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install lazypredict


# In[ ]:


from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# In[ ]:


data = load_breast_cancer()


# In[ ]:


X = data.data
y= data.target


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)


# In[ ]:


clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)


# In[ ]:


models,predictions = clf.fit(X_train, X_test, y_train, y_test)


# In[ ]:


models


# In[ ]:





# In[ ]:


from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np


# In[ ]:


boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)


# In[ ]:


offset = int(X.shape[0] * 0.9)


# In[ ]:


X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]


# In[ ]:


reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )


# In[ ]:


models,predictions = reg.fit(X_train, X_test, y_train, y_test)


# In[ ]:


models


# In[ ]:


predictions


# In[ ]:


# pip install jovian


# In[ ]:


# import jovian


# In[ ]:


# jovian.commit(project='lazy-predict')

