#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt

from sklearn.datasets import load_digits


# In[ ]:


digits = load_digits()


# In[ ]:


dir(digits)


# In[ ]:


df = pd.DataFrame(digits.data, digits.target)
df.head()


# In[ ]:


df['target'] = digits.target
df.head()


# In[ ]:


X = df.drop(['target'],axis='columns')
y = df.target


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


# ### Using RBF kernel

# In[ ]:


from sklearn.svm import SVC
rbf_model = SVC(C=20, kernel='rbf')


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


rbf_model.fit(X_train, y_train)
rbf_model.score(X_test, y_test)


# ### Using Linear kernel

# In[ ]:


linear_model = SVC(kernel='linear')
linear_model.fit(X_train,y_train)


# In[ ]:


linear_model.score(X_test,y_test)

