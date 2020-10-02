#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##importing relevant libraries
import numpy as np
from sklearn.model_selection import train_test_split


# In[ ]:


##generating some data
a = np.arange(1,101)
a


# In[ ]:


b = np.arange(401,501)
b


# In[ ]:


##spliting the database
a_train,a_test,b_train,b_test = train_test_split(a,b,test_size=0.2,random_state=365)


# In[ ]:


##now exploring the dataset
a_train.shape,a_test.shape


# In[ ]:


a_train


# In[ ]:


a_test


# In[ ]:


b_train


# In[ ]:


b_test


# In[ ]:




