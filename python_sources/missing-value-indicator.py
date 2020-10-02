#!/usr/bin/env python
# coding: utf-8

# # Missing Value Indicator   
# We are going to see how to mark imputed values using a missing indicator transformer. This is very useful in order to indicate the presence or the absence of missing value in a data.  
# While using the Univariate or the Multivariate imputation, it is very important to have a trace of the places where those missing values existed earlier.  

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


from sklearn.impute import MissingIndicator


# Let's take an example of features in order to understand the concept of MissingIndicator. In this case, we are going to represent the missing values by '-1', but we could also have np.nan. All the -1 can be considered to be missing values.   

# In[ ]:


features = [[4,2,1],
            [24,12,6],
            [8,-1, 2],
            [28,14,7],
            [32,16,-1],
            [600,300,150],
            [-1,-1,1]]


# In[ ]:


'''
Instanciate the missing indicator by telling it 
that our missing values are represented by '-1'
'''
indicator = MissingIndicator(missing_values=-1)


# In[ ]:


mark_missing_values_only = indicator.fit_transform(features)
mark_missing_values_only


# We have a binary matrix where we can see that the missing values have been replaced with the value of **True**.  
# We can also determine which features have missing values by using the **indicator.features_**.  

# In[ ]:


indicator.features_


# Now we can see that that the first, second and the third features have missing values. 

# In[ ]:




