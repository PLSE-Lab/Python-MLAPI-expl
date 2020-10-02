#!/usr/bin/env python
# coding: utf-8

# #### Imputer
# Scikit-lern provides a handy class to take care of missing value: Imputer. Here is how to use it.
# 
# * Reference video: 
# https://www.youtube.com/watch?v=L5MDXnuWHL0
# 
# * Blog article with Pros and Cons: 
# https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779
# This article describes imputation using the following:
# 
#     - Imputation Using (Mean/Median) Values
#     - Imputation Using (Most Frequent) or (Zero/Constant) Values
#     - Imputation Using k-NN
#         - Imputation Using Multivariate Imputation by Chained Equation (MICE)
#         - Imputation Using Deep Learning (Datawig)
#    
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/Datapreprocessing.csv")
df


# First, you need to create an Imputer Instance, specifying what stratergy 
# 
# * Mean Imputation (default)
# * Mode Imputation
# * Median Imputation

# In[ ]:


from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

# Since the median can only be computed on numerical attributes, we need to create a copu of the data without the text 
# attributes 

df_numerical = df[["Age", "Salary"]]
df_numerical


# In[ ]:


# Now you can fit the imputer instance to the training data using the fit() method
imputer.fit(df_numerical)


# In[ ]:


# The imputer has simply computed the median of each attributes and stored the result in its statistics_ instance variable.

imputer.statistics_


# In[ ]:


# Now you can use the trained imputer to transform the training set by replacing the missing values
X = imputer.transform(df_numerical)

# The result is a plain Numpy array containing the transformed features. 
X


# In[ ]:


# If you want to put the Numpy array back into Pandas DataFrame
pd.DataFrame(X,columns=["Age", "Salary"])

