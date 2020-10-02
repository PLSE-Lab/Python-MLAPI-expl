#!/usr/bin/env python
# coding: utf-8

# # Binary variables
# 
#                  Whether to apply OneHotencoding to a column or not, depends on the number unique values (cardinality) it has. When the number of unique values are less than a certain value, then the column is one hot encoded. While encoding many columns, there is a threshold (upper limit) applied to the cardinality, below which the columns can be One-Hot Encoded. But most often the lower limit is ignored. What if a column has only two unique values? 
#  
#                  Its better to know what if the binary category columns are used in OneHotEncoding. And what should be the lower limit for One-Hot-Encoding a column.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# The Answer can be  in a single line:
# 
#         One hot encoding converts a 'n' class variable into n binary classes. Similarly it would convert one binary class variable into two binary class variable!
#     
#         We can explore by applying One-Hot-Encoder to a column of binary variables.

# Here a dataframe consisting of a column of consisting either 0 or 1 (binary) is created. 

# In[ ]:


df=pd.DataFrame(np.random.randint(0,2,100),columns=['bin'])


# In[ ]:


df.head()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False) #OneHotEncoder by default returns CSR format
df2=ohe.fit_transform(df)


#      When Onehotencoding is performed on binary variable it yields two columns. 
#    1. The first has a value 1 if the original column had a value '0' else it has a value 0. 
#    2. The second column has a value 1 if the original column had a value '1' else it has a value 0.

# In[ ]:


df2


# Second column is the replica of the original and first is the inverse of the original. 
# 1. This leads to **[multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity)**. 
# 2. This will occupy a little more space and also makes the algorithm to run longer..
# 
# So it is better to leave the binary class on One hot encoding. 

# In[ ]:


df.to_numpy().T==df2[:,1]


# In[ ]:


df.to_numpy().T==df2[:,0]


# On n-1 One Hot Encoding, this is not a problem as the first column would be dropped.

# In[ ]:


pd.get_dummies(df,columns=['bin'],drop_first='True')


#  But n-1 OHE is not advisable with n>2 when using random forest or some other algorithms which uses feature elimination. So it is always better to deal with binary variables seperately. 
#  
#  Also it is not better to use such columns as category features in catboost.
#  
#  **Suggestions are most welcomed!**
