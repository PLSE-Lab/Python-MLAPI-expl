#!/usr/bin/env python
# coding: utf-8

# ### Objective: 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Libraries 

# In[ ]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Importing the three datasets avilable 
df_1 = pd.read_csv('../input/telecom-churn-data-sets/customer_data.csv')
df_2 = pd.read_csv('../input/telecom-churn-data-sets/churn_data.csv')  
df_3 = pd.read_csv('../input/telecom-churn-data-sets/internet_data.csv')


# In[ ]:


df_1.head()


# In[ ]:


df_2.head()


# In[ ]:


df_3.head()


# #### We have three datasets avilable here, let us cobine all in to one single file

# In[ ]:


# Merging on 'customerID' column

temp_df = pd.merge(df_1,df_2,how='inner',on='customerID')
temp_df.head()


# In[ ]:


# Final dataframe with all predictor variables

telecom = pd.merge(temp_df,df_3,how='inner',on='customerID')
telecom.head()


# In[ ]:


# Final shape of the data frame.

telecom.shape


# In[ ]:


# let's look at the statistical aspects of the dataframe
telecom.describe()


# In[ ]:


# Let's see the type of each column
telecom.info()


# ##### Usually we see many null values but here we dont see any thing. Lets check why..

# In[ ]:


# Let us check the numerical aswell as catagorical features in the data sets

# Catagorical features
telecom.select_dtypes(include=['object']).columns


# ##### One thig I noticed here is that why the 'Total charges' are here..?

# In[ ]:


# Numerical features

telecom.select_dtypes(exclude=['object']).columns


# In[ ]:


#The varaible was imported as a string we need to convert it to float
telecom['TotalCharges'] = telecom['TotalCharges'].apply(pd.to_numeric,errors='coerce')


# In[ ]:


# Lets check the null value now
telecom.isnull().sum()


# ##### there it is 11 instances, it is only .1% of all data. there for we remove the 11 instances.

# ## Please upvote.
# 
# ##### more to come
# 
# Fahad vadakkumpadath
# 
# 2/11/2020
