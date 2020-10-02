#!/usr/bin/env python
# coding: utf-8

# #### Import all the required libraries.

# In[ ]:


import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats as stats


# #### Import the dataset

# In[ ]:


df = pd.read_csv('/kaggle/input/apple-data-analysis/AppleStore.csv')
pd.set_option("display.max_columns",None)
df.head()


# In[ ]:


cat_var = df.columns[df.dtypes == 'object']
cat_var


# In[ ]:


num_var = df.columns[df.dtypes != 'object']
num_var = list(num_var)
num_var


# In[ ]:


print('Number of unique values in each column:')
for col in df.columns[0:]:
    print(col,':')
    print('nunique =', df[col].nunique())
    print('unique =', df[col].unique())
    print()


# In[ ]:


df.head()


# ### Custormer - CLV

# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot = True)


# In[ ]:


df.dtypes


# In[ ]:





# In[ ]:


# Change data types

df['Effective To Date'] = pd.to_datetime(df['Effective To Date'])


# In[ ]:


# get dummies for categorical nominal columns

# One Hot Encoding

# encoded_columns = pd.get_dummies(data['column'])
# data = data.join(encoded_columns).drop('column', axis=1)


# making dataframe using get_dummies() 
dummies_state = df["State"].str.get_dummies()


# In[ ]:


dummies_state.head()


# In[ ]:


df['State'].nunique()


# In[ ]:


df['Response'].head()


# In[ ]:


df.columns


# In[ ]:



df['Response'].replace(['No','Yes'],[0,1], inplace = True)
df['Coverage'].replace(['Basic','Extended','Premium'],[0,1,2], inplace = True)
df['Education'].replace(['Bachelor','College','Master','High school or Below','Doctor'], [0,1,2,3,4], inplace = True)
df['EmploymentStatus'].replace(['Unemployed', 'Employed', 'Medical Leave', 'Disabled', 'Retired'],[0,1,2,3,4], inplace = True)
df['Response'].replace(['F','M'],[0,1], inplace = True)
df['Location Code'].replace(['Rural','Suburban','Urban'],[0,1,2], inplace = True)
df['Marital Status'].replace(['Divorced','Single','Married'], inplace = True)
df['Policy Type'].replace(['Corporate Auto', 'Personal Auto', 'Special Auto'],[0,1,2], inplace = True)
df['Policy'].replace(['Corporate L1','Corporate L2','Corporate L3','Personal L1','Personal L2','Personal L3'],
                    [0,1,2,3,4,5,6,7,8], inplace = True)

df['Renew Offer Type'].replace(['Offer1', 'Offer2', 'Offer3', 'Offer4'],[0,1,2,3], inplace = True)
df['Sales Channel'].replace([ ])


# In[ ]:


df['Marital Status'].dtype


# In[ ]:


df['Renew Offer Type'].unique()

