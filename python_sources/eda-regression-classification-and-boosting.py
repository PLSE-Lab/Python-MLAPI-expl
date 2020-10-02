#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Daniel Cruz @giordannocruz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## PRE-PROCESSING 
# ### DATA CLEANING

# In[ ]:


df1 = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv',index_col=['Unnamed: 0'])
df2 = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')


# In[ ]:


df1.head()


# In[ ]:


df1.info()


# In[ ]:


df1.describe(include = 'all').T


# In[ ]:


for col in df1:
    print(df1[col].value_counts(), '\n')


# As we can see, not all the columns have unique values at all, the ones have those unique values are the followings:
# * city
# * rooms
# * bathrooms
# * floor
# * animal
# * furniture
# 
# And the ones we will choose to encode are the following:
# * animal
# * furniture
# 
# These ones will be used later:
# * rooms
# * bathrooms
# * parking spaces

# In[ ]:


df1.isnull().sum()


# In[ ]:


df1.isna().sum()


# In[ ]:


furniture_mapper = {'not furnished': 0, 'furnished': 1}
animal_mapper= {'not acept': 0, 'acept': 1}
df1['furniture'].replace(furniture_mapper, inplace=True)
df1['animal'].replace(animal_mapper, inplace=True)
df1['floor'] = df1['floor'].replace('-',np.nan)
df1['floor'] = pd.to_numeric(df1['floor'])


# In[ ]:


df1.isnull().sum()


# In[ ]:


df1.describe(include='all').T


# In[ ]:


df1.floor.describe()


# In[ ]:


df1['floor'].replace(99,df1['floor'].mean(), inplace=True)
df1['floor'].fillna((math.floor(df1['floor'].mean())), inplace=True)


# In[ ]:


df1.floor.describe()


# In[ ]:


df1.isnull().sum()


# In[ ]:


df1.isna().sum()


# In[ ]:


df1.describe(include = 'all').T


# In[ ]:


df1.info()


# In[ ]:


df1["hoa"] = (df1["hoa"].str.strip("$R"))
df1["rent amount"] = (df1["rent amount"].str.strip("$R"))
df1["property tax"] = (df1["property tax"].str.strip("$R"))
df1["fire insurance"] = (df1["fire insurance"].str.strip("$R"))
df1["total"] = (df1["total"].str.strip("$R"))


# In[ ]:


df1.head()


# In[ ]:


df1["hoa"] = (df1["hoa"].str.replace(',', ''))
df1["rent amount"] = (df1["rent amount"].str.replace(',', ''))
df1["property tax"] = (df1["property tax"].str.replace(',', ''))
df1["fire insurance"] = (df1["fire insurance"].str.replace(',', ''))
df1["total"] = (df1["total"].str.replace(',', ''))
df1.head()


# In[ ]:


df1.info()


# In[ ]:


for i in df1.hoa:
    print(i)


# In[ ]:


df1['hoa'] = df1['hoa'].replace('Sem info',np.nan).replace('Incluso',np.nan)


# In[ ]:


df1['hoa'] = pd.to_numeric(df1['hoa'])


# In[ ]:


df1.isnull().sum()


# In[ ]:


df1.describe(include='all').T


# In[ ]:


df1['hoa'].replace(220000,df1['hoa'].mean(), inplace=True)
df1['hoa'].fillna((math.floor(df1['hoa'].mean())), inplace=True)


# In[ ]:


df1.info()


# In[ ]:


df1['property tax'] = df1['property tax'].replace('Sem info',np.nan).replace('Incluso',np.nan)


# In[ ]:


df1['property tax'] = pd.to_numeric(df1['property tax'])
df1.isnull().sum()


# In[ ]:


df1.describe(include='all').T


# In[ ]:


df1['property tax'].replace(366300,df1['property tax'].mean(), inplace=True)
df1['property tax'].fillna((math.floor(df1['property tax'].mean())), inplace=True)
df1.info()


# In[ ]:


#df1['property tax'] = pd.to_numeric(df1['rent amount'])
df1['rent amount'] = df1['rent amount'].astype('int64')
df1['property tax'] = df1['property tax'].astype('int64')
df1['fire insurance'] = df1['fire insurance'].astype('int64')
df1['total'] = df1['total'].astype('int64')
df1['floor'] = df1['floor'].astype('int64')
df1['hoa'] = df1['hoa'].astype('int64')
df1.info()


# In[ ]:


df1.head()


# In[ ]:


df1 = df1.drop(['total'], axis = 1) 


# In[ ]:


df1.head()


# In[ ]:


df1['total'] = df1['hoa']+df1['rent amount']+df1['property tax']+df1['fire insurance']


# In[ ]:


df1.head()


# Now the total columns is fine 

# #### This is the second dataframe from the second csv

# In[ ]:


df2.head()


# In[ ]:


df2 = df2.rename(columns={'hoa (R$)': 'hoa', 'rent amount (R$)': 'rent amount',
                        'property tax (R$)':'property tax',
                        'fire insurance (R$)':'fire insurance','total (R$)':'total'})
df2.head()


# In[ ]:


df2.city.value_counts()


# In[ ]:


df2.info()


# In[ ]:


df2.isnull().sum()


# In[ ]:


df2.describe(include = 'all').T


# In[ ]:


df2['floor'] = df2['floor'].replace('-',np.nan)
df2['floor'] = pd.to_numeric(df2['floor'])
df2['parking spaces'] = df2['parking spaces'].astype('int64')
df2['furniture'].replace(furniture_mapper, inplace=True)
df2['animal'].replace(animal_mapper, inplace=True)
df2['hoa'] = df2['hoa'].replace('Sem info',np.nan).replace('Incluso',np.nan)
df2['hoa'] = pd.to_numeric(df2['hoa'])


# In[ ]:


df2.isnull().sum()


# In[ ]:


df2.floor.describe()


# In[ ]:


df2['floor'].replace(301,df2['floor'].mean(), inplace=True)
df2['floor'].fillna((math.floor(df2['floor'].mean())), inplace=True)
df2.isnull().sum()


# In[ ]:


df2['floor'] = df2['floor'].astype('int64')


# In[ ]:


df2.info()


# In[ ]:


corr_matrix  = df1.corr()
mask  = np.zeros_like(corr_matrix, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

# Correlation heatmap
fig, ax = plt.subplots(figsize = (15,9))

heatmap = sns.heatmap(corr_matrix,
                     mask = mask,
                     square = True,
                      cmap = "coolwarm",
                      cbar_kws = {"ticks": [-1,-0.5,0,0.5,1]},
                     vmin=-1,
                     vmax=1,
                     annot = True,
                     annot_kws = {"size": 10})

plt.show()


# ### This is what I have done until far, I will keep commiting my notebooks
