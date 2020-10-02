#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/house-prices-advanced-regression-techniques/housetrain.csv',index_col=0)


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.size


# In[ ]:


df.info


# In[ ]:


df1=df._get_numeric_data()


# In[ ]:


df1


# In[ ]:


df1=df._get_numeric_data().columns


# In[ ]:


nfc=list(df1)


# In[ ]:


nfc


# In[ ]:


df2=df.columns


# In[ ]:


dd=list(df2)


# In[ ]:


a=set(nfc)


# In[ ]:


b=set(dd)


# In[ ]:


b


# In[ ]:


b-a


# univarient are distplot and countplot 

# In[ ]:


sns.distplot(df['SalePrice'])


# In[ ]:


sns.distplot(df[nfc[0]],kde=False) #univariable for numeric also boxplot 


# In[ ]:


sns.countplot('HalfBath',data=df)


# In[ ]:


sns.countplot(y='FullBath',data=df)   #univarient for categorical


# In[ ]:


sns.lmplot('GrLivArea','SalePrice',data=df,fit_reg=True) #multivarient for numeric


# In[ ]:


sns.jointplot('GrLivArea','SalePrice',data=df,kind='reg')


# In[ ]:


sns.jointplot('GrLivArea','SalePrice',data=df,kind='hex')


# In[ ]:


#categorical vs categorical


# In[ ]:


crosstab=pd.crosstab(index=df["Neighborhood"],columns=df['OverallQual'])


# In[ ]:


crosstab


# In[ ]:


crosstab.plot(kind='bar',figsize=(12,8),stacked=True,colormap='Paired')


# In[ ]:


cols=['SalePrice','OverallQual','GrLivArea','GarageCars']


# In[ ]:


sns.pairplot(df[cols])


# In[ ]:


sns.heatmap(df.corr(),cmap='viridis')


# In[ ]:


sns.boxplot('Neighborhood','SalePrice',data=df) #categorical vs numeric


# In[ ]:


sns.swarmplot('Street','SalePrice',data=df) 


# In[ ]:


sns.violinplot('Street','SalePrice',data=df)

