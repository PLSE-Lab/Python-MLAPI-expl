#!/usr/bin/env python
# coding: utf-8

# In[177]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Data Visualization 
import seaborn as sns # Data Visualization
from scipy.stats import norm # Normalizer
from sklearn.preprocessing import StandardScaler # Standard Scaling
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[178]:


# Imported the datasets
train = pd.read_csv('../input/Train.csv')
test = pd.read_csv('../input/Test.csv')


# In[179]:


#Inspection of columns
train.columns


# **Univariate Analysis of column Item_Outlet_Sales**

# In[180]:


train['Item_Outlet_Sales'].describe()


# In[181]:


#Histogram
sns.distplot(train['Item_Outlet_Sales'])


# * Deviate from the normal distribution.
# * Have appreciable positive skewness.
# * Show peakedness.

# In[182]:


print('Skew : {0:.2f}'.format(train['Item_Outlet_Sales'].skew()))
print('Kurtosis : {0:.2f}'.format(train['Item_Outlet_Sales'].kurt()))


# **Bi-Variate Analysis for Item_Outlet_Sales vs Categorical Features**

# *Item_Outlet_Sales Vs Item_Fat_Content*

# In[183]:


#But first Let's check the categories present in column Item_Fat_Content
train['Item_Fat_Content'].value_counts()


# In[184]:


train.loc[train['Item_Fat_Content']== 'LF','Item_Fat_Content'] = 'Low Fat'
train.loc[train['Item_Fat_Content']== 'low fat','Item_Fat_Content'] = 'Low Fat'
train.loc[train['Item_Fat_Content']== 'reg','Item_Fat_Content'] = 'Regular'


# In[185]:


train['Item_Fat_Content'].value_counts()


# In[186]:


sns.boxplot(train['Item_Outlet_Sales'],train['Item_Fat_Content'])


# *Item_Outlet_Sales Vs Item_Type*

# In[187]:


train['Item_Type'].value_counts()


# In[188]:


sns.boxplot(train['Item_Outlet_Sales'],train['Item_Type'])


# In[189]:


train.pivot_table('Item_Outlet_Sales','Item_Type',aggfunc=np.average)


# In[190]:


#Item_Outlet_Sales vs Outlet_Identifier
sns.boxplot(train['Item_Outlet_Sales'],train['Outlet_Identifier'])


# In[191]:


train.pivot_table('Item_Outlet_Sales','Outlet_Identifier',aggfunc=np.average)


# *So outlet OUT027 is performing great in terms of Outlet_Sales and OUT010 and OUT019 are performing poorly as compared to others which are around the same range*

# In[192]:


train.pivot_table('Item_Outlet_Sales','Outlet_Identifier',aggfunc=np.average).plot.bar()


# In[193]:


train.pivot_table('Item_Outlet_Sales','Outlet_Establishment_Year',aggfunc=np.average)


# In[194]:


sns.boxplot(train['Item_Outlet_Sales'],train['Outlet_Size'])


# In[195]:


train.pivot_table('Item_Outlet_Sales','Outlet_Size',aggfunc=np.average)


# In[196]:


train.pivot_table('Item_Outlet_Sales','Outlet_Size',aggfunc=np.average).plot.bar(color='red')


# ***'Item_Outlet_Sales' Vs 'Outlet_Location_Type'***

# In[197]:


sns.boxplot(train['Item_Outlet_Sales'],train['Outlet_Location_Type'])


# In[198]:


train.pivot_table('Item_Outlet_Sales','Outlet_Location_Type',aggfunc=np.average)


# In[199]:


train.pivot_table('Item_Outlet_Sales','Outlet_Location_Type',aggfunc=np.average).plot.bar()


# ***'Item_Outlet_Sales' Vs 'Outlet_Type'***

# In[200]:


sns.boxplot(train['Item_Outlet_Sales'],train['Outlet_Type'])


# In[201]:


train.pivot_table('Item_Outlet_Sales','Outlet_Type',aggfunc=np.average)


# In[202]:


train.pivot_table('Item_Outlet_Sales','Outlet_Type',aggfunc=np.average).plot.bar()


# In[203]:


plt.scatter(train['Item_Weight'],train['Item_Outlet_Sales'])


# In[204]:


plt.scatter(train['Item_Visibility'],train['Item_Outlet_Sales'])
plt.xlabel('Item Visib')


# In[205]:


plt.scatter(train['Item_Visibility'],train['Item_Outlet_Sales'])

