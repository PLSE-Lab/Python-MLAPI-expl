#!/usr/bin/env python
# coding: utf-8

# In[131]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
data = pd.read_csv('../input/pokemon.csv')


# In[132]:


data.head(6)


# In[133]:


data.tail(6)


# In[134]:


data.columns


# In[135]:


data.shape


# In[ ]:





# <a id="18"></a> <br>
# ### EXPLORATORY DATA ANALYSIS
# value_counts(): Frequency counts
# <br>outliers: the value that is considerably higher or lower from rest of the data
# * Lets say value at 75% is Q3 and value at 25% is Q1. 
# * Outlier are smaller than Q1 - 1.5(Q3-Q1) and bigger than Q3 + 1.5(Q3-Q1). (Q3-Q1) = IQR
# <br>We will use describe() method. Describe method includes:
# * count: number of entries
# * mean: average of entries
# * std: standart deviation
# * min: minimum entry
# * 25%: first quantile
# * 50%: median or second quantile
# * 75%: third quantile
# * max: maximum entry
# 
# <br> What is quantile?
# 
# * 1,4,5,6,8,9,11,12,13,14,15,16,17
# * The median is the number that is in **middle** of the sequence. In this case it would be 11.
# 
# * The lower quartile is the median in between the smallest number and the median i.e. in between 1 and 11, which is 6.
# * The upper quartile, you find the median between the median and the largest number i.e. between 11 and 17, which will be 14 according to the question above.

# In[136]:


data.info()


# In[137]:


data.describe()


# In[138]:


#############


# In[139]:


print(data['Type 2'].value_counts(dropna=False))


# In[140]:


data.boxplot(column='Attack')


# In[141]:


data.boxplot(column='Attack',by = 'Legendary')


# In[142]:


data_new=data.head(100).loc[20:25,["Type 1","Type 2","HP","Attack"]]
data_new


# In[143]:


melting_data=pd.melt(frame=data_new,id_vars='Type 1',value_vars=['Type 2','HP','Attack'])
melting_data


# In[144]:


data_new = data.head()    # I only take 5 rows into new data
data_new
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted


# In[145]:


melted.pivot(index = 'Name', columns = 'variable',values='value')


# In[146]:


data1 = data['Type 1'].head()
data2= data['Type 1'].tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row


# In[147]:


data.dtypes


# In[148]:


data['Type 1']=data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')


# In[149]:


data.dtypes


# In[150]:


data.info() #Type 2 414 not equal to 800


# In[151]:


data['Type 2'].value_counts(dropna=False)


# In[152]:


data1=data.copy()


# In[153]:


data1["Type 2"].dropna(inplace = True)


# In[154]:


assert data1['Type 2'].notnull().all()


# In[155]:


data['Type 2'].fillna('empty',inplace=True)


# In[156]:





# In[157]:


data1.info()
data1['Type 2'].value_counts(dropna=False)


# In[158]:


assert  data1['Type 2'].notnull().all()


# In[ ]:




