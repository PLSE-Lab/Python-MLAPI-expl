#!/usr/bin/env python
# coding: utf-8

# # London Housing Dataset Exploration
# 
# There's some description about the daataset, but let's see if we can figure out what exactly is this data.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[ ]:


df_mo = pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv')
df_ye = pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_yearly_variables.csv')


# ## Frequency of the data
# There are two files, one corresponds to monthly data and the other corresponds to yearly data.

# In[ ]:


for df in [df_mo, df_ye]:
    print(df.shape)


# The monthly file size is approximately 12 times the size of the yearly data. So, they perhaps correspond to the same timespan.

# In[ ]:


df_mo.sample(5)


# In[ ]:


df_ye.sample(5)


# In[ ]:


def characterize(df):
    column, distinct, dist_per, type_ = [], [], [], []
    for col in df.columns:
        column.append(col) 
        distinct.append(df[col].nunique()),         
        dist_per.append(round(1e2*df[col].nunique()/len(df), 2)), 
        type_.append(df[col].dtypes)
    return pd.DataFrame({'Col': column, 'Distinct': distinct, 'Dist_per': dist_per, 'Type':type_})           .sort_values('Distinct').set_index('Col').T


# In[ ]:


characterize(df_ye)


# In[ ]:


characterize(df_mo)


# Columns that are common to both the files:

# In[ ]:


set(df_mo.columns).intersection(df_ye.columns)


# In[ ]:


df_ye.date.value_counts().sort_index()


# In[ ]:


df_ye[df_ye.date=='1999-12-01'].borough_flag.value_counts(dropna=False)


# In[ ]:


df_mo[df_mo.date.str[:4]=='2019'].date.str[5:7].value_counts().sort_index()


# In[ ]:


df_mo.date.str[:4].value_counts().sort_index().head(10)


# 45 * 12 = 540

# In[ ]:


df_mo.info()


# In[ ]:


df_ye.sample(5)


# In[ ]:


set(df_ye.area) - set(df_mo.area).intersection(set(df_ye.area))


# In[ ]:


set(df_mo.area) - set(df_mo.area).intersection(set(df_ye.area))


# In[ ]:


df_ye.sample(5)


# In[ ]:


df_ye.groupby('date')['median_salary'].mean().plot(kind='bar', figsize=(20, 5))
plt.title('Median Salary over the years')
plt.legend(['Median Salary'])
plt.xlabel('Date')
plt.show()


# In[ ]:


sns.pairplot(df_ye)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




