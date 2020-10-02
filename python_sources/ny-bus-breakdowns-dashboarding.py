#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#import Dataset
dataset = pd.read_csv("../input/bus-breakdown-and-delays.csv")


# In[ ]:


#Get some info about the dataset and the structure
dataset.info()


# In[ ]:


dataset.head()


# In[ ]:


dataset['School_Year'].unique()


# In[ ]:


dataset[dataset['School_Year']=="2019-2020"]


# In[ ]:


#Delete the wrong data because one data occured in 2020 (Now it is 2018)
import datetime
now = datetime.datetime.now().strftime("%Y-%m-%d")

for i in range(0,len(dataset['Occurred_On'])):
    if dataset.iloc[i]['Occurred_On'] > now:
        df_new = dataset.drop(dataset.index[i])


# In[ ]:


df_new.info()


# In[ ]:


plt.figure(figsize=(15,6))
p = sns.countplot(x='Reason', data=df_new)
p.set_title('Breakdown and delay reasons')
for item in p.get_xticklabels():
    item.set_rotation(30)


# In[ ]:


sorted_year = sorted(df_new['School_Year'].unique())
p = sns.countplot(x='School_Year', data=df_new, order=sorted_year)
p.set_title('Breakdown and delay in different years')


# In[ ]:


plt.figure(figsize=(15,6))
p = sns.countplot(x='Reason', data=df_new, hue='School_Year')
p.set_title('Reasons in different years')
for item in p.get_xticklabels():
    item.set_rotation(30)


# In[ ]:




