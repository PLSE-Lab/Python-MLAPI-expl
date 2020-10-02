#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Reading data file from directory

# In[ ]:


os.chdir("../input")
os.listdir()


# In[ ]:


data = pd.read_csv("vgsales.csv")


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.isnull().values.any()


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:



dt = data[['NA_Sales',
       'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']].sum().reset_index()
dt.columns = ['Area','Sales_tot']


# In[ ]:


dt


# In[ ]:


ax = sns.barplot(x="Area", y="Sales_tot", data=dt)
ax.set_title("Bar plot total sales vs area")


# In[ ]:


plt.pie(dt['Sales_tot'],labels=dt['Area'])


# In[ ]:


dtg = data.groupby(['Genre'])
dtg_t = dtg['NA_Sales',
       'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'].aggregate(np.sum)


# In[ ]:


dtg_t


# In[ ]:


dtg_t.plot()


# In[ ]:


dtm = data.groupby(['Year'])
dtm_t = dtm['NA_Sales',
       'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'].aggregate(np.mean)
dtm_t


# In[ ]:


dtm_t.plot()


# In[ ]:




