#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
df = pd.read_excel('/kaggle/input/oxford-covid19-government-response-tracker/OxCGRT_Download_latest_data.xlsx')


# In[ ]:


df.sort_values('Date',ascending=False).head(10)


# In[ ]:


sns.heatmap(df.isnull(), cbar=False)


# In[ ]:


df.columns.values


# In[ ]:


df.shape


# In[ ]:




