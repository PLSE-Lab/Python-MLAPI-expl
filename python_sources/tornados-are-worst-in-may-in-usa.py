#!/usr/bin/env python
# coding: utf-8

# In[45]:


import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[1]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
noaa_spc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="noaa_spc")

# print all the tables in this dataset (there's only one!)
noaa_spc.list_tables()


# In[2]:


noaa_spc.table_schema('tornado_reports')


# In[3]:


query = """SELECT state, count(timestamp) 
            FROM `bigquery-public-data.noaa_spc.tornado_reports` 
            GROUP BY state
            """
res = noaa_spc.query_to_pandas_safe(query)


# In[4]:


res


# Texas has most number of tornados.
# There were only 6 tornados in Massachusetts. 

# In[5]:


query = """SELECT state, f_scale, count(timestamp) 
            FROM `bigquery-public-data.noaa_spc.tornado_reports` 
            GROUP BY state, f_scale
            """
res2 = noaa_spc.query_to_pandas_safe(query)


# In[6]:


res2


# In[23]:


query = """SELECT state, extract(month from timestamp), count(timestamp) 
            FROM `bigquery-public-data.noaa_spc.tornado_reports` 
            GROUP BY state, extract(month from timestamp)
            """
res3 = noaa_spc.query_to_pandas_safe(query)


# In[24]:


res3copy = res3.copy()


# In[29]:


#res3 = res3copy.copy()


# In[30]:


res3 = res3.rename(columns = {'f0_':'Month','f1_':'count'})


# In[26]:


res3


# In[31]:


res3.Month = res3.Month.astype('float')


# In[32]:


res3['count'] = res3['count'].astype('float')


# In[39]:


res3 = res3.reset_index()


# In[41]:


res3.state.unique()


# In[43]:


smallres = res3[res3.state.isin(['TX','KS','OK','GA','IL'])]


# In[44]:


smallres


# In[50]:


sns.regplot(x = 'Month', y = 'count',data = smallres)


# In[51]:


cmap = {'TX': 'red', 'KS': 'blue', 'OK': 'yellow', 'GA':'green', 'IL':'orange'}

smallres.plot(x='Month', y='count', kind='scatter', 
    c=[cmap.get(c, 'black') for c in smallres.state])


# In[54]:


sm = res3.groupby('Month')['count'].sum()


# In[55]:


sm.plot()


# In[ ]:




