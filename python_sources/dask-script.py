#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import dask.dataframe as dd
import pandas as pd


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


os.popen("ls -lh ../input/train.csv").read()


# In[ ]:


df = dd.read_csv("../input/train.csv",usecols=['hotel_cluster'])
df_sample=df.sample(frac=0.1)
df_cluster=pd.DataFrame(df_sample.hotel_cluster.value_counts().head(10))


# In[ ]:


df_cluster


# In[ ]:


df_cluster=df_cluster.reset_index()
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x="index",y="hotel_cluster", data=df_cluster)


# In[ ]:


train = pd.read_csv("../input/train.csv", parse_dates=['date_time', 'srch_ci', 'srch_co'], nrows=100000)
train.info()


# In[ ]:


df_cont = dd.read_csv("../input/train.csv",usecols=['hotel_continent'])
df_continent=df_cont.sample(frac=0.1)
df_continent=pd.DataFrame(df_continent.hotel_continent.value_counts().head(10))


# In[ ]:


df_continent=df_continent.reset_index()


# In[ ]:


sns.barplot(x='index',y='hotel_continent', data=df_continent)
sns.plt.title('Continent destination')
plt.show() 


# In[ ]:




