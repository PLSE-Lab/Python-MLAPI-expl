#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/BreadBasket_DMS.csv')


# In[ ]:


df.head(10)


# In[ ]:


df['datetime'] = pd.to_datetime(df['Date']+" "+df['Time'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['weekday'] = df['datetime'].dt.weekday
df['hour'] = df['datetime'].dt.hour

df = df.drop(['Date'], axis=1)


# In[ ]:


df.head(5)


# In[ ]:


df.info()


# In[ ]:


df['yearmonth'] = pd.to_datetime(df['datetime']).dt.to_period('M')
trans_permonth = df[['yearmonth','Transaction']].groupby(['yearmonth'], as_index=False).count().sort_values(by='yearmonth')
trans_permonth.set_index('yearmonth' ,inplace=True)


# In[ ]:


ax = trans_permonth.plot(kind='bar',figsize=(10,5))
ax.set_xlabel("")
trans_permonth.head(10)


# In[ ]:


item_total = df[['Item','Transaction']].groupby(['Item'], as_index=False).count().sort_values(by='Transaction', ascending=False)
item_total = item_total[item_total.Item != 'NONE']
top = item_total.iloc[0:10, :]
other_item = item_total.iloc[10:, :]['Transaction'].count()
top.append(pd.DataFrame([['Other', other_item]], columns=['Item', 'Transaction']), ignore_index=True)


# In[ ]:


top.set_index('Item' ,inplace=True)


# In[ ]:


pie = top.plot(kind='pie', y='Transaction',figsize=(10,10))


# In[ ]:


trans_perhour = df[['hour','Transaction']].groupby(['hour'], as_index=False).count()
trans_perhour.head(10)
trans_perhour.set_index('hour' ,inplace=True)


# In[ ]:


az = trans_perhour.plot(kind='bar',figsize=(10,5))
az.set_xlabel("")
trans_perhour.head(10)

