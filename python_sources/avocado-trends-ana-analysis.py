#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df=pd.read_csv("../input/avocado.csv")
print(df.head())
print(df.columns)


# Any results you write to the current directory are saved as output.


# In[ ]:


df.info()


# In[ ]:


df['Date']=pd.to_datetime(df['Date'])
df.info()


# In[ ]:


df['region'].value_counts().sort_index()[:10]


# In[ ]:


import seaborn as sns
#sns.scatterplot('Small Bags','Large Bags',data=df)
sns.scatterplot('AveragePrice','Total Volume',data=df)


# In[ ]:


import matplotlib.pyplot as plt
ax=sns.lineplot('Date','AveragePrice',data=df)
plt.xticks(rotation=40)


# In[ ]:


lowprices=df.sort_values('AveragePrice')
sns.boxplot(lowprices['region'],lowprices['AveragePrice'],data=df)
plt.xticks(rotation=90)
plt.figure(figsize=(100,30))


# In[ ]:


date_price=pd.DataFrame(data=df,columns=['Date','AveragePrice'])
date_price.columns=['ds','y']
print(date_price)


# In[ ]:


from fbprophet import Prophet
m=Prophet()
m.fit(date_price)


# In[ ]:


future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[ ]:


forecast.tail()


# In[ ]:


m.plot(forecast)


# In[ ]:


m.plot_components(forecast)

