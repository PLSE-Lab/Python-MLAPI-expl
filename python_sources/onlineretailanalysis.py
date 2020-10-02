#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/onlineretail/OnlineRetail.csv', encoding = "ISO-8859-1")


# In[ ]:


df.head()


# In[ ]:


df=df[['CustomerID', 'InvoiceNo', 'StockCode','Quantity', 'UnitPrice','Description', 'InvoiceDate','Country']]


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# **1. What kind of business is this company doing ? **
# => Online retail in 38 countries 

# In[ ]:


df['Country'].unique()


# In[ ]:


TotalAmount = df['Quantity'] * df['UnitPrice']

df.insert(loc=5, column = 'TotalAmount', value=TotalAmount) 


# In[ ]:


df.head()


# In[ ]:


new_df = df[['CustomerID', 'InvoiceNo', 'StockCode', 'Quantity','TotalAmount', 'InvoiceDate', 'Country']]
new_df2 = df.copy()


# In[ ]:


new_df.head()


# In[ ]:


new_df2.head()


# In[ ]:


country_price = new_df.groupby('Country')['Quantity'].sum().sort_values(ascending = False)
country_price


# In[ ]:


country_totalAmount = new_df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False)
country_totalAmount


# In[ ]:


country_price[:5].plot(kind='bar')


# In[ ]:


country_price[33:].plot(kind='bar')


# In[ ]:


new_df['Year'] = pd.DatetimeIndex(new_df['InvoiceDate']).year
new_df.head()


# In[ ]:


new_df.groupby('Year')['TotalAmount'].sum().plot(kind='bar')


# In[ ]:


new_df.head()


# In[ ]:


new_df['Mon'] = pd.DatetimeIndex(new_df['InvoiceDate']).month
new_df['month'] = pd.DatetimeIndex(new_df['InvoiceDate']).month_name
new_df.groupby(['Mon', 'Year'])['TotalAmount'].sum().plot(kind='bar', title = 'Sales Moth Wise') 



# In[ ]:


new_df.head()


# In[ ]:


new_df=new_df.dropna()


# In[ ]:


new_df.isnull().sum()


# In[ ]:


cus_id =  pd.DataFrame(new_df.groupby('Country')['CustomerID'].count().sort_values(ascending = False))
cus_id[:5].plot(kind='bar', title='Country with most customers')


# In[ ]:


cus_id[-5:].plot(kind='bar', title='Least customers for country')


# In[ ]:


new_df2 = new_df2.dropna() 
new_df2.isnull().sum()
new_df2.head()


# In[ ]:


avg_sales = new_df2.groupby(['StockCode','Description'])['Quantity', 'TotalAmount'].mean().sort_values(by='Quantity', ascending = False)
avg_sales

