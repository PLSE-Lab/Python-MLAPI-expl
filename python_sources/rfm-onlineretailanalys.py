#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
pd.set_option('display.max_columns', None);
pd.set_option('display.max_rows', None);
pd.set_option('display.float_format', lambda x:'%.0f' % x)
import matplotlib.pyplot as plt
df = pd.read_csv('../input/online-retail-ii-uci/online_retail_II.csv')


# In[ ]:


df.head()


# In[ ]:


df['Description'].nunique()


# In[ ]:


df['Description'].value_counts().head()


# In[ ]:


df.groupby('Description').agg({'Quantity':'sum'}).head()


# In[ ]:


df.groupby('Description').agg({'Quantity':'sum'}).sort_values('Quantity', ascending = False).head()


# In[ ]:


# TOTAL INVOICE AMOUNT
df['Invoice'].nunique()


# In[ ]:


df['Total_Price'] = df['Quantity']*df['Price']


# In[ ]:


df.head(2)


# In[ ]:


df['InvoiceDate'].max()


# In[ ]:





# In[ ]:


df.groupby('Invoice').agg({'TotalPrice':'sum'}).head()


# In[ ]:


df.sort_values('Price', ascending = False).head()


# In[ ]:


df['Country'].value_counts()


# In[ ]:


df.groupby('Country').agg({'TotalPrice':'sum'}).head()


# In[ ]:


df.groupby('Country').agg({'TotalPrice':'sum'}).sort_values('TotalPrice', ascending = False).head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna(inplace = True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T


# In[ ]:


for feature in ['Quantity','Price','TotalPrice']:
    Q1 = df[feature].quantile(0.01)
    Q3 = df[feature].quantile(0.99)
    IQR = Q3-Q1 
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR
    
    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis = None):
        print(feature, 'yes')
        print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])
    else:
        print(feature, 'No')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df['InvoiceDate'].min()


# In[ ]:


df['InvoiceDate'].max()


# In[ ]:


import datetime as dt
Today_time= dt.datetime(2011,12,9)
Today_time


# In[ ]:


df.info()


# In[ ]:


df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()


# In[ ]:


df["Customer ID"] = df["Customer ID"].astype(int)
df["Customer ID"].head()


# In[ ]:


(Today_time-df.groupby("Customer ID").agg({"InvoiceDate":"max"})).head()

