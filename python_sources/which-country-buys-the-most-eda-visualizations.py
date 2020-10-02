#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# This is a transactional data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
# 
# The goal of the analysis is to explore the dataset ans visualize company sales.

# In[ ]:


#Import Python3 libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# In[ ]:


#Import data file
df = pd.read_csv('../input/data.csv', encoding = 'ISO-8859-1')
print('Dataframe dimensions:', df.shape)
df.head()
df.describe().round(2)


# Two points to note here is that the minimum quantity & unit price is negative. This could be due to cancelled orders or price adjustments.
# We'll delete these in a section below.

# ### Data Cleaning
# 
# Here we will do the following steps:
# 1. analyse and correct the datatypes of each column in the dataset
# 2. check for missing values and delete
# 3. check for duplicates and delete

# In[ ]:


print(df.dtypes)

#Change the format for the invoice date
df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")
df.head()


# In[ ]:


#Drop missing values from the dataset
df.dropna(inplace=True)

#Drop duplicates from the dataset
df.drop_duplicates(inplace=True)


# In[ ]:


# Change customer id type from float to string 
df['CustomerID'] = df['CustomerID'].astype('int')

print(df.dtypes)
print('Dataframe dimensions:', df.shape)


# ### Data Analysis

# In[ ]:


temp = df[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop = False)
countries = temp['Country'].value_counts()


# In[ ]:


pd.DataFrame([{'products': len(df['StockCode'].value_counts()),    
               'transactions': len(df['InvoiceNo'].value_counts()),
               'customers': len(df['CustomerID'].value_counts()),  
              }], columns = ['products', 'transactions', 'customers'], index = ['Counts'])


# ### Canceled Orders
# 
# Let's count the number of canceled orders

# In[ ]:


temp = df.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
temp[ : 5]
temp['canceled_orders'] = temp['InvoiceNo'].apply(lambda x: int('C' in x))
temp = temp.rename(columns = {'InvoiceDate':'No of products'})
temp[ : 5]

num_of_canceled_orders = temp['canceled_orders'].sum()
total_orders = temp.shape[0]
print('Percent of canceled orders: {:.2f}% '.format(num_of_canceled_orders/total_orders * 100))


# In[ ]:


#Remove line items with negative quantity and unit price
df_new = df[(df.Quantity >= 0) & (df.UnitPrice >= 0)] 
df_new.describe().round(2)


# In[ ]:


df_new.head()


# In[ ]:


#Get the total spent for each purchase - Basket price
df_new['TotalDollars'] = df_new['Quantity'] * df_new['UnitPrice']
df_new.head()


# > ### Orders and Average Spend by Country

# In[ ]:


orders_by_country = df_new.groupby('Country')['InvoiceNo'].count().sort_values(ascending=False)

#Plot
orders_by_country.plot('bar')
plt.xlabel('Number of Orders')
plt.ylabel('Country')
plt.title('Number of Orders per Country', fontsize=16)
plt.show()


# In[ ]:


spend_by_country = df_new.groupby('Country')['TotalDollars'].mean().sort_values(ascending=False)
#Plot
spend_by_country.plot('bar')
plt.xlabel('Average spend amount in $')
plt.ylabel('Country')
plt.title('Average spend amount per Country', fontsize=16)
plt.show()


# Even though most of the orders come from UK, the mean spend is relatively low.
# 
# As UK has the highest number of orders, we cannot see the exact picture of the other countries. Let's remove UK from our analysis

# In[ ]:


orders_by_country = df_new.groupby('Country')['InvoiceNo'].count().sort_values(ascending=False)
del orders_by_country['United Kingdom']

#Plot
orders_by_country.plot('bar')
plt.xlabel('Number of Orders')
plt.ylabel('Country')
plt.title('Number of Orders per Country', fontsize=16)
plt.show()

