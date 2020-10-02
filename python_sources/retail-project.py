#!/usr/bin/env python
# coding: utf-8

# Import required libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import calendar
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Load the data

# In[ ]:


df = pd.read_csv('/kaggle/input/Order sales data_01.csv',encoding = 'latin')
df_org = df.copy()                    #Keep a reference of original data
df.head()


# In[ ]:


df.info()


# # 1. Which month of the year will companyX generate high sales?

# In[ ]:


#Convert Order date to Date format
df['Order Date'] = pd.to_datetime(df['Order Date'])


# In[ ]:


#Create a column and fetch only month from Order Date
df['Order Month'] = df['Order Date'].dt.month


# In[ ]:


df['Order Month'] = df['Order Month'].apply(lambda x : calendar.month_abbr[x])


# In[ ]:


df['Order Month'].head()


# In[ ]:


#Group by Month and get sum of Sales and find out highest Sales Month
highsalesmonth = df.groupby('Order Month',as_index = False)['Sales'].sum()
highsalesmonth.sort_values('Sales',ascending = False).head(1)


# # 2. How discounts have a impact on my sales?

# In[ ]:


dfcor = df[['Discount','Sales']]
#Calculate Correlation between Discount and Sales
dfcor.corr()


# There is very negligible negative correlatiob between Discount and Sales which is pretty apparent from below graph.

# In[ ]:


ax = sns.regplot(x='Discount',y='Sales',data = df, color='g')
plt.figure(figsize = [7,7])
sns.set_style('darkgrid')


# # Does any customer segment has affinity towards any particular product/category?

# In[ ]:


#Group by Customer Segement and Product category and get the count
dfcustseg = df.groupby(['Customer Segment','Product Category'],as_index=False)['Order ID'].count()
dfcustseg


# In[ ]:


#Plot a graph to check any affinity between Customer Segment and Product Category
sns.barplot(x = 'Customer Segment', y ='Order ID', hue = 'Product Category', data = dfcustseg)
plt.figure(figsize = (20,15))
plt.show()


# Its pretty clear from graph that Office Supplies is most popular Product Category among all Customer Segments.

# # Who all are my loyal customers? Are these customers behave different from remaining customers?

# In[ ]:


dfloyal = df[['Customer Name','Order Date']]
dfloyal.head()


# In[ ]:


dfloyal.sort_values(by = ['Customer Name','Order Date'], inplace = True)


# In[ ]:


dfloyal.reset_index(inplace=True)


# In[ ]:


dfloyal.drop('index',axis =1,inplace=True)
dfloyal.head()


# In[ ]:


#To get loyal Customer first calculate average days betwen visits for all customers
dfloyal['Frequency']  = 0
rows = dfloyal.shape[0]

for i in range(rows-1):
  if dfloyal['Customer Name'][i+1]==dfloyal['Customer Name'][i]:
    dfloyal['Frequency'][i+1] = (dfloyal['Order Date'][i+1]-dfloyal['Order Date'][i])/np.timedelta64(1,'D')


# In[ ]:


dfloyal.head()


# In[ ]:


dfloyalcust = dfloyal.groupby('Customer Name',as_index=False)['Frequency'].mean()
dfloyalcust.head(10)


# In[ ]:


#Calculate average Frequency between each visit for all customer
dfloyalcust.Frequency.mean()


# In[ ]:


#Considering average frequency betwen visits for each customer as 116, write a query to update Loyal Customer if average frequency for each customer is <116 else update as Non loyal customer
dfloyalcust['Loyal Customer'] = 'NA'
dfloyalcust['Loyal Customer'] = np.where(dfloyalcust['Frequency']>116,'Yes','No')


# In[ ]:


dfloyalcust.head(10)


# In[ ]:


#Is there any correlation between profit and sales
dfprofitsales = df[['Profit', 'Sales']]
dfprofitsales.head()


# In[ ]:


dfprofitsales.corr()


# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
X = dfprofitsales['Sales']
X = X.values.reshape(-1,1)
y = dfprofitsales['Profit']
y = y.values.reshape(-1,1)


# In[ ]:


reg.fit(X,y)


# In[ ]:


SalesVal = pd.Series(5000)
SalesVal = SalesVal.values.reshape(-1,1)


# In[ ]:


reg.predict(SalesVal)

