#!/usr/bin/env python
# coding: utf-8

# In this Kernel we will look at the E commerce data.We will do an exploratory data analysis and get insights from the data.If you like my work please do vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Importing Python Modules**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
import pandas_profiling
import gc
import datetime
plt.style.use('ggplot')


# **Importing and Displaying Data**

# In[ ]:


df=pd.read_csv('../input/ecommerce-data/data.csv',encoding='ISO-8859-1')
df.head()


# In[ ]:


df.info()


# **Checking missing values**

# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# We have many data missing for the columns customer ID and Description

# **Check out the rows with Missing Values**

# In[ ]:


df[df.isna().any(axis=1)].head(10)
#df[df.isnull.any(axis=1)].head(10)


# **Changing Invoice Date Format**

# In[ ]:


df['InvoiceDate']=pd.to_datetime(df.InvoiceDate,format='%m/%d/%Y %H:%M')


# In[ ]:


#df.info()


# **Dropping missing values**

# In[ ]:


df=df.dropna()


# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# **Change the column type from String to Integer**

# In[ ]:


df['CustomerID']=df['CustomerID'].astype('int64')


# In[ ]:


#df.info()


# **Making copy of dataframe**

# In[ ]:


df2=df.copy()


# In[ ]:


df2.describe().round(2)


# **Quantity**

# In[ ]:


sns.set(style='whitegrid')
ax=sns.violinplot(x=df2['Quantity'])


# We can see there are negative values in Quantity.This is possible when Items are given back.We need to Drop this negative Outliers from our Dataset

# **Removing Outlies from Quantity**

# In[ ]:


df2=df2[df2.Quantity>0]
df2.describe().round(2)


# So the minimum value for us in the quantity is 1 so we have managed to remove outlier from out dataset

# **Lets Calculate the Amount Spend by a customer**

# In[ ]:


df2['AmountSpent']=df2['Quantity']*df2['UnitPrice']
df2.head()


# **Adding Month,Day and Hour Columns to our dataset**

# In[ ]:


#import datetime
df2['month_year']=pd.to_datetime(df2['InvoiceDate']).dt.to_period('M')
df2.head()


# In[ ]:


L=['year','month','day','dayofweek','dayofyear','weekofyear','quarter']
df2=df2.join(pd.concat((getattr(df2['InvoiceDate'].dt,i).rename(i) for i in L),axis=1))
df2.head()


# In[ ]:


df2.dayofweek.unique()


# We can see that the day of the week starts with 0 as Sunday 

# **Make Day of Week Starting with Monday as 1**

# In[ ]:


df2['dayofweek']=df2['dayofweek']+1


# In[ ]:


df2.head()


# **Sales by customer and Country**

# In[ ]:


sales_per_cust=df2.groupby(by=['CustomerID','Country'],as_index=False)['InvoiceNo'].count().sort_values(by='InvoiceNo',ascending=False)
sales_per_cust.columns=['CustomerID','Country','NumberofSales']
sales_per_cust.head()


# **Sales for All Customers**

# In[ ]:


orders=df.groupby(by=['CustomerID','Country'],as_index=False)['InvoiceNo'].count()
plt.subplots(figsize=(15,6))
plt.plot(orders.CustomerID,orders.InvoiceNo);
plt.xlabel('Customer ID')
plt.ylabel('Number of Orders')
plt.title('Number of Orders for different Customers')
plt.ioff()


# They long bars indicate the customers that make the highest sale by Volume

# In[ ]:


orders=df2.groupby(by=['CustomerID','Country'],as_index=False)['AmountSpent'].sum()
plt.subplots(figsize=(15,6))
plt.plot(orders.CustomerID,orders.AmountSpent);
plt.xlabel('Customer ID')
plt.ylabel('Money spent in Dollars')
plt.title('Money spend by different Customers')
plt.ioff()


# So now we have the info of he customers spending the highest amount.This customers are important for us.They drive the business revenue.

# **Lets findout the Customer ID of highest spending Customers**

# In[ ]:


spent_per_cust=df2.groupby(by=['CustomerID','Country'],as_index=False)['AmountSpent'].sum().sort_values(by='AmountSpent',ascending=False)
#spent_per_cust.columns=['CustomerID','Country','TotalSpent']
spent_per_cust.head(10)


# In[ ]:


df2.insert(loc=2,column='year_month',value=df2['InvoiceDate'].map(lambda x: 100*x.year + x.month))
df2.insert(loc=5,column='hour',value=df2.InvoiceDate.dt.hour)


# **Number of orders per month**

# In[ ]:


ax=df2.groupby('InvoiceNo')['year_month'].unique().value_counts().sort_index().plot('bar',color='blue',figsize=(15,6))
ax.set_xlabel('Month',fontsize=15)
ax.set_ylabel('Number of Orders',fontsize=15)
ax.set_title('Number of orders for per Month(1st Dec 2010 - 9th Dec 2011)',fontsize=15)
ax.set_xticklabels(('Dec_10','Jan_11','Feb_10','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11','Dec_11'));


# We can see that the Peak Sales are happening in Month of November.

# **Orders Per Week**

# In[ ]:


df2.groupby('InvoiceNo')['dayofweek'].unique().value_counts().sort_index()


# In[ ]:


ax=df2.groupby('InvoiceNo')['dayofweek'].unique().value_counts().sort_index().plot('bar',color='blue',figsize=(15,6))
ax.set_xlabel('Day',fontsize=15)
ax.set_ylabel('Number of Orders',fontsize=15)
ax.set_title('Number of orders for different Days',fontsize=15)
ax.set_xticklabels(('Mon','Tue','Wed','Thur','Fri','Sun'),rotation='horizontal',fontsize=15);


# Highest sale is taking place on thursday and Saturday seems to be Holiday

# Order by time of the Day

# In[ ]:


ax=df2.groupby('InvoiceNo')['hour'].unique().value_counts().iloc[:-1].sort_index().plot('bar',color='blue',figsize=(15,6))
ax.set_xlabel('Hour',fontsize=15)
ax.set_ylabel('Number of Orders',fontsize=15)
ax.set_title('Number of orders for different Hour',fontsize=15)
ax.set_xticklabels(range(6,21),rotation='horizontal',fontsize=15);
plt.show()


# More sale happends in the Afternoon.We can allot resources accordingly to cater to everyday needs.

# **Sales Per week of the Year**

# In[ ]:


ax=df2.groupby('InvoiceNo')['weekofyear'].unique().value_counts().iloc[:-1].sort_index().plot('bar',color='blue',figsize=(15,6))
ax.set_xlabel('Hour',fontsize=15)
ax.set_ylabel('Number of Orders',fontsize=15)
ax.set_title('Number of orders for different Hour',fontsize=15)
ax.set_xticklabels(range(0,52),rotation='horizontal',fontsize=15);
plt.show()


# Again we can see that more sale is Happening in the End of November

# **Analysis Unit Price of our items**

# In[ ]:


df2.UnitPrice.describe()


# **Distribution of Unit Price**

# In[ ]:


#plt.subplot(figsize=(12,6))
sns.boxplot(df2.UnitPrice)
plt.show()


# Most items are below 600 Dollars but there are some outliers upto the price range of 8000 Dollars

# **Which items are Free?**

# In[ ]:


df_free=df2[df2.UnitPrice==0]
print(len(df_free))
df_free.head()


# Above items are given as free of cost to the customers

# **When are items given for Free?**

# In[ ]:


df_free.year_month.value_counts().sort_index()


# **Plotting the above data******

# In[ ]:


ax=df_free.year_month.value_counts().sort_index().plot('bar',figsize=(12,6),color='blue')
ax.set_xlabel('Month',fontsize=15)
ax.set_ylabel('Frequency',fontsize=15)
ax.set_title('Frequency for different Months (Dec 2010 -Dec 2011)',fontsize=15)
ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','Aug_11','Sep_11','Oct_11','Nov_11'),rotation='horizontal',fontsize=15);
plt.show()


# So Free items are given away more in the month of November.On an average 2-3 items are given out free every month.

# **Analysis Per country**

# In[ ]:


group_country_orders=df2.groupby('Country')['InvoiceDate'].count().sort_values()
plt.subplots(figsize=(15,8))
group_country_orders.plot('barh',fontsize=12,color='blue');
plt.xlabel('Number of orders',fontsize=12)
plt.ylabel('Country',fontsize=12)
plt.title('Number of orders of different Countries',fontsize=12)
plt.ioff()


# We see that UK by far has the highest number of orders.To get better representation of other countries we need to drop the UK data and then make a plot.

# **Excluding UK Now**

# In[ ]:


group_country_orders=df2.groupby('Country')['InvoiceDate'].count().sort_values()
group_country_orders_without_uk=group_country_orders.copy()
del group_country_orders_without_uk['United Kingdom']

#plot number of unique customers in each country (without UK)
plt.subplots(figsize=(15,8))
group_country_orders_without_uk.plot('barh',fontsize=12,color='blue');
plt.xlabel('Number of orders',fontsize=12)
plt.ylabel('Country',fontsize=12)
plt.title('Number of orders of different Countries without UK',fontsize=12)
plt.ioff()


# Germany,Netherland and Ireland are other countries that have high Sales in terms of items sold.

# **Repeat Customers**

# In[ ]:


# Get our date range for our data
print('Date Range: %s to %s' % (df2['InvoiceDate'].min(),df2['InvoiceDate'].max()))

# Since Our data ends at Nov-30 2011 we're taking all the transcations that ocurred before December 01,2011
df2=df2.loc[df2['InvoiceDate']<'2011-12-01']


# In[ ]:


# Get total amount spent per invoice and associate it with CustomerID and Country
invoice_customer_df=df2.groupby(by=['InvoiceNo','InvoiceDate']).agg({'AmountSpent':sum,'CustomerID':max,'Country':max,}).reset_index()
invoice_customer_df.head()


# In[ ]:


# Sort on Amount spent,this gives us largest invoices 
invoice_customer_df.sort_values(by='AmountSpent',ascending=False).head(10)


# In[ ]:


#
#
monthly_repeat_customers_df=invoice_customer_df.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'),'CustomerID']).filter(lambda x: len(x)>1).resample('M').nunique()['CustomerID']
monthly_repeat_customers_df


# **Number of Unique Customers per month**

# In[ ]:


monthly_unique_customer_df=df2.set_index('InvoiceDate')['CustomerID'].resample('M').nunique()
monthly_unique_customer_df


# In[ ]:


monthly_repeat_customers_df=invoice_customer_df.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'),'CustomerID']).filter(lambda x:len(x)>1).resample('M').nunique()['CustomerID']
monthly_repeat_customers_df


# In[ ]:


monthly_repeat_percentage=monthly_repeat_customers_df/monthly_repeat_customers_df*100.0
monthly_repeat_percentage


# In[ ]:


#Plotting this visully
#Note were using a 2 scale y axis (left and right)

ax=pd.DataFrame(monthly_repeat_customers_df.values).plot(figsize=(12,8))
pd.DataFrame(monthly_unique_customer_df.values).plot(ax=ax,grid=True)

ax2=pd.DataFrame(monthly_repeat_percentage.values).plot.bar(ax=ax,grid=True,secondary_y=True,color='blue',alpha=0.3)

ax.set_xlabel('Date')
ax.set_ylabel('Number of Customers')
ax.set_title('Number of Unique vs. Repeat Customers Over Time')

ax2.set_ylabel('percentage (%)')

ax.legend(['Repeat Customers','All Customers'])
ax2.legend(['Percentage of Repeat'],loc='upper right')

ax.set_ylim([0,monthly_unique_customer_df.values.max()+100])
ax2.set_ylim([0,100])

plt.xticks(range(len(monthly_repeat_customers_df.index)),[x.strftime('%m.%Y') for x in monthly_repeat_customers_df.index],rotation=45)
plt.show()


# **Relation between Repeat customer and Revenue**

# In[ ]:


monthly_revenue_df=df2.set_index('InvoiceDate')['AmountSpent'].resample('M').sum()
monthly_rev_repeat_customer_df=invoice_customer_df.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'),'CustomerID']).filter(lambda x:len(x) > 1).resample('M').sum()['AmountSpent']

monthly_rev_per_repeat_customers_df=monthly_rev_repeat_customer_df/monthly_revenue_df*100
monthly_rev_per_repeat_customers_df


# In[ ]:


#Plotting this visully
#Note were using a 2 scale y axis (left and right)

ax=pd.DataFrame(monthly_revenue_df.values).plot(figsize=(12,8))
pd.DataFrame(monthly_rev_per_repeat_customers_df.values).plot(ax=ax,grid=True)

#ax2=pd.DataFrame(monthly_repeat_percentage.values).plot.bar(ax=ax,grid=True,secondary_y=True,color='blue',alpha=0.3)

ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.set_title('Total Revenue vs. Revenue from Repeat Customer')

ax.legend(['Total Revenue','Repeat Customer Revenue'])
ax.set_ylim([0,max(monthly_revenue_df.values)+100000])
ax2=ax.twinx()

pd.DataFrame(monthly_rev_per_repeat_customers_df.values).plot(ax=ax2,kind='bar',color='blue',alpha=0.3)
ax2.set_ylim([0,max(monthly_rev_per_repeat_customers_df.values)+30])
ax2.set_ylabel('Percentage (%)')
ax2.legend(['Repeat Revenue Percentage'])
ax2.set_xticklabels([x.strftime('%m.%Y') for x in monthly_rev_per_repeat_customers_df.index]);


# In[ ]:




