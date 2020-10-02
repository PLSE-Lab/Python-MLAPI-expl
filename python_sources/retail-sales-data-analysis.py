#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.style as style
from datetime import timedelta
import datetime as dt
import time
import os


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


customer = pd.read_csv("/kaggle/input/retail-case-study-data/Customer.csv")
prod_cat= pd.read_csv("/kaggle/input/retail-case-study-data/prod_cat_info.csv")
transactions = pd.read_csv("/kaggle/input/retail-case-study-data/Transactions.csv")


# **DATA PREPROCESSING**

# In[ ]:


customer.isnull().sum()
"""Both Gender and city_code columns have null values"""

#To fix this, I applied ffill (fill forward) to the null cells
customers = customer.fillna({
    'Gender': customer['Gender'].ffill(),
    'city_code': customer['city_code'].ffill()
})

#Splitting transaction date into year, month and day of week


# In[ ]:


transactions['tran_date'] = pd.to_datetime(transactions['tran_date'], errors='coerce')


# In[ ]:


transactions.insert(loc=3, column='year', value= transactions.tran_date.dt.year)
transactions.insert(loc=4, column='month', value= transactions.tran_date.dt.month)
transactions.insert(loc=5, column='day', value=(transactions.tran_date.dt.weekday_name))

transactions.head()


# In[ ]:


###### Transactions and Product Category datasets have no null cells.

df = pd.merge(left = customers, right = transactions, left_on = 'customer_Id', right_on = 'cust_id').drop('cust_id', axis =1)
#This joins the customers and transactions dataset on customer_Id and cust_id. The duplicate column (cust_id)
#is dropped.
df.duplicated().sum()
#There are 13 duplicate cells in the df dataframe. Next step: drop duplicates.

df.drop_duplicates(inplace = True)
df_new = pd.merge(df, prod_cat, left_on = ('prod_subcat_code', "prod_cat_code"), right_on = ('prod_sub_cat_code', "prod_cat_code")).drop('prod_sub_cat_code', axis =1)
df_new.shape
#Columns from the prod_cat dataset have been added to the df dataframe

df_new.describe()
#showing basic statistical details


customer_city=df_new[['city_code','customer_Id']]
customer_city.groupby(['city_code'])['customer_Id'].aggregate('count').reset_index().sort_values('customer_Id', ascending=False)


# In[ ]:


rdf = df_new[df_new['Qty'] <= -1]
sdf = df_new[df_new['Qty'] >= 0]
"""To analyze negative and positive order values seperately, 2 new datasets were created."""


# #### Returned Orders Dataset

# **1. Returned Orders Per Store Category**

# In[ ]:



"""The e-shop accounted for the most returns in the returned orders dataset."""
orders = rdf.groupby(by=['Store_type'], as_index = False)['Qty'].count()
plt.figure(figsize=(6,4))
sns.set_style('whitegrid')
sns.barplot(x = "Store_type", y = 'Qty', data = orders,  palette= "magma")
plt.xlabel('Store Category')
plt.ylabel('Returned Orders')
plt.title('Total number of returned orders per store category')
plt.show()


# **2. Returned Orders Per Product Category**

# In[ ]:


"""Books, Electronics and Home & Kitchen were the most returned product categories."""
category = rdf.groupby(by=['prod_cat'], as_index = False)['Qty'].count()
plt.figure(figsize=(8,4))
sns.set_style('whitegrid')
sns.barplot(x = "prod_cat", y = 'Qty', data = category,  palette= "inferno")
plt.xlabel('Product Category')
plt.ylabel('Returned Orders')
plt.title('Total number of returned orders per product category')
plt.show()


# **4. Returned Orders By City**

# In[ ]:


"""Returns across cities were quite similar, with the highest recorded for City Code 8."""
city = rdf.groupby(by= ['city_code'], as_index = False)['Qty'].count()
plt.figure(figsize=(8,4))
sns.set_style('whitegrid')
sns.barplot(x = "city_code", y = 'Qty', data = city,  palette= "viridis")
plt.xlabel('City Code')
plt.ylabel('Returned Orders')
plt.title('Total number of returned orders per city')
plt.show()


# **5. Returned Orders By Year**

# In[ ]:


"""The highest product returns were recorded in 2012."""
order_year = rdf.groupby(by=['year'], as_index = False)['Qty'].count()
plt.figure(figsize=(6,5))
sns.barplot(x = "year", y = 'Qty', data = order_year,  palette= "plasma")
plt.xlabel('Year')
plt.ylabel('Returned Orders')
plt.title('Returned Orders Per Year')
plt.show()

"""Chart below shows revenue loss."""
sales = rdf.groupby(by=['year'], as_index = False)['total_amt'].sum()
plt.figure(figsize=(6,5))
sns.barplot(x = "year", y = 'total_amt', data = sales,  palette = 'plasma')
plt.xlabel('Year')
plt.ylabel('Returned Orders')
plt.title('Revenue Lost Due To Returns')
plt.show()


# **Successful Orders Dataset**

# **1. Total Orders Per Store Category**

# In[ ]:


"""The most purchases were made through the e-shop."""
orders1 = sdf.groupby(by=['Store_type'], as_index = False)['Qty'].count()
plt.figure(figsize=(6,4))
sns.set_style('whitegrid')
sns.barplot(x = "Store_type", y = 'Qty', data = orders1,  palette= "inferno")
plt.xlabel('Store Category')
plt.ylabel('Total Orders')
plt.title('Total successful orders per store category')
plt.show()


# **2. Total Orders Per Product Category**

# In[ ]:


"""Books, Electronics and Home & Kitchen were the most purchased product categories."""
category1 = sdf.groupby(by=['prod_cat'], as_index = False)['Qty'].count()
plt.figure(figsize=(8,4))
sns.set_style('whitegrid')
sns.barplot(x = "prod_cat", y = 'Qty', data = category1,  palette= "plasma")
plt.xlabel('Product Category')
plt.ylabel('Total Orders')
plt.title('Total successful orders per product category')
plt.show()


# **3. Total Successful Orders Per City**

# In[ ]:


location1 = sdf.groupby(by= ['city_code'], as_index = False)['Qty'].count()
plt.figure(figsize=(8,4))
sns.set_style('whitegrid')
sns.barplot(x = "city_code", y = 'Qty', data = location1,  palette= "viridis")
plt.xlabel('City Code')
plt.ylabel('Successful Orders')
plt.title('Orders placed per location')
plt.show()


# **5. Total Orders Per Year**

# In[ ]:


"""Highest sales occured in the years 2012 and 2013."""
order_year1 = sdf.groupby(by=['year'], as_index = False)['Qty'].count()
plt.figure(figsize=(6,5))
sns.barplot(x = "year", y = 'Qty', data = order_year1,  palette= "plasma")
plt.xlabel('Year')
plt.ylabel('Total Orders')
plt.title('Order Quantity Per Year')
plt.show()

"""The most successful sales occured in 2012 and 2013"""
sales1 = sdf.groupby(by=['year'], as_index = False)['total_amt'].sum()
plt.figure(figsize=(6,5))
sns.barplot(x = "year", y = 'total_amt', data = sales1,  palette= "plasma")
plt.xlabel('Year')
plt.ylabel('Total Orders')
plt.title('Revenue Generated')
plt.show()


# **6. Revenue Per Category**

# In[ ]:


"""Products in the Women, Mens and Kids categories sold better than other categories."""
subcategory = sdf.groupby(by=['prod_subcat'], as_index = False)['total_amt'].sum()
plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
sns.barplot(x = "total_amt", y = 'prod_subcat', data = subcategory, palette= "inferno")
plt.xlabel('Amount Spent')
plt.ylabel('Product Subcategories')
plt.title('Amount Spent Per Subcategories')
plt.show()


# **7. Purchase By Gender**

# In[ ]:


"""Purchases by men accounted for the highest percentage across all product categories except footwear and bags."""
plt.figure(figsize=(8,4))
sns.set_style('whitegrid')
sns.countplot(x = 'prod_cat', hue = "Gender", data = sdf, palette= "inferno")
plt.xlabel('Amount Spent')
plt.ylabel('Product Categories')
plt.title('Purchase By Gender')
plt.show()

"""Pivot chart representation"""
product_by_gender = sdf.groupby(["Gender","prod_cat"])[["Qty"]].sum().reset_index()
product_by_gender.pivot(index="Gender",columns="prod_cat",values="Qty")


# In[ ]:


"""To analyze data based on customer age, a new age column is created."""
now = pd.Timestamp('now')
sdf['DOB'] = pd.to_datetime(sdf['DOB'], errors = 'coerce')    #1
sdf['DOB'] = sdf['DOB'].where(sdf['DOB'] < now, sdf['DOB'] -  np.timedelta64(100, 'Y'))   # 2
sdf['AGE'] = (now - sdf['DOB']).astype('<m8[Y]').round()

#check for max and min age
sdf['AGE'].max()
sdf['AGE'].min()

"""The cut() method is used to bin age values into discrete intervals."""
sdf['age_category'] = pd.cut(x = sdf['AGE'], bins = [24, 30, 39, 49], labels=['24-29','30-39','40-50'],include_lowest=True)


# In[ ]:


sdf['age_category'].value_counts()


# **7. Purchase By Age Category**

# In[ ]:


"""Customers aged betweeen 40-50 purchased the most products and 24-30 customers purchased the least"""
plt.figure(figsize=(8,6))
sns.countplot(x = 'prod_cat', hue = 'age_category', data = sdf, palette= "inferno")

"""Pivot chart representation"""
spend_per_category = sdf.groupby(['age_category','prod_cat'])['total_amt'].sum().reset_index()
spend_per_category.pivot(index = "age_category", columns = "prod_cat", values = 'total_amt').round(0)


# **7. Quantity Sold By Month**
# 

# In[ ]:


"""Showing category sales by month."""
plt.figure(figsize=(12,6))
sns.countplot(x = 'prod_cat', hue = 'month', data = sdf, palette= "plasma")

sale_by_month = sdf.groupby(['month','prod_cat'])['Qty'].count().reset_index()
sale_by_month.pivot(index = "month", columns = "prod_cat", values = 'Qty').round(0)

sdf['month'].value_counts()


# **RFM Score Calculation**

# In[ ]:


"""When it comes to finding out who your best customers are, the old RFM matrix principle is the best. RFM stands for Recency, Frequency and Monetary. It is a customer segmentation technique that uses past purchase behavior to divide customers into groups.
RFM Score Calculations
RECENCY (R): Days since last purchase
FREQUENCY (F): Total number of purchases
MONETARY VALUE (M): Total money this customer spent"""

df_new['tran_date'].min()

df_new['tran_date'].max()

NOW = dt.datetime(2014,12,3)

rfmTable = sdf.groupby('customer_Id').agg({'tran_date': lambda x: (NOW - x.max()).days, 'transaction_id': lambda x: len(x), 'total_amt': lambda x: x.sum()})

rfmTable['tran_date'] = rfmTable['tran_date'].astype(int)

rfmTable.rename(columns={'tran_date': 'recency', 
                         'transaction_id': 'frequency', 
                         'total_amt': 'monetary_value'}, inplace=True)

rfmTable.head()


# In[ ]:


sort_by_monetary_value = rfmTable.sort_values('monetary_value',ascending=False)

"""Customers with the highest purchases:"""
print(sort_by_monetary_value.head(n=10))


# In[ ]:


"""Getting rows for the most valuable customer."""
most_valued_customer = sdf.loc[sdf['customer_Id'] == 271834]


# In[ ]:


print(most_valued_customer)


# Results Obtained From The EDA (Exploratory Data Analysis)
# Transactions Period - January 2nd 2011 to December 2nd 2014.
# 
# 1. There was a sharp decline in sales from 2013 to 2014. It is nor clear why this occurred. Highest sales was recorded in the years 2012 and 2013.
# 
# 
# 2. More men were recorded as customers than women.
# 
# 
# 3. The city with the highest number of customers was city 4 -  the highest revenue was recorded in cities 4 and 3 and the lowest in city 6.
# 
# 
# 4.  Purchases were largely made online, with the e-Shop also accounting for the most returns.
# 
# 
# 5. Product categories: Books, Electronics and Home & Kitchen accounted for the most sold and most returned (difference between successful sales and returns is quite wide).
# 
# 
# 6. Considering subcategories - Women, Mens and Kids products sold more than the rest.
# 
# 
# 7. Customers aged between 40 - 50 accounted for the most sales.
# 
# 
# 8. Across the period in review, the most sales were recorded in the month January.
# 
# 
# 9. 8 out of 10 of the most valuable customers purchased items over 300 days before the last sale for the year 2014 was recorded.
# 
# 
# 10. Customer with ID 271834 spent the most ($44,967), purchasing only 10 items. He is a 38 year old resident of city 9. His purchases spanned across 2011 and 2013.
# 

# In[ ]:




