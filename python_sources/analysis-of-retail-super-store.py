#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 02:28:14 2019

@author: Bheemeswara Sarma Kalluri

Content for Retail Sales Analysis
    0. Clear Memory
    1. Import
    2. Read data
    3. Function
    4. Explore data
    5. Visualization
    6. Data Analysis
    
Resource on Kaggle: https://www.kaggle.com/jr2ngb/superstore-data
"""

# 0 Clear memory
get_ipython().run_line_magic('reset', '-f')


# In[ ]:


# 1.1 Call data manipulation libraries
import pandas as pd
import numpy as np
import os
# 1.2 Plotting libraries to plot feature importance
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# 2.0 Set working directory and read file
print(os.getcwd())
os.listdir()
pd.options.display.max_columns = 300
# 2.1 Read train/test files
data = pd.read_csv("../input/superstore_dataset2011-2015.csv", encoding = "ISO-8859-1")


# In[ ]:


#3 Functions
# 3.1 Print 5 Rows for any column
def print_rows(name_column):
    return data[name_column][0:5]
# 3.2 Get Details of the Column
def describe_column(name_column):
    return data[name_column].describe()


# In[ ]:


# 4 Explore data
# 4.1 Shape
data.shape


# In[ ]:


# 4.2 Columns
data.columns 


# In[ ]:


# 4.3 Is Null
data.isnull().sum()


# In[ ]:


# 4.4 Head 5 Rows (Default)
data.head()


# In[ ]:


# 4.5 Tail gives last 5 rows
data.tail()


# In[ ]:


# 4.6 Information
data.info


# In[ ]:


# 4.7 Code for Order ID & its Columns
print_rows('Order ID')


# In[ ]:


describe_column('Order ID')


# In[ ]:


# 4.8 Code for Order Date & its Columns
print_rows('Order Date')


# In[ ]:


data['Order Date'] = pd.to_datetime(data['Order Date'])


# In[ ]:


describe_column('Order Date')


# In[ ]:


# 4.9 Code for Ship Date & its Columns
print_rows('Ship Date')


# In[ ]:


data['Ship Date'] = pd.to_datetime(data['Ship Date'])


# In[ ]:


describe_column('Ship Date')


# In[ ]:


# 4.10 Code for Ship Mode & its Columns
print_rows('Ship Mode')


# In[ ]:


describe_column('Ship Mode')


# In[ ]:


data['Ship Mode'].unique()


# In[ ]:


# 4.11 Code for Customer ID & its Columns
print_rows('Customer ID')


# In[ ]:


describe_column('Customer ID')


# In[ ]:


# 4.12 Code for Customer Name & its Columns
print_rows('Customer Name')


# In[ ]:


describe_column('Customer Name')


# In[ ]:


data['Customer Name'].unique()


# In[ ]:


# 4.13 Code for Segment & its Columns
print_rows('Segment')


# In[ ]:


describe_column('Segment')


# In[ ]:


data['Segment'].unique()


# In[ ]:


# 4.14 Code for City & its Columns
print_rows('City')


# In[ ]:


describe_column('City')


# In[ ]:


data['City'].unique()


# In[ ]:


# 4.15 Code for State & its Columns
print_rows('State')


# In[ ]:


describe_column('State')


# In[ ]:


data['State'].unique()


# In[ ]:


# 4.16 Code for Country & its Columns
print_rows('Country')


# In[ ]:


describe_column('Country')


# In[ ]:


data['Country'].unique()


# In[ ]:


# 4.17 Code for Postal Code & its Columns
print_rows('Postal Code')


# In[ ]:


describe_column('Postal Code')


# In[ ]:


# 4.18 Code for Market & its Columns
print_rows('Market')


# In[ ]:


describe_column('Market')


# In[ ]:


data['Market'].unique()


# In[ ]:


# 4.19 Code for Region & its Columns
print_rows('Region')


# In[ ]:


describe_column('Region')


# In[ ]:


data['Region'].unique()


# In[ ]:


# 4.20 Code for Product ID & its Columns
print_rows('Product ID')


# In[ ]:


describe_column('Product ID')


# In[ ]:


data['Product ID'].unique()


# In[ ]:


# 4.21 Code for Category & its Columns
print_rows('Category')


# In[ ]:


describe_column('Category')


# In[ ]:


data['Category'].unique()


# In[ ]:


# 4.22 Code for Sub-Category & its Columns
print_rows('Sub-Category')


# In[ ]:


describe_column('Sub-Category')


# In[ ]:


data['Sub-Category'].unique()


# In[ ]:


# 4.23 Code for Product Name & its Columns
print_rows('Product Name')


# In[ ]:


describe_column('Product Name')


# In[ ]:


data['Product Name'].unique()


# In[ ]:


# 4.24 Code for Sales & its Columns
print_rows('Sales')


# In[ ]:


describe_column('Sales')


# In[ ]:


# 4.25 Code for Quantity & its Columns
print_rows('Quantity')


# In[ ]:


describe_column('Quantity')


# In[ ]:


# 4.26 Code for Discount & its Columns
print_rows('Discount')


# In[ ]:


describe_column('Discount')


# In[ ]:


# 4.27 Code for Profit & its Columns
print_rows('Profit')


# In[ ]:


describe_column('Profit')


# In[ ]:


# 4.28 Code for Shipping Cost & its Columns
print_rows('Shipping Cost')


# In[ ]:


describe_column('Shipping Cost')


# In[ ]:


# 4.29 Code for Order Priority & its Columns
print_rows('Order Priority')


# In[ ]:


describe_column('Order Priority')


# In[ ]:


data['Order Priority'].unique()


# In[ ]:


# 5 Data Visualization
# 5.1 Market
plt.figure(figsize=(16,8))
data['Market'].value_counts().plot.bar()
plt.title('Market Wise Sales')
plt.ylabel('Count')
plt.xlabel('Market Region')
plt.show()
# APAC tops all the Markets


# In[ ]:


# 5.2 Top 20 
# 5.2.1 Top 20 Countries in sales
plt.figure(figsize=(16,8))
top20countries = data.groupby('Country')['Row ID'].count().sort_values(ascending=False)
top20countries = top20countries [:20]
top20countries.plot(kind='bar', color='green')
plt.title('Top 20 Countries in Sales')
plt.ylabel('Count')
plt.xlabel('Countries')
plt.show()
# United States as a Country tops all the Countries in Sales


# In[ ]:


# 5.2.2 Top 20 States in Sales
plt.figure(figsize=(16,8))
top20states = data.groupby('State')['Row ID'].count().sort_values(ascending=False)
top20states = top20states [:20]
top20states.plot(kind='bar', color='blue')
plt.title('Top 20 States in Sales')
plt.ylabel('Count')
plt.xlabel('States')
plt.show()
# California as a State tops all the States in Sales


# In[ ]:


# 5.2.3 Top 20 Cities in Sales
plt.figure(figsize=(16,8))
top20city = data.groupby('City')['Row ID'].count().sort_values(ascending=False)
top20city = top20city [:20]
top20city.plot(kind='bar', color='red')
plt.title('Top 20 Cities in Sales')
plt.ylabel('Count')
plt.xlabel('Cities')
plt.show()
# New York City as a City tops all the Cities in Sales


# In[ ]:


# 5.2.4 Top 20 Product by its Product ID's in Sales
plt.figure(figsize=(16,8))
top20pid = data.groupby('Product ID')['Row ID'].count().sort_values(ascending=False)
top20pid = top20pid [:20]
top20pid.plot(kind='bar', color='Yellow')
plt.title('Top 20 Products by Product IDs in Sales')
plt.ylabel('Count')
plt.xlabel('Product IDs')
plt.show()
# OFF-AR-10003651 Product tops all the Products in Sales


# In[ ]:


# 5.2.5 Top 20 Products in Sales
plt.figure(figsize=(16,8))
top20pname = data.groupby('Product Name')['Row ID'].count().sort_values(ascending=False)
top20pname = top20pname [:20]
top20pname.plot(kind='bar', color='Orange')
plt.title('Top 20 Products in Sales')
plt.ylabel('Count')
plt.xlabel('Products')
plt.show()
# Staples tops all the Products in Sales


# In[ ]:


# 5.2.6 Top 20 Profitable Customers
# Who are the top-20 most profitable customers.
plt.figure(figsize=(16,8))
bar_width = 0.2
opacity= 0.8
top20profit = data.sort_values('Profit', ascending=False)
top20 = top20profit.head(20)
top20[['Customer Name', 'Profit']]
sns.barplot(x = "Customer Name", y= "Profit", data=top20)  # plotting of top 20 profitable customers
plt.show()
# Tamara Chand tops all the Customers in Profits.


# In[ ]:


# 5.3 Segment
plt.figure(figsize=(16,8))
data['Segment'].value_counts().plot.bar()
# sns.countplot("Segment", data = data)           #Distribution of custome Segment
plt.title('Segment Wise Sales')
plt.ylabel('Count')
plt.xlabel('Segments')
plt.show()
# APAC tops all the Markets


# In[ ]:


# 5.4 Ship Mode
plt.figure(figsize=(16,8))
data['Ship Mode'].value_counts().plot.bar()
plt.title('Ship Mode Wise Sales')
plt.ylabel('Sales')
plt.xlabel('Ship Modes')
plt.show()
# Standard Class tops all the Transport Methods


# In[ ]:


# 5.5 Region
plt.figure(figsize=(16,8))
data['Region'].value_counts().plot.bar()
plt.title('Region Wise Sales')
plt.ylabel('Sales')
plt.xlabel('Regions')
plt.show()
# Central Region tops all the Sales in Regions


# In[ ]:


# 5.6 Category
plt.figure(figsize=(16,8))
data['Category'].value_counts().plot.bar()
plt.title('Category Wise Sales')
plt.ylabel('Sales')
plt.xlabel('Categories')
plt.show()
# Office Supplies tops all the Sales in Categories


# In[ ]:


# 5.7 Sub-Category
plt.figure(figsize=(16,8))
data['Sub-Category'].value_counts().plot.bar()
plt.title('Sub-Category Wise Sales')
plt.ylabel('Sales')
plt.xlabel('Sub Categories')
plt.show()
# Binders tops all the Sales in Sub Categories


# In[ ]:


# 5.8 Order Priority
plt.figure(figsize=(16,8))
data['Order Priority'].value_counts().plot.bar()
plt.title('Order Priority Wise Sales')
plt.ylabel('Sales')
plt.xlabel('Order Priorities')
plt.show()
# Medium tops all the Sales in Order Priorities


# In[ ]:


# 5.9 Relationship of Order Priority and Profitability
plt.figure(figsize=(16,8))
sns.boxplot("Order Priority","Profit",data= data)
plt.title('Relationship of Order Priority and Profitability')
plt.show()
# Profits slightly higher when Order priority is Medium


# In[ ]:


# 5.10 Distribution of Customers Market wise.
#6. What is the distribution of customers Market wise?
plt.figure(figsize=(16,8))
sns.countplot("Market",data = data)
plt.title('Distribution of Customers Market wise')
plt.show()
# Market has 7 levels. APAC has the largest # of customers followed by LATAM, and US in that order
# Canada has the least # of customers


# In[ ]:


# 5.11 Distribution of customers Market wise and Region wise
plt.figure(figsize=(16,8))
data['Region'].value_counts()
cmr = pd.DataFrame({'Count' : data.groupby(["Market","Region","Customer Name"]).size()}).reset_index()
sns.countplot("Market", hue= "Region", data = cmr)
plt.show()
#for APAC, the largest # of customers are basd out of Oceania, followed by Southeast Asia
#for US, the largest # of customers are based out of Western Region followed by East


# In[ ]:


# 5.12 Distribution of  Customers by Country & State - top 20
plt.figure(figsize=(16,8))
CusCountry = pd.DataFrame({'Count' : data.groupby(["Country","State"]).size()}).reset_index().sort_values('Count',ascending = False).head(20)
sns.barplot(x = "Country", y= "Count", hue="State", data = CusCountry.sort_values('Country'))
plt.show()
## US has the largest number of customers -California being the largest followed by New York, Washington, Illinois & Ohio
## UK has the next largest population of Customers -England


# In[ ]:


# 5.13 sales by product Category, Sub-category
plt.figure(figsize=(16,8))
sale_category = data.groupby(["Category","Sub-Category"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False)
sns.barplot(x = "Category", hue="Sub-Category", y= "Quantity", data=sale_category)
plt.show()
# Binders in Office Supplies tops the list.


# In[ ]:


# 5.14 Customer Segment
# 2. What is the distribution of our customer segment
plt.figure(figsize=(24,15))
sns.catplot(x="Segment", col="Market", data=data, kind="count")
plt.show()
# Market wise segments are shown.


# In[ ]:


# 5.15 Order Priority and Profit
#5. Relationship of Order Priority and Profit
plt.figure(figsize=(16,8))
sns.barplot("Order Priority", y= "Profit",data=data)
plt.show()
# Low Order of Priority has High Profit.


# In[ ]:


# 6 No Charts
# 6.1 Top 20 Long Standing Customers
#3. Who are our top-20 oldest customers
data['Order Date'] = pd.to_datetime(data['Order Date'])      
top20Cust= data.sort_values(['Order Date'], ascending=False).head(20)
top20Cust.loc[:,['Customer Name']]
# No Chart.


# In[ ]:


# 6.2 Customers visited only once
#4. Which customers have visited this store just once
Visit=data.groupby('Customer ID').apply(lambda x: pd.Series(dict(visit_count=x.shape[0])))
Visit.loc[(Visit.visit_count==1)]
# No Chart.
# 7 Customers visited only once.


# In[ ]:




