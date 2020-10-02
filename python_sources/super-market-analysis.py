#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


sales = pd.read_csv('../input/supermarket_sales - Sheet1.csv')


# In[ ]:


sales.head()


# In[ ]:


sales.info()


# By inspection, the 'Date' datatype is an object, we need to change it to datetime

# In[ ]:


sales['date'] = pd.to_datetime(sales['Date'])


# In[ ]:


sales['date'].dtype


# In[ ]:


type(sales['date'])


# In[ ]:


sales['date'] = pd.to_datetime(sales['date'])


# In[ ]:


sales['day'] = (sales['date']).dt.day
sales['month'] = (sales['date']).dt.month
sales['year'] = (sales['date']).dt.year


# In[ ]:


sales['Time'] = pd.to_datetime(sales['Time'])


# In[ ]:


sales['Hour'] = (sales['Time']).dt.hour    #type(sales['Time'])


# Let's see the unique hours of sales in this dataset

# In[ ]:


sales['Hour'].nunique()  #gives us the number of unique hours 


# In[ ]:


sales['Hour'].unique()


# In[ ]:


sales.describe()


#  ### Let's find the number of unique values in columns with object datatype

# In[ ]:


categorical_columns = [cname for cname in sales.columns if sales[cname].dtype == "object"]


# In[ ]:


categorical_columns


# In[ ]:


print("# unique values in Branch: {0}".format(len(sales['Branch'].unique().tolist())))
print("# unique values in City: {0}".format(len(sales['City'].unique().tolist())))
print("# unique values in Customer Type: {0}".format(len(sales['Customer type'].unique().tolist())))
print("# unique values in Gender: {0}".format(len(sales['Gender'].unique().tolist())))
print("# unique values in Product Line: {0}".format(len(sales['Product line'].unique().tolist())))
print("# unique values in Payment: {0}".format(len(sales['Payment'].unique().tolist())))


# In[ ]:


sns.set(style="darkgrid")       #style the plot background to become a grid
genderCount  = sns.countplot(x="Gender", data =sales).set_title("Gender_Count")


# In[ ]:


sns.boxplot(x="Branch", y = "Rating" ,data =sales).set_title("Ratings by Branch") 


# Branch B has the lowest rating among all the branches

# *Sales by the hour in the comapny* Most of the item were sold around 14:00 hrs local time

# In[ ]:


genderCount  = sns.lineplot(x="Hour",  y = 'Quantity',data =sales).set_title("Product Sales per Hour")


# Below we can see how each branch's sales quantity looks like by the hour in a monthly fashion 

# In[ ]:


genderCount  = sns.relplot(x="Hour",  y = 'Quantity', col= 'month' , row= 'Branch', kind="line", hue="Gender", style="Gender", data =sales)


#  Below we can see each branch's sales by the hour in a monthly fashion 

# In[ ]:


genderCount  = sns.relplot(x="Hour",  y = 'Total', col= 'month' , row= 'Branch', estimator = None, kind="line", data =sales)


# In[ ]:


sales['Rating'].unique()


# In[ ]:


ageDisSpend = sns.lineplot(x="Total", y = "Rating", data =sales)


# ## Product Analysis
# 
# Let's look at the various products' performance.

# In[ ]:


sns.boxenplot(y = 'Product line', x = 'Quantity', data=sales )


# From the above visual, Health and Beauty,Electronic accessories, Homem and lifestyle, Sports and travel have a better average quantity sales that food and beverages as well as Fashion accessories. 

# In[ ]:


sns.countplot(y = 'Product line', data=sales, order = sales['Product line'].value_counts().index )


# From the above image shows the top product line item type sold in the given dataset. Fashion Accessories is the highest while Health and beauty is the lowest

# In[ ]:


sns.boxenplot(y = 'Product line', x = 'Total', data=sales )


# In[ ]:


sns.stripplot(y = 'Product line', x = 'Total', hue = 'Gender', data=sales )


# In[ ]:


sns.relplot(y = 'Product line', x = 'gross income', data=sales )


# In[ ]:


sns.boxenplot(y = 'Product line', x = 'Rating', data=sales )


# Food and Beverages have the highest average rating while sports and travel the lowest

# Let's see when customers buy certain products in the various branches. 

# In[ ]:


productCount  = sns.relplot(x="Hour",  y = 'Quantity', col= 'Product line' , row= 'Branch', estimator = None, kind="line", data =sales)


# From the above plots, we can see that food and beverages sales usually high in all three branches at evening especially around 19:00 

# # Payment Channel

# Let see how customers make payment in this business

# In[ ]:


sns.countplot(x="Payment", data =sales).set_title("Payment Channel") 


# Most of the customers pay through the Ewallet and Cash Payment while under 40 percent of them pay with their credit card. We would also like to see this payment type distribution across all the branches

# In[ ]:


sns.countplot(x="Payment", hue = "Branch", data =sales).set_title("Payment Channel by Branch") 


# # Customer Analysis

# From inspection, there are two types of customers. Members and Normal. Let's see how many they are and where they are 

# In[ ]:


sales['Customer type'].nunique()


# In[ ]:


sns.countplot(x="Customer type", data =sales).set_title("Customer Type") 


# In[ ]:


sns.countplot(x="Customer type", hue = "Branch", data =sales).set_title("Customer Type by Branch") 


# ## Does customer type influences the sales 

# In[ ]:


sales.groupby(['Customer type']).agg({'Total': 'sum'})


# In[ ]:


sns.barplot(x="Customer type", y="Total", estimator = sum, data=sales)


# Do the customer type influence customer rating? Let's find out 

# In[ ]:


sns.swarmplot(x="Customer type",  y = "Rating",  hue = "City", data =sales).set_title("Customer Type") 


# With the use of google search, I was able to get the longitude and latitude of each cities. We can 

# In[ ]:


long = {"Yangon": 16.8661, "Naypyitaw": 19.7633, "Mandalay": 21.9588 }
lat = {"Yangon": 96.1951, "Naypyitaw": 96.0785, "Mandalay": 96.0891 }
for set in sales:
    sales['long'] = sales['City'].map(long)
    sales['lat'] = sales['City'].map(lat)


# In[ ]:


sns.scatterplot(x="long",  y = "lat",size = "Total", data =sales, legend = "brief").set_title("Customer Type") 


# In[ ]:


sns.relplot(x="Total",  y = "Quantity", data =sales)


# # Hello guys, Thank you everyone who have been following this kernel. Thank you for the likes. I have managed to learning some machine learning techniques after hosting this kernel some ten months ago. I will apply my machine learning knowledge to this data set so we can learn how can be learn more of this dataset. Stay Tuned!
# 
# https://www.kaggle.com/akpflow/super-market-analysis/edit/run/16569511

# In[ ]:




