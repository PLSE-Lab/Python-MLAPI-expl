#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **PROBLEM STATEMENT**
# 
# We will assume that the supermarket company is deciding to expand in any one of the cities of - Yangon, Naypyitaw or Mandalay. Using the sales data from already available stores of the company, we have to make a recommendation on which city shall they choose. Also, we will try to identify the liked product category and the target audience (gender, membership options, payment modes etc).
# 
# **OBJECTIVES**
# 
# Using the dataset to find out shopping behaviour of customers in the supermarket. 
# 1. We will do all required analysis to find out where (city and branch) is doing well and where it is not successful 
# 2. Find out most liked product categories and analyse the revenue they generate
# 3. Also try to gain insights on how customer behaviour varies depending on gender, membership status (Member / Normal)

# In[ ]:


import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv')
df.head(10)


# Initial Preprocessing : 
# 1. We can set the invoice id as index, as it is unique for every customer
# 2. We should drop the Tax 5% columns as this variable cannot be controlled by customer, neither by business owner and hence will provide no insight
# 3. Gross margin percentage column can also be dropped since it is same irresepctive of any factor

# In[ ]:


df = df.set_index('Invoice ID')


# In[ ]:


df = df.drop('Tax 5%', axis = 1)
df = df.drop('gross margin percentage', axis = 1)
df


# We can also drop Branch columns as each city has only one branch : 
# * Yangon - A
# * Mandalay - B
# * Naypyitaw - C

# In[ ]:


df = df.drop('Branch', axis = 1)


# Let us look at number of sales in all the cities and product types we might be interested in

# In[ ]:


df['City'].value_counts().sort_values(ascending = False)


# In[ ]:


df['Product line'].value_counts().sort_values(ascending = False)


# All cities have done almost equal in sales. 
# However, in prodcut category, Fashion accesories, Food and Beverages, Electronic accesories stand out from the rest.
# Now we shall investigate the revenue brought in by each category and city.

# In[ ]:


df_revenue_by_city = df.groupby('City').agg([np.sum, np.median])
df_revenue_by_city[['gross income']].T


# This is a common but interesting phenomenon. All cities did almost equal in sales (Naypyitaw had least if strictly seen). However, the net income (here gross income) is exceptionally high in Naypyitaw. Let us investigate this further.

# In[ ]:


df_revenue_by_city[['gross income', 'Unit price', 'Quantity']]


# This makes the picture clearer. Number of customers in Naypyitaw are the least but they tend to spend more on each purchase (Higher unit price) and also buy more number of goods in each purchase (6 as compared to 5 in other two cities).
# 
# **This makes Naypyitaw the clear choice for opening up another store**

# Now let us look at the product categories and find out which type of product should be stocked well in the new store in Naypyitaw

# In[ ]:


## Now onwards looking at revenue data only for Naypyitaw
df_revenue_by_product = df[df['City']== "Naypyitaw"].groupby('Product line').agg([sum, np.median])
df_revenue_by_product.sort_values(by = ('Quantity', 'sum'), ascending = False)


# In[ ]:


df_revenue_by_product.sort_values(by = ('gross income', 'sum'), ascending = False)


# In[ ]:


df_revenue_by_product.sort_values(by = ('Rating', 'median'), ascending = False)


# Here, there is no ambiguity. Food and Beverages, Fashion accesories, Electronic accesories are higly popular choices and also generate highest profits and hence they should be in the inventory well and also be included in marketing campaigns and advertisements.
# 
# Since Electronic accesories customers have rated the service lowest amongst others, the company needs to improve customer service in this domain in the new store. 

# Now we will find out the target market.

# In[ ]:


df_revenue_by_gender = df.groupby('Gender').agg([np.median, sum])
df_revenue_by_gender.T


# An interesting insight can be drawn from here. Looking at the first three variables - Unit Price, Quantity and Total : 
# 
# * Women spend less on each visit and men spend more
# * Women however buy more items on each visit
# * Also the total profit is higher for women
# * Women gave higher median rating
# 
# This implies : 
# 
# * Women could be interested in offers, especially the ones that offer to price less for more goods
# * Women buy cost-effective options
# * Hence, complimentary products should be kept together in the store and the company should try to introduce offers targeted to women
# * Ads should be designed keeping women shoppers trying to make most out of their money

# In[ ]:


df_revenue_by_mem = df.groupby('Customer type').agg([np.median, sum])
df_revenue_by_mem.T


# In[ ]:


df[df['City'] == "Naypyitaw"].groupby(['Customer type','Gender']).size()


# Findings : 
# 
# * Members spend more on each visit 
# * Both members and non-members buy approximately the same number of goods in a visit
# * Members clearly shop for more, maybe they give in to the urge of using their membership
# * Female shoppers are more iclined towards memberships
# 
# Recommendations : 
# 
# * Initiate extensive membership programs, especially for women

# In[ ]:


df[df['City'] == "Naypyitaw"].groupby(['Gender', 'Customer type', 'Product line']).size().sort_values(ascending=False)


# In[ ]:


df_revenue_by_pay = df[df['City'] == "Naypyitaw"].groupby('Payment')
df_revenue_by_pay.size().sort_values(ascending = False)


# In[ ]:


df[df['City'] == "Naypyitaw"].groupby(['Gender', 'Customer type', 'Payment']).size().sort_values(ascending=False)


# FINAL COMPLETE BUSINESS RECOMMENDATION : 
# 
# * The city chosen for expansion should be Naypyitaw
# * Open the store in Naypyitaw with a full-backed inventory of goods in product categories - Food and Beverages, Fashion Assesories and Electronics
# * The company should improve service in Electronics domain
# * The prime target for marketing should be women
# * Persausive ads for offering memberships, more to women
# * Increase service in cash payment modes 
