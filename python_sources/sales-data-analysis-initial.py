#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# reading data
# 

# In[ ]:


sales_data=pd.read_csv("../input/sales-data-sample.csv")
sales_data.head()


# Initial descriptive statistics to explore data set

# In[ ]:


sales_data.describe()


# returns a Series with the data type of each column

# In[ ]:


sales_data.dtypes


# In[ ]:


companies = sales_data[['name','price','date']]
companies.head()


# organize Pandas dataframes by group. 

# In[ ]:


companies_group = companies.groupby('name')
companies_group.size()


# Agreggrating data in the level of companies

# In[ ]:


total_sales = companies_group.sum()


# Visualization with bar chart.

# In[ ]:


plot1 = total_sales.plot(kind='bar')


# Define colums to be analyzed for faster analysis

# In[ ]:


products = sales_data[['product_name','price','date']]
products.head()


# organize Pandas dataframes by group. 

# In[ ]:


products_group = products.groupby('product_name')
products_group.size()


# In[ ]:


Aggregration


# In[ ]:


total_sales_by_product = products_group.sum()


# Bar chart representing total sales accross product types

# In[ ]:


plot2 = total_sales_by_product.plot(kind='bar')


# Distribution of values wit a histogram

# In[ ]:


plot3 = total_sales.plot(kind='hist')


# Box Plot (Whisker Box Plot) showing min, max ,median values along with quartiles

# In[ ]:


plot4 = total_sales.plot(kind='box')


# Box Plot (Whisker Box Plot) showing min, max ,median values of price column

# In[ ]:


plot5 = sales_data.price.plot(kind='box')


# Kernel density plot visualises the distribution of data over price

# In[ ]:


plot6 = sales_data.price.plot(kind='kde')

