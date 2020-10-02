#!/usr/bin/env python
# coding: utf-8

# <h1> <i> <b> Super Market Data Analysis. </i> </b> </h1>

# <h3>A supermarket is self-service shop offering a wide variety of food, beverages and household products, organized into sections. It is larger and has a wider selection than earlier grocery stores, but is smaller and more limited in the range of merchandise than a hypermarket or big-box market.<br></br>

# <img src = "https://www.bereketinvestment.com/images/111X1250X600X1/1.1939257985995600424afa.jpg">

# <h3> Here I have used different techniques to viaualize the data set of supermarket.<br>
# What will you discover from this analysis?<p></h3>
# 1.Relation of customers with SuperMarket<br>
# 2.Payment methods used in supermarket.<br>
# 3.Products relation with quantities.<br>
# 4.Types of product and their sales.<br>
# 5.Products and their ratings.<br>
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h3> Loading dataset with computation time.</h3>

# In[ ]:


get_ipython().run_line_magic('time', 'data=pd.read_csv("/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv")')
print(data.shape)


# In[ ]:


data.head()


# <h3> Data Cleaning</h3>

# In[ ]:


data.isnull().sum()


# There are no missing value and the data set is clean so we will continue with data visualization.

# <h3> Checking information of data set.</h3>

# In[ ]:


data.info()


# In[ ]:


data.describe()


# <h3> Checking number of rows and columns</h3>

# In[ ]:


print("Dataset contains {} row and {} colums".format(data.shape[0],data.shape[1]))


# <h3> Visualization</h3>

# <img src = "https://media.mehrnews.com/d/2019/07/31/4/3194094.jpg" >

# <h3><i> Now we use different visualization tools to check different aspects of Supermarket sales.</i> </h3>

# <h3> Let's start with gender count</h3>

# In[ ]:


plt.figure(figsize=(14,6))
plt.style.use('fivethirtyeight')
ax= sns.countplot('Gender', data=data , palette = 'copper')
ax.set_xlabel(xlabel= "Gender",fontsize=18)
ax.set_ylabel(ylabel = "Gender count", fontsize = 18)
ax.set_title(label = "Gender count in supermarket", fontsize = 20)
plt.show()


# Here we can see that the number of males and females entering the store is almost equal. But the visulaization looks suspicious. Let's check numeric data. 

# In[ ]:


data.groupby(['Gender']). agg({'Total':'sum'})


# The visualization looks good. Let's carry on

# <h3> Customer type</h3>

# In[ ]:


plt.figure(figsize= (14,6))
ax = sns.countplot(x = "Customer type", data = data, palette = "rocket_r")
ax.set_title("Type of customers", fontsize = 25)
ax.set_xlabel("Customer type", fontsize = 16)
ax.set_ylabel("Customer Count", fontsize = 16)


# The visualization looks suspicious let's check numeric data.
# 

# In[ ]:


data.groupby(['Customer type']). agg({'Total':'sum'})


# <b>Seems about right

# <h3> Above we can see the type of customer in all branch combined now let's check for different branch.</h3>

# In[ ]:


plt.figure(figsize=(14,6))
ax = sns.countplot(x = "Customer type", hue = "Branch", data = data, palette= "rocket_r")
ax.set_title(label = "Customer type in different branch", fontsize = 25)
ax.set_xlabel(xlabel = "Branches", fontsize = 16)
ax.set_ylabel(ylabel = "Customer Count", fontsize = 16)


# <h3> Checking the different payment methods used.</h3>

# In[ ]:


plt.figure(figsize = (14,6))
ax = sns.countplot(x = "Payment", data = data, palette = "tab20")
ax.set_title(label = "Payment methods of customers ", fontsize= 25)
ax.set_xlabel(xlabel = "Payment method", fontsize = 16)
ax.set_ylabel(ylabel = " Customer Count", fontsize = 16)


# <h3> Payment method distribution in all branches</h3>

# In[ ]:


plt.figure(figsize = (14,6))
ax = sns.countplot(x="Payment", hue = "Branch", data = data, palette= "tab20")
ax.set_title(label = "Payment distribution in all branches", fontsize= 25)
ax.set_xlabel(xlabel = "Payment method", fontsize = 16)
ax.set_ylabel(ylabel = "Peple Count", fontsize = 16)


# <h3>Now let's see the rating distribution in 3 branches</h3>

# In[ ]:


plt.figure(figsize=(14,6)) 
ax = sns.boxplot(x="Branch", y = "Rating" ,data =data, palette= "RdYlBu")
ax.set_title("Rating distribution between branches", fontsize = 25)
ax.set_xlabel(xlabel = "Branches", fontsize = 16)
ax.set_ylabel(ylabel = "Rating distribution", fontsize = 16)


# We can see that the average rating of branch A and C is more than seven and branch B is less than 7.
# 

# <h3> Max sells time<h3>

# In[ ]:


data["Time"]= pd.to_datetime(data["Time"])


# In[ ]:


data["Hour"]= (data["Time"]).dt.hour


# In[ ]:


plt.figure(figsize=(14,6)) 
SalesTime = sns.lineplot(x="Hour", y ="Quantity", data = data).set_title("product sales per Hour")


# We can see that the supermarket makes most of it's sells in 14:00 hrs local time.

# <h3> Rating vs sales</h3>

# In[ ]:


plt.figure(figsize=(14,6)) 
rating_vs_sales = sns.lineplot(x="Total", y= "Rating", data=data)


# <h3> Using boxen plot</h3>

# In[ ]:


plt.figure(figsize=(10,6)) 
ax = sns.boxenplot(x = "Quantity", y = "Product line", data = data,)
ax.set_title(label = "Average sales of different lines of products", fontsize = 25)
ax.set_xlabel(xlabel = "Qunatity Sales",fontsize = 16)
ax.set_ylabel(ylabel = "Product Line", fontsize = 16)


# Here we can see that the average sales of different lines of products. Health and beauty making the highest sales whereas Fashon accessories making the lowest sales.

# <h3> Let's see the sales count of these products.</h3> 

# In[ ]:


plt.figure(figsize=(14,6))
ax = sns.countplot(y='Product line', data=data, order = data['Product line'].value_counts().index)
ax.set_title(label = "Sales count of products", fontsize = 25)
ax.set_xlabel(xlabel = "Sales count", fontsize = 16)
ax.set_ylabel(ylabel= "Product Line", fontsize = 16)


# We can see the top sold products form the above figure.

# <h3> Total sales of product using boxenplot</h3> 

# In[ ]:


plt.figure(figsize=(14,6))
ax = sns.boxenplot(y= "Product line", x= "Total", data = data)
ax.set_title(label = " Total sales of product", fontsize = 25)
ax.set_xlabel(xlabel = "Total sales", fontsize = 16)
ax.set_ylabel(ylabel = "Product Line", fontsize = 16)


# <h3> Now let's see average ratings of products.</h3>

# In[ ]:


plt.figure(figsize = (14,6))
ax = sns.boxenplot(y = "Product line", x = "Rating", data = data)
ax.set_title("Average rating of product line", fontsize = 25)
ax.set_xlabel("Rating", fontsize = 16)
ax.set_ylabel("Product line", fontsize = 16)


# <h3>Product sales on the basis of gender</h3>

# In[ ]:


plt.figure(figsize = (14,6))
ax= sns.stripplot(y= "Product line", x = "Total", hue = "Gender", data = data)
ax.set_title(label = "Product sales on the basis of gender")
ax.set_xlabel(xlabel = " Total sales of products")
ax.set_ylabel(ylabel = "Product Line")


# <h3>Product and gross income</h3> 

# In[ ]:


plt.figure(figsize = (14,6))
ax = sns.relplot(y= "Product line", x = "gross income", data = data)
# ax.set_title(label = "Products and Gross income")
# ax.set_xlabel(xlabel = "Total gross income")
# ax.set_ylabel(ylabel = "Product line")


# <h3> Here we can see the gross income of different product line </h3> 

# In[ ]:




