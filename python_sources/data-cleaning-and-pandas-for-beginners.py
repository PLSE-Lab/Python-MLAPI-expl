#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Loading the Data to Pandas**

# In[ ]:


flipkart_data = pd.read_csv(r"/kaggle/input/filpkart-onlineorders/OnlineOrders_of_a_ecommerce_website.csv")


# In[ ]:


flipkart_data.head()


# This Dataset content E-Commerce data with the date and time in which they have ordered.
# As this a simple dataset, It has done to test your skills on pandas and data visualization. As the dataset is a bit messy
# and not smooth, our goal is to make it clean and organized for data visualization. It is the first step to become 
# a Data Scientist i.e., you have to work on data cleaning

# **First, Giving proper column names**

# In[ ]:


flipkart_data.rename(columns={'crawl_timestamp': 'Timestamp',
                              'product_name': 'Product_Name',
                             'product_category_tree': 'Product_Category_Tree',
                             'retail_price': 'Retail_Price',
                             'discounted_price': 'Discounted_Price',
                             'brand': 'Brand'}, inplace=True)


# In[ ]:


flipkart_data.head()


# As we can see Product Category Tree column is not organised. So first make that
# column as category and extract the category name from it. Also we don't 
# want need the whole information.
# 
# As it have squared barckets and right arrow signs it look complex and confused structure. But break the things and watch carefully, the first word of the string is needed to fulfill our requirment.
# 
# So lets do it 

# In[ ]:


flipkart_data['Category'] = flipkart_data['Product_Category_Tree'].apply(lambda x: re.split('\[]*|\"|\>>|\,', x)[2])


# **So we successfully added a new column as category**

# In[ ]:


flipkart_data.head(5)


# **Removing  Product_Category_Tree Column**

# In[ ]:


flipkart_data.drop(['Product_Category_Tree'], axis = 1, inplace= True)


# In[ ]:


flipkart_data.head()


# **Removing unnecessary Zero(Milliseconds) from Timestamp**

# In[ ]:


flipkart_data['Timestamp'] = flipkart_data['Timestamp'].apply(lambda x: x.split('+')[0])


# In[ ]:


flipkart_data.head()


# **So finally our dataset is clean and organized. So save it as CSV file for future use!**

# **Read in updated dataframe**

# In[ ]:


# Save the data as csv file
flipkart_data.to_csv('fkartDataset.csv', index=False)


# In[ ]:


flkart_data = pd.read_csv('fkartDataset.csv')
flkart_data.head()


# **Clean up the missing value**

# In[ ]:


flkart_data.isnull().sum()


# **We don't have any missing value**

# ####  Question about this Dataset
# 
# ##### 1. What was the best month for sales? How much was earned that months?
# ##### 2. What time should we display advertisements to maximize the likelihood of purchases? 
# ##### 3. Which category sold most in that six month period?
# ##### 4. Top 10 product sold most in that six month period?

# ***************************************************************

# **1. What was the best month for sales? How much was earned that months?**

# So We can observe that we have a Timestamp Column but as per the Question We need months column, So we have the extract months from timestamp

# In[ ]:


#Adding the month column
flkart_data['Month'] = pd.to_numeric(pd.DatetimeIndex(flkart_data['Timestamp']).month)
flkart_data.head()


# In[ ]:


totalsum = flkart_data.groupby('Month').sum()
totalsum 


# In[ ]:


months = range(1, 7)
plt.bar(months, totalsum['Discounted_Price'])
plt.xticks(months)
plt.xlabel("Months")
plt.ylabel('Sales in INR')
plt.show()


# So January month is having more sales. May be due to new year eve

# **2.What time should we display advertisements to maximize the likelihood of purchases?**

# In[ ]:


flkart_data['Timestamp'] = pd.to_datetime(flkart_data['Timestamp'])


# As we have converted Timestamp column to datetime field. So we can easily create hour and minute columns

# In[ ]:


flkart_data['Hour'] = flkart_data['Timestamp'].dt.hour
flkart_data['Minute'] = flkart_data['Timestamp'].dt.minute


# In[ ]:


flkart_data.head()


# In[ ]:


hours = [hour for hour, df in flkart_data.groupby('Hour')]

plt.plot(hours, flkart_data.groupby(['Hour']).count())
plt.xticks(hours)
plt.xlabel('Hour')
plt.ylabel('Number of Orders')
plt.grid()
plt.show()


# **So we can conclude that morning and evening are the perfect time to displaying advertisements**

# **3. Which category sold most in that six month period?**

# In[ ]:


# So our target is to look after duplicates rows
dups_category = flkart_data.pivot_table(index=['Category'], aggfunc='size')


# In[ ]:


print(dups_category.nlargest(6))

x =list(range(1,7))

fig, ax = plt.subplots()
bar = sns.barplot(data=flkart_data, x=x , y=dups_category.nlargest(6), edgecolor="white")
ax.set_xticklabels(["Clothes", "Jewel", 'Mobile&Accessories', 'Home Decor', 'Footwear', 'Tools&Hardware'], rotation=90)
plt.show();


# **So obviously Clothings and Jewellery are the top categories to sold most.**

# **4. Top 10 product sold most in that six month period?**

# In[ ]:


# So our target is to look after duplicates rows
dups_product = flkart_data.pivot_table(index=['Product_Name'], aggfunc='size')

print(dups_product.nlargest(10))
items = range(10)

x =list(range(1,11))
fig, ax = plt.subplots()
bar = sns.barplot(data=flkart_data, x=x , y=dups_category.nlargest(10), edgecolor="white")
plt.show();


# **Here are the top 10 products available respectively**

# **You can do more research and dig into it for more information.**

# In[ ]:




