#!/usr/bin/env python
# coding: utf-8

# **Importing Libraries**

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


# **Importing Dataset**

# In[ ]:


customersDF = pd.read_csv("../input/brazilian-ecommerce/olist_customers_dataset.csv")
geolocationDF = pd.read_csv("../input/brazilian-ecommerce/olist_geolocation_dataset.csv")
order_itemsDF = pd.read_csv("../input/brazilian-ecommerce/olist_order_items_dataset.csv")
order_paymentsDF = pd.read_csv("../input/brazilian-ecommerce/olist_order_payments_dataset.csv")
order_reviewsDF = pd.read_csv("../input/brazilian-ecommerce/olist_order_reviews_dataset.csv")
ordersDF = pd.read_csv("../input/brazilian-ecommerce/olist_orders_dataset.csv")
productsDF = pd.read_csv("../input/brazilian-ecommerce/olist_products_dataset.csv")
sellersDF = pd.read_csv("../input/brazilian-ecommerce/olist_sellers_dataset.csv")
product_category_name_translation = pd.read_csv("../input/brazilian-ecommerce/product_category_name_translation.csv")


# # Data Cleaning

# * **Customer Dataset - No Nulls** 
# * **Geolocation Dataset - No Nulls**
# * **Order Items Dataset - No Nulls**
# * **Order Payments Dataset - No Nulls**
# * **Sellers Dataset - No Nulls**
# * **Product Category Name Translation Dataset - No Nulls**

# In[ ]:


customersDF.info()
geolocationDF.info()
order_itemsDF.info()
order_paymentsDF.info()
sellersDF.info()
product_category_name_translation.info()


# > Changing column name of zip code prefix in customer, seller and geolocation dataframe

# **Order Reviews Dataset - Nulls are there**  
# > review_comment_title and review_comment_message fields are empty which is fine, we will not be using them in out analysis so we will drop the columns in addition to review_creation_date and review_answer_timestamp columns

# In[ ]:


order_reviewsDF.info()
order_reviewsDF.drop(columns = ['review_comment_title','review_comment_message','review_creation_date','review_answer_timestamp'],inplace=True)
order_reviewsDF.head()


# In[ ]:


order_reviewsDF.info()
order_reviewsDF.describe()


# **Orders Dataset - Nulls are there**  
# > dropping columns : order_approved_at,order_delivered_carrier_date,order_delivered_customer_date and order_estimated_delivery_date

# In[ ]:


ordersDF.info()
ordersDF.drop(columns = ['order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date'],inplace=True)
ordersDF.head()


# In[ ]:


ordersDF.info()
ordersDF.describe()


# **Products Dataset - Nulls are there**  
# > We will be dropping the rows that have missing product_category_name, product_name_lenght, product_description_lenght, product_photos_qty
# > which are 610 rows of 32951

# In[ ]:


productsDF.info()
print("\n")
print(productsDF.shape[0])
indexOfNullProductCategory = productsDF[productsDF['product_category_name'].isnull()].index
productsDF.drop(inplace=True, index=indexOfNullProductCategory)


# In[ ]:


productsDF.info()
print("\n")
print(productsDF.shape[0])
indexOfNullAdditional = productsDF[productsDF['product_weight_g'].isnull()].index
productsDF.drop(inplace=True, index=indexOfNullAdditional)
print("\n")
print(productsDF.shape[0])
print("\n")
productsDF.info()


# In[ ]:


productsDF.describe()


# # Creating Master DataFrame (By Applying Inner Join)

# In[ ]:


masterDF = ordersDF.copy()
masterDF = masterDF.merge(customersDF,on='customer_id',indicator = True)
masterDF = masterDF.merge(order_reviewsDF,on='order_id')
masterDF = masterDF.merge(order_paymentsDF,on='order_id')
masterDF = masterDF.merge(order_itemsDF,on='order_id')
masterDF = masterDF.merge(productsDF,on='product_id')
masterDF = masterDF.merge(sellersDF,on='seller_id')
masterDF.head()


# In[ ]:


masterDF.shape


# > Testing to confirm there is no null values in master dataframe

# In[ ]:


masterDF.isnull().sum()


# # Product Recommendation based on Popularity for New Customers
# > As we don't know anything about the likes of customer since we don't have their buying history

# In[ ]:


masterDF.info()


# **Most Sold Products**

# In[ ]:


popular_products = pd.DataFrame(masterDF.groupby('product_id')['review_score'].count())
most_sold = popular_products.sort_values('review_score', ascending=False)
most_sold.head(30).plot(kind = "bar")


# **Highest Rated Products**

# In[ ]:


highestRated = pd.DataFrame(masterDF.groupby('product_id').agg(
    review_score_Avg = ('review_score', 'mean'),
    review_score_Count = ('review_score', 'count')
    ))

highestRated.sort_values(['review_score_Avg','review_score_Count'],ascending=False,inplace=True)           
highestRated.head(30)


# # Rough

# In[ ]:


import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use("ggplot")


import sklearn
from sklearn.decomposition import TruncatedSVD


# In[ ]:




