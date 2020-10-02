#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load Datasets
customers = pd.read_csv("../input/olist_customers_dataset.csv")
geolocation = pd.read_csv("../input/olist_geolocation_dataset.csv")
order_items = pd.read_csv("../input/olist_order_items_dataset.csv")
order_payments = pd.read_csv("../input/olist_order_payments_dataset.csv")
reviews = pd.read_csv("../input/olist_order_reviews_dataset.csv")
orders = pd.read_csv("../input/olist_orders_dataset.csv")
products = pd.read_csv("../input/olist_products_dataset.csv")
sellers = pd.read_csv("../input/olist_sellers_dataset.csv")
translations = pd.read_csv("../input/product_category_name_translation.csv")


# In[ ]:


products.head()


# # Data Exploration
# 
# First, lets see how many different products we are working with, how many categories and distribution per category. We will translate the names as a first step.

# In[ ]:


# Convert the translations to dictionary format
translations = translations.set_index('product_category_name')['product_category_name_english'].to_dict()

# translate the product category column in the products df to English
products['product_category_name'] = products['product_category_name'].map(translations)


# In[ ]:


#total number of unique products and categories
print("total unique products = " + str(len(products['product_id'])))
print("total unique categories = " + str(len(products['product_category_name'].unique())))


# In[ ]:


# Function to plot bar graphs
def plot_bar_graph(x,y,title):
    fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharey=True)
    axs.bar(x, y)
    axs.set_title(title)
    plt.xticks(rotation =90)

    # data labels
    for i, v in enumerate(y):
        axs.text(i-.25, 
                  v+10, 
                  y[i], 
                  fontsize=8, 
                  #color=label_color_list[i]
                )
    return plt.show()

# Function to plot line graph
def plot_line_graph(x,y,title):
    fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharey=True)
    axs.plot(x, y)
    axs.set_title(title)
    plt.xticks(rotation =90)

    # data labels
    for i, v in enumerate(y):
        axs.text(i-.25, 
                  v+10, 
                  y[i], 
                  fontsize=8, 
                  #color=label_color_list[i]
                )
    return plt.show()

# Function to plot bar graphs
def plot_scatter_graph(x,y,title):
    fig, axs = plt.subplots(1, 1, figsize=(20, 10), sharey=True)
    axs.scatter(x, y)
    axs.set_title(title)
    plt.xticks(rotation =90)

    # data labels
    for i, v in enumerate(y):
        axs.text(i-.25, 
                  v+10, 
                  y[i], 
                  fontsize=8, 
                  #color=label_color_list[i]
                )
    return plt.show()


# In[ ]:


# check for missing values
# since we are only concerned about these two columns we will check only these two:
print("Missing Values = " + str(products[["product_id","product_category_name"]].isna().values.sum()))

products[["product_id","product_category_name"]].isna().any(axis =1)

# drop the products with missing category names
products = products.dropna(subset=['product_id', 'product_category_name'])
products.head()


# In[ ]:


# distribution of unique products per category
product_category = products[['product_id','product_category_name']] .groupby('product_category_name')['product_id'] .count().sort_values(ascending=False) .to_dict()

product_category_names = list(product_category.keys())[:20]
product_category_values = list(product_category.values())[:20]

plot_bar_graph(product_category_names,product_category_values,"Top 20 Unique Products per Category")


# In[ ]:


# Orders per Product
# Join the two datasets
order_products = pd.merge(order_items, products, left_on = 'product_id', right_on = 'product_id')

# plot the data
order_products[['product_category_name','order_id']]


# In[ ]:


# check for missing values
# since we are only concerned about these two columns we will check only these two:
print("Missing Values = " + str(order_products[["product_category_name","order_id"]].isna().values.sum()))

order_products[order_products[["product_category_name","order_id"]].isna().any(axis =1)]

# drop the orders with missing category names
order_products = order_products.dropna(subset=["product_category_name","order_id"])


# In[ ]:


# plot the data
order_products_dict = order_products[['product_category_name','order_id']] .groupby('product_category_name')['order_id'] .count().sort_values(ascending=False) .to_dict()


order_product_names = list(order_products_dict.keys())[:20]
order_product_values = list(order_products_dict.values())[:20]

plot_bar_graph(order_product_names,order_product_values,"Top 20 Product Categories by Orders")


# In[ ]:


# Check how many orders have more than one product
print("Total Orders: " + str(len(order_items['order_id'].unique())))
print("Total Orders with 2 or more items: " + str(len(order_items[order_items['order_item_id'] == 2])))


# In[ ]:


order_products.head()


# # Association Mining

# We will perform association mining on two levels, category wise and then product wise within those categories. This is because there are too many different products for us to perform association mining across all products.

# ## Category Association

# ### Data Preprocessing

# In[ ]:


# Transforming the data into the correct format

basket = order_products.groupby(['order_id','product_category_name'])['order_item_id']                                    .sum()                                     .unstack()                                     .reset_index()                                     .fillna(0)                                     .set_index('order_id')

# recode all multiple purchases to 1
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1: 
        return 1
basket_sets = basket.applymap(encode_units)
basket_sets.head()


# In[ ]:


# remove all orders with less than 2 different categories
basket_sets = basket_sets[basket_sets.sum(axis = 1) > 1]
basket_sets[basket_sets.sum(axis = 1) > 1]


# ### Data Analysis

# In[ ]:


frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames = True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)

df_results1 = pd.DataFrame(rules)
df_results1.sort_values(["confidence"],ascending=False)

#remove values with lift < 1
df_results1[df_results1["lift"] > 1]


# ## Product Association

# We are going to focus on the two categories that are frequently purchased together.

# ### Data Preprocessing

# In[ ]:


# Transforming the data into the correct format

basket_product = order_products[(order_products['product_category_name'] == "home_confort") | 
                               (order_products['product_category_name'] == "bed_bath_table")]\
                                    .groupby(['order_id','product_id'])['order_item_id']\
                                    .sum() \
                                    .unstack() \
                                    .reset_index() \
                                    .fillna(0) \
                                    .set_index('order_id')


basket_product_sets = basket_product.applymap(encode_units)
basket_product_sets.head()


# In[ ]:


# remove all orders with less than 2 different categories
basket_product_sets = basket_product_sets[basket_product_sets.sum(axis = 1) > 1]
basket_product_sets.head()


# ### Data Analysis

# In[ ]:


frequent_itemsets_product = apriori(basket_product_sets, min_support=0.005, use_colnames = True)
rules_product = association_rules(frequent_itemsets_product, metric='confidence', min_threshold=0.5)

pd.DataFrame(rules_product).sort_values(["confidence"],ascending = False)


# In[ ]:





# In[ ]:




