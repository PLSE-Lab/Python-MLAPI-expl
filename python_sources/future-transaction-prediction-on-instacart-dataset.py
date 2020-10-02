#!/usr/bin/env python
# coding: utf-8

# In this work, I will try to analyze data for better exploration and different prediction analysis will be applied to compare them.
# 1. **Introduction**
# 
# This competition is basically prepared for prediction future transaction of a sample customer by using transaction past of that customer. Our dataset has information about transaction, reordering, ordering date and information about ordered product.
# 
# 2. **Data Visualization**
# 
# First, we load our libraries and dataset.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn'

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))


# In[ ]:


aisles_data = pd.read_csv("../input/aisles.csv")
departments_data = pd.read_csv("../input/departments.csv")
order_products_train_data = pd.read_csv("../input/order_products__train.csv")
order_products_prior_data = pd.read_csv("../input/order_products__prior.csv")
orders_data = pd.read_csv("../input/orders.csv")
products_data = pd.read_csv("../input/products.csv")


# In order to have an idea of which file contains what we can make use of head() function for each read file:

# In[ ]:


aisles_data.head(10)


# In[ ]:


departments_data.head(10)


# In[ ]:


order_products_train_data.head(10)


# In[ ]:


order_products_prior_data.head(10)


# In[ ]:


orders_data.head(15)


# In[ ]:


products_data.head(10)


# From above tables, we can have some idea about future usage of each file for purpose of train our model. In this dataset, our main goal is predicting future transactions. We need to think about which attributes are convenient to work on it and which are not. Reducing unnecessary attributes decreases time and computation costs. But before going into dimension reduction, Lets check "orders.csv" which keeps all of the data train and test data.

# In[ ]:


cnt_srs = orders_data.eval_set.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Eval set type', fontsize=12)
plt.title('Count of rows in each dataset', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


data = orders_data.groupby("eval_set")["user_id"]
data_cnt = data.size()
data = 0
print(data_cnt)
print("Total customer: ", len(orders_data.groupby("user_id")))


# From above graph, we can observe that we have 3214874 transactions. We have 131209 customers' transaction for train our model and 75000 customers' transaction for testing model.

# In[ ]:


transactions = orders_data.groupby("user_id")["eval_set"].size()
plt.hist(transactions)
plt.title("Number of elements for  different transactions")
plt.xlabel("# of elements")
plt.ylabel("count")
plt.show()
print("min number of element: ", min(transactions))
print("max number of element: ", max(transactions))
transactions = 0


# Above graph indicates that most of the transactions are consist of 4 different products. Maximum element for a transaction in this dataset is 100.

# In[ ]:


sns.countplot(x="order_dow", data = orders_data)
plt.title("Transactions for day of week")
plt.show()


# On the 'x' axis 0 is saturday, 1 is sunday and so on... It is clear that most of the transactions are done in weekend and number of  orders for wednesday is the lowest. Actually, remaining part of this file is not important for us. Order days and hours are irrelevant for predicting next order. Since we do not know and we can not predict which day it will be for a specific customer. We can develep a model for predicting the product for given day, but in here it can be any day and any hour.
# 
# There are some more in this file to observe. Now, we can check how frequently people buy products. 

# In[ ]:


frequent_order_time_period = orders_data.groupby("days_since_prior_order")["user_id"].count()
plt.stem(frequent_order_time_period)
plt.show()


# We can clearly see that people are more likely to buy every week and every month.

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="order_hour_of_day", data=orders_data, color=color[2])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Hour of day', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of order by hour of day", fontsize=15)
plt.show()


# People generally buy their products mostly in day time.

# In[ ]:


grouped_data = orders_data.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
grouped_data = grouped_data.pivot('order_dow', 'order_hour_of_day', 'order_number')

plt.figure(figsize=(12,6))
sns.heatmap(grouped_data)
plt.title("Frequency of Day of week Vs Hour of day", fontsize=15)
plt.show()


# Weekend day times are the most order having periods.
# 
# From now on, let's check ordered products. But first to observe all prior and train transactions, we can concanate order_products_train_data and order_products_prior_data.

# In[ ]:


order_products_prior_data.reordered.sum() / order_products_prior_data.shape[0]


# In the prior orderings file, probability of a product to be reordered is almost 0.59

# In[ ]:


order_products_train_data.reordered.sum() / order_products_train_data.shape[0]


# For train orderings file, products are almost 60% reordered. It is clear that it is more likely to be reordered for a product.

# In[ ]:


grouped_data = order_products_prior_data.groupby("order_id")["reordered"].aggregate("sum").reset_index()
grouped_data["reordered"].loc[grouped_data["reordered"]>1] = 1
grouped_data.reordered.value_counts() / grouped_data.shape[0]


# Above calculation shows that probability of a product being once bought but not reordered is 0.12 for prior orderings.

# In[ ]:


grouped_data = order_products_train_data.groupby("order_id")["reordered"].aggregate("sum").reset_index()
grouped_data["reordered"].loc[grouped_data["reordered"]>1] = 1
grouped_data.reordered.value_counts() / grouped_data.shape[0]


# Same situation for training data, probability is 0.06.

# In[ ]:


grouped_data = order_products_train_data.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()
cnt_srs = grouped_data.add_to_cart_order.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=15)
plt.xlabel('Number of products in the given order', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# Above graph shows number of products in a single order. On average people buy 5 different products.
# 
# After now, we will merge some files for analyzing and visualizing relations better.

# In[ ]:


order_products_train__data = pd.merge(order_products_train_data, products_data, on='product_id', how='left')
order_products_train__data = pd.merge(order_products_train__data, aisles_data, on='aisle_id', how='left')
order_products_train__data = pd.merge(order_products_train__data, departments_data, on='department_id', how='left')
order_products_train__data.head()


# Let's check most ordered products.

# In[ ]:


cnt_srs = order_products_train__data['product_name'].value_counts().reset_index().head(20)
cnt_srs.columns = ['product_name', 'frequency_count']
cnt_srs


# Most ordered products are fruits and diaries.

# In[ ]:


cnt_srs = order_products_train__data['aisle'].value_counts().head(20)
plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[5])
plt.ylabel('Number of Occurrences', fontsize=15)
plt.xlabel('Aisle', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# Above graph also prooves that most of the products sold are from vegetables and fruits. This attribute is highly correlated with number of orderings of a specific product. 
# 
# It might be helpful for analyzing a departments size and reorderings from departments for visualyzing if any relation exist.

# In[ ]:


plt.figure(figsize=(10,10))
temp_series = order_products_train__data['department'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
plt.pie(sizes, labels=labels, 
        autopct='%1.1f%%', startangle=200)
plt.title("Departments distribution", fontsize=15)
plt.show()


# In[ ]:


grouped_data = order_products_train__data.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_data['department'].values, grouped_data['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# Biggest departments are "produce", "diary eggs" and snacks, but relation between size of a department and reordering seems to be unrelated. Because, "breakfast", "alcohol" and "bakery" are small departments but they have also high reordering rates. Above graph is also informative for detecting which departments have higest reordering ratio. For instace, alcohols are not placed in top 20 products but they have high reordering rates.
# 
# Below, aisle wise reordering is analyzed.

# In[ ]:


grouped_data = order_products_train__data.groupby(["department_id", "aisle"])["reordered"].aggregate("mean").reset_index()

fig, ax = plt.subplots(figsize=(12,20))
ax.scatter(grouped_data.reordered.values, grouped_data.department_id.values)
for i, txt in enumerate(grouped_data.aisle.values):
    ax.annotate(txt, (grouped_data.reordered.values[i], grouped_data.department_id.values[i]), rotation=45, ha='center', va='center', color='green')
plt.xlabel('Reorder Ratio')
plt.ylabel('department_id')
plt.title("Reorder ratio of different aisles", fontsize=15)
plt.show()


# Again we can see that beverages, breakfast, fruit and vegetables are most reordered products. People who orders weekly might be buying thiese kind of products.
# 
# Add-to-cart order is another feature given in the dataset. This is an interesting feature because people might act to add mostly ordered products first. In one of above graphs, we see that people generally buy 5 products in an order but in below, we see that there is a peak at 1. This might proove the hypothesis we have. Another peaks in the graph might be because of low number of instances that has more than 50 products. This misleads the ratio since number of samples are much lower.

# In[ ]:


order_products_train__data["add_to_cart_order_mod"] = order_products_train__data["add_to_cart_order"].copy()
order_products_train__data["add_to_cart_order_mod"].loc[order_products_train__data["add_to_cart_order_mod"]>70] = 70
grouped_data = order_products_train__data.groupby(["add_to_cart_order_mod"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_data['add_to_cart_order_mod'].values, grouped_data['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Add to cart order', fontsize=12)
plt.title("Add to cart order - Reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# In the last step, we tried to see whether there exist a relation between day of week and reorderings or not.

# In[ ]:


order_products_train__data = pd.merge(order_products_train__data, orders_data, on='order_id', how='left')
grouped_data = order_products_train__data.groupby(["order_dow"])["reordered"].aggregate("mean").reset_index()
grouped_data.head()


# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(grouped_data['order_dow'].values, grouped_data['reordered'].values, alpha=0.8, color=color[3])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Reorder ratio across day of week", fontsize=15)
plt.xticks(rotation='vertical')
plt.ylim(0.5, 0.7)
plt.show()


# Visual representation of the graph is almost the same with the "DoW and Number of Orders" graph. This is because of there is not exist any significant relation between them.
