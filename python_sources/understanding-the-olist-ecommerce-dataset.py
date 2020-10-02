#!/usr/bin/env python
# coding: utf-8

# # Understanding the Olist Ecommerce Dataset
# The dataset contains informations about orders, products purchased, customer and customer reviews from that orders. Lets take a look at the data to have an initial understanding of what it is and how it works.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns; sns.set(rc={'figure.figsize':(16,9)})
import matplotlib.pyplot as plt

df = pd.read_csv('../input/olist_classified_public_dataset.csv')
df.head()


# ## Initial Data Cleaning
# 
# The columns named  ```votes_```  contains information about how many votes each subcategory has received from each analyts that classified the dataset. The results are summarized in the ```most_voted_``` columns. For the purpose of this initial study I'm going to discard ```votes_```.

# In[ ]:


# select all votes columns
votes_columns = [s for s in df.columns if "votes_" in s]

# drop them
df.drop(votes_columns, axis=1, inplace=True)

# Lets also drop the first two columns
df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)


# In[ ]:


# Lets see the info of our dataset
df.info()


# In[ ]:


# Looks like we may also drop the review_comment_title column, as all values are null
df.drop(['review_comment_title'], axis=1, inplace=True)


# In[ ]:


# We also need to convert datetime features to the correct format
df.order_purchase_timestamp = pd.to_datetime(df.order_purchase_timestamp)
df.order_aproved_at = pd.to_datetime(df.order_aproved_at)
df.order_estimated_delivery_date = pd.to_datetime(df.order_estimated_delivery_date)
df.order_delivered_customer_date = pd.to_datetime(df.order_delivered_customer_date)
df.review_creation_date = pd.to_datetime(df.review_creation_date)
df.review_answer_timestamp = pd.to_datetime(df.review_answer_timestamp)

# Lets see if it looks ok now
df.info()


# ## Seeing Features Through Time
# Now we may want to see how some of those features are related with the purchase date. 

# In[ ]:


# creating a purchase day feature
df['order_purchase_date'] = df.order_purchase_timestamp.dt.date

# creating an aggregation
sales_per_purchase_date = df.groupby('order_purchase_date', as_index=False).order_products_value.sum()
ax = sns.lineplot(x="order_purchase_date", y="order_products_value", data=sales_per_purchase_date)
ax.set_title('Sales per day')


# In[ ]:


# creating a purchase day feature
df['order_purchase_week'] = df.order_purchase_timestamp.dt.to_period('W').astype(str)

# creating an aggregation
sales_per_purchase_month = df.groupby('order_purchase_week', as_index=False).order_products_value.sum()
ax = sns.lineplot(x="order_purchase_week", y="order_products_value", data=sales_per_purchase_month)
ax.set_title('Sales per week')


# In[ ]:


# creating an aggregation
avg_score_per_date = df.groupby('order_purchase_week', as_index=False).review_score.mean()
ax = sns.lineplot(x="order_purchase_week", y="review_score", data=avg_score_per_date)
ax.set_title('Average score per week')


# ## Which product category has lower scores? 

# In[ ]:


# creating an aggregation
avg_score_per_category = df.groupby('product_category_name', as_index=False).agg({'review_score': ['count', 'mean']})
avg_score_per_category.columns = ['product_category_name', 'count', 'mean']

# filtering to show only categories with more than 50 reviews
avg_score_per_category = avg_score_per_category[avg_score_per_category['count'] > 50]
avg_score_per_category = avg_score_per_category.sort_values(by='mean', ascending=False)
avg_score_per_category


# In[ ]:


ax = sns.barplot(x="mean", y="product_category_name", data=avg_score_per_category)
ax.set_title('Categories Review Score')


# ## How do classes relate to the categories and review scores?
# Lets see how classes are distributed for the best and worse category that we just found earlier.

# In[ ]:


eletronicos = df[df.product_category_name == 'eletronicos']['most_voted_class'].value_counts().reset_index()
eletronicos.columns = ['class', 'qty']
eletronicos['percent_qty'] = eletronicos.qty / eletronicos.qty.sum() 
ax = sns.barplot(x="percent_qty", y="class", data=eletronicos)
ax.set_title('Eletronicos Reviews Classes')


# At the eletronicos category, most customers (~85%) are satisfied with their purchase. Around 10% complained about delivery issues and 5% about quality issues.

# In[ ]:


informatica_acessorios = df[df.product_category_name == 'informatica_acessorios']['most_voted_class'].value_counts().reset_index()
informatica_acessorios.columns = ['class', 'qty']
informatica_acessorios['percent_qty'] = informatica_acessorios.qty / informatica_acessorios.qty.sum() 
ax = sns.barplot(x="percent_qty", y="class", data=informatica_acessorios)
ax.set_title('Informatica Acessorios Reviews Classes')


# At the Informatica e Acessorios category, only ~45% of customers are satisfied. Other 35% complained about delivery issues and ~22% of customers found out that the product had quality issues.

# # Now it's your turn!
# This was intended to be just an introduction to the dataset. Did you enjoy it? Have any questions about the data? We are available to help you!

# In[ ]:




