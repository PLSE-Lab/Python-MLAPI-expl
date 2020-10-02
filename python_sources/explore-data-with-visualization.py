#!/usr/bin/env python
# coding: utf-8

# Load Package

# In[2]:



get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# Load Data

# In[3]:


train =pd.read_csv("../input/train.tsv",sep="\t",index_col ='train_id')
test =pd.read_csv("../input/test.tsv",sep="\t",index_col ='test_id')


# In[4]:


train


# In[5]:


print("amount of train data = {} | amount of test data = {}" .format(len(train), len(test)))


# In[6]:


train.head()


# In[7]:


test.head()


# In[8]:


# check unique data
train_columns = train.columns
for i in train_columns:
    print("{} = {} unique_data".format(i,len(train[i].unique())))
print(len(train))


# In[9]:


test_columns = test.columns
for i in test_columns:
    print("{} = {} unique_data".format(i,len(test[i].unique())))
print(len(test))


# Missing Data

# In[10]:


train.info()


# In[11]:


train_columns = train.columns
for i in train_columns:
    print("{0} = {1:.2f}% null_data".format(i,(len(train[train[i].isnull()])/ len(train))*100 ))


# In[12]:


test.info()


# In[13]:


test_columns = test.columns
for i in test_columns:
    print("{0} = {1:.2f}% null_data".format(i,(len(test[test[i].isnull()])/ len(test))*100 ))


# Explore train Data

# In[14]:


train.head(10)


# In[15]:


train.loc[train["brand_name"].isnull(),"brand_name"] ="No_Brand"
test.loc[test["brand_name"].isnull(),"brand_name"] ="No_Brand"


# In[16]:


x_train = train.drop("price", axis =1)
y_train = train["price"]


# In[17]:


train.describe().astype("float16")


# In[18]:


#outlier exist
plt.scatter(train["price"].values,train["price"].index)


# Analyze relationship between item condition and price

# In[19]:


grouped = train.groupby("item_condition_id")["price"].aggregate({"count_of_price":'count'}).reset_index()


# In[20]:


grouped


# In[21]:


count_price = grouped["count_of_price"]


# In[22]:


grouped = train.groupby("item_condition_id")["price"].aggregate({"sum_of_price":'sum'}).reset_index()


# In[23]:


grouped["standard_of_price"] = grouped["sum_of_price"] / count_price
grouped


# In[24]:


grouped["count_of_price"] = count_price
grouped


# In[25]:


figure, (axe1,axe2,axe3) =plt.subplots(nrows =3, ncols =1)
figure.set_size_inches(12,10)
sns.barplot(grouped["item_condition_id"], grouped["sum_of_price"], ax = axe1)
sns.barplot(grouped["item_condition_id"], grouped["standard_of_price"], ax = axe2)
sns.barplot(grouped["item_condition_id"], grouped["count_of_price"], ax = axe3)


# Analyze relationship between brand name and price 

# In[26]:


grouped = train.groupby("brand_name")["price"].aggregate({"sum_of_price":"sum"}).reset_index()
x = train.groupby("brand_name")["price"].aggregate({"count_of_brand":"count"}).reset_index()
grouped["count_of_brand"] = x["count_of_brand"]


# Top10 Sale

# In[27]:


top_10_sales = grouped.sort_values("sum_of_price",ascending=False).head(10)
top_10_sales


# In[28]:


figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(10,8)
sns.barplot(top_10_sales["brand_name"], top_10_sales["sum_of_price"])


# Top 10 volumne of Brand

# In[29]:


top_10_volume_sales = grouped.sort_values("count_of_brand",ascending=False).head(10)
figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(10,8)
sns.barplot(top_10_volume_sales["brand_name"], top_10_volume_sales["count_of_brand"])


# In[30]:


grouped["standard_of_price"] = grouped["sum_of_price"] / grouped["count_of_brand"]
brand_std = grouped.sort_values("standard_of_price",ascending=False)


# In[31]:


Top_20_brand_price = brand_std.head(20)
Top_20_brand_price.head()


# Top 20 High-end Brand

# In[32]:


Top_20_brand_price


# In[33]:


figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(35,10)
sns.barplot(Top_20_brand_price["brand_name"], Top_20_brand_price["standard_of_price"])


# Analyze between product_name and price

# In[34]:


# grouped = 
grouped = train.groupby("name")["price"].aggregate({"sum_of_price":"sum"}).reset_index()


# In[35]:


grouped.head()


# In[36]:


grouped["amount_of_product"]= train.groupby("name")["price"].aggregate({"amount_of_product":"count"}).values
grouped["mean_price"] = grouped["sum_of_price"] / grouped["amount_of_product"]


# Top 10 amount of product

# In[37]:


Top_amount_of_product = grouped.sort_values("amount_of_product",ascending=False).head(10)
Top_amount_of_product


# In[38]:


train[train["name"]=="BUNDLE"]


# Bundle and BUNDLE are different?

# In[39]:


figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(20,5)
sns.barplot(Top_amount_of_product["name"], Top_amount_of_product["amount_of_product"])


# Top 10 sales of product

# In[40]:


Top_sales_of_product = grouped.sort_values("sum_of_price",ascending=False).head(10)
Top_sales_of_product


# In[41]:


figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(20,5)
sns.barplot(Top_sales_of_product["name"], Top_sales_of_product["sum_of_price"])


# Top mean of price

# In[42]:


Top_mean_price = grouped.sort_values("mean_price",ascending=False).head(30)
Top_mean_price


# In[43]:


figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(40,5)
sns.barplot(Top_mean_price["name"], Top_mean_price["mean_price"])


# Analyze between category and price

# In[44]:


train.head()


# In[45]:


grouped = train.groupby("category_name")["price"].aggregate({"sum_of_price":"sum"}).reset_index()
grouped.head()


# In[46]:


grouped["amount_of_category"] = train.groupby("category_name").size().values
grouped.head()


# In[47]:


grouped["standard_price"] = grouped["sum_of_price"] / grouped["amount_of_category"].astype("float16")
grouped.head()


# Top 10 sales_category 

# In[48]:


Top10_sales= grouped.sort_values("sum_of_price",ascending=False).head(10)
Top10_sales


# In[49]:


figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(50,8)
sns.barplot(Top10_sales["category_name"],Top10_sales["sum_of_price"])


# Top 10 amount of sales

# In[50]:


Top10_amount_category= grouped.sort_values("amount_of_category",ascending=False).head(10)
Top10_amount_category


# In[51]:


sample  = train[(train["category_name"] == "Women/Athletic Apparel/Pants, Tights, Leggings") &(train["brand_name"] == "LuLaRoe")
      &(train["item_condition_id"]==1)&(train["shipping"]==1)]
sample.sort_values("price", ascending=False)


# In[52]:


figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(50,8)
sns.barplot(Top10_amount_category["category_name"],Top10_amount_category["amount_of_category"])


# Top 10 mean of price

# In[53]:


Top10_mean_price= grouped.sort_values("standard_price",ascending=False).head(10)
Top10_mean_price


# In[54]:


figure, axe = plt.subplots(nrows =1, ncols =1)
figure.set_size_inches(50,8)
sns.barplot(Top10_mean_price["category_name"],Top10_mean_price["standard_price"])


# In[55]:


train["category_name"] ="Home/Home Appliances/Air Conditioners"
train.head()


# In[56]:


train.loc[train["category_name"].isnull(),"category_name"] ="No_category"
test.loc[test["category_name"].isnull(),"category_name"] ="No_category"


# In[57]:


train[train["category_name"] =='Home/Home Appliances/Air Conditioners']


# Analyze between shipping and price

# In[58]:


grouped = train.groupby("shipping")["price"].aggregate({"sum_of_price":"sum"}).reset_index()


# In[59]:


x = train.groupby("shipping")["price"].aggregate({"amount_of_price":"count"}).reset_index()


# In[60]:


grouped["amount_of_price"] = x["amount_of_price"]
grouped["mean_of_price"] = grouped["sum_of_price"] / grouped["amount_of_price"]
grouped


# In[61]:


figure, (axe1,axe2,axe3) = plt.subplots(nrows =1, ncols =3)
figure.set_size_inches(14,4)
sns.barplot(grouped["shipping"],grouped["amount_of_price"],ax = axe1)
sns.barplot(grouped["shipping"],grouped["sum_of_price"], ax = axe2)
sns.barplot(grouped["shipping"],grouped["mean_of_price"],ax = axe3)

