#!/usr/bin/env python
# coding: utf-8

# # Instacart Market Basket Analysis & Recommendation Algorythm
# 
# ### Hi, my name is Hernando Andrianto Willy Ren and I would like to perform data exploration , and create recommendation algorythm for Instacart 
# 
# <img src="http://whatpixel.com/images/2016/01/instacart-logo-redesign-rebrand-2016.jpg" style="float: right; height: 120px">
# 
# 
# Instacart is an American company that operates as a same-day grocery delivery service. Customers select groceries through a web application from various retailers and delivered by a personal shopper.
# 
# 
# https://www.kaggle.com/c/instacart-market-basket-analysis
# 
# https://www.instacart.com/
# 
# https://en.wikipedia.org/wiki/Instacart
# 
# 
# **GOAL:**
# 
# The goal is to predict which products will be in a user's next order. The dataset is anonymized and contains a sample of over 3 million grocery orders from more than 200,000 Instacart users.
#  
#  I will conduct it in three steps:
#  1. EDA
#  2. Clustering Analysis
#  3. Recommendation System algorythm *not yet*
#  
#  This kernel actually just for my private learning how to code, so i sometimes copy and trying to understand how codes works. credit to  
#  https://www.kaggle.com/serigne/instacart-simple-data-exploration 
#  https://www.kaggle.com/asindico/customer-segments-with-pca 
#  https://www.kaggle.com/sudalairajkumar

# In[ ]:


#Import libraries

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ord_prod_train_df = pd.read_csv("../input/order_products__train.csv")
ord_prod_prior_df = pd.read_csv("../input/order_products__prior.csv")
orders_df = pd.read_csv("../input/orders.csv")
products_df = pd.read_csv("../input/products.csv")
dept_df = pd.read_csv("../input/departments.csv")
aisles_df = pd.read_csv("../input/aisles.csv")


# # Knowing Our Files
# ### Orders_df
# This file tells to which set (prior, train, test) an order belongs. You are predicting reordered items only for the test set orders. 'order_dow' is the day of week.

# In[ ]:


orders_df.head()


# In[ ]:


orders_in_1_order_id = orders_df[(orders_df.order_id <10)].sort_values(by =['order_id'])
orders_in_1_order_id  
#so each order_id is unique and


# In[ ]:


orders_df.info()


# Orders_df includes 3,4 millions orders that instacart has, and it has cleaned data (no NaN), so probably we wouldnt need to conduc imputation for missing data. 

# In[ ]:


orders_df.hist(bins=12,figsize = (10,10))


# based on all order, we can find initial information:\
# 1. most repeat customer purchase around 7 days or 30 days after their last purchase
# 2. from 7 days of the week, the most active day most is day 0 & 1, most probably it is Saturday and Sunday
# 3. Based on order hour, the popular hours is between 10-15 pm. so presumable its around weekend afternoon to purchase same day delivery. 
# but we will need to take a look more on more precise plots on these informations.
# 

# In[ ]:


cnt_eval = orders_df.eval_set.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_eval.index, cnt_eval.values, alpha=0.8, color='blue')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Eval set type', fontsize=12)
plt.title('Count of rows in each dataset', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


cnt_eval


# we can see that the prior purchase is 3,2 millions orders, train set with 130,000 orders, and

# ## Orders Product Prior & Train
# These files specify which products were purchased in each order. order_products__prior.csv contains previous order contents for all customers. 'reordered' indicates that the customer has a previous order that contains the product. Note that some orders will have no reordered items. You may predict an explicit 'None' value for orders with no reordered items. See the evaluation page for full details.

# In[ ]:


ord_prod_prior_df.head()


# In[ ]:


items_in_1_order_id = ord_prod_prior_df[(ord_prod_prior_df.order_id <3)].sort_values(by =['order_id'])
items_in_1_order_id


# In[ ]:


ord_prod_train_df.head()


# In[ ]:


ord_prod_prior_df.info()


# In[ ]:


ord_prod_train_df.info()


# # Products,  Department & Aisles File

# In[ ]:


products_df.head()


# In[ ]:


products_df.info()


# dept_df.head()

# In[ ]:


dept_df.info()


# In[ ]:


aisles_df.head()


# In[ ]:


aisles_df.info()


# # File Schema
# Based on the files type exploration, we can understand more the datas using this schema
# 
# 
# <img src="https://kaggle2.blob.core.windows.net/forum-message-attachments/183176/6539/instacartFiles.png">
# 
# https://www.kaggle.com/frednavruzov/instacart-exploratory-data-analysis/notebook
# 

# ## Merge products, dept and aisles df into all_products dataset 

# In[ ]:


all_products= pd.merge(left=products_df, right= dept_df, left_on='department_id',right_on='department_id', how = 'left')
all_products= pd.merge(left=all_products, right= aisles_df, left_on='aisle_id', right_on='aisle_id', how = 'left')
all_products.head()


# In[ ]:


# in total we have 49678 types of product,20 department and 133 different aisles.
all_products.info()


# ## Merge Train & Prior Orders with all products information

# In[ ]:


#aggregate training dataset transactions
train_transactions= pd.merge(left=ord_prod_train_df, right= all_products, left_on='product_id',right_on='product_id', how = 'left')
prior_transactions= pd.merge(left=ord_prod_prior_df, right= all_products, left_on='product_id',right_on='product_id', how = 'left')


# In[ ]:


train_transactions.info()


# In[ ]:


prior_transactions.info()


# In[ ]:


orders_df.info()


# # Concacenate Train & Prior Dataset (For EDA Purpose)

# In[ ]:


df_all = pd.concat([train_transactions, prior_transactions], axis=0)

print("The order_products_all size is : ", df_all.shape)


# In[ ]:


df_all.head(20)


# In[ ]:


orders_df[(orders_df.order_id == 1)]


# Here we can see that actually in order_df, it contains all orders ID  , while actually based on df_all , we can see that 1 order ID  can contain many item orders (example: order id 1, executed by user id 112108, was actually ordering 8 items. So actually in this dataset where have 3 million orders, presumably it can contain so much more orders. In this case 30 million items order considering the mean item was 10 items per order

# In[ ]:


orders_df.shape


# In[ ]:


df_all.shape


# ## Missing Data

# In[ ]:


total = df_all.isnull().sum().sort_values(ascending=False)
percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])
print (missing_data)


# It's cleaned data with zero Null values

# ### Now we are down from 6 files into 4 files:
# 1. all_products = Aggregated total products available in instacart info
# 2. train_transactions = All transactions occur in train dataset orders
# 3. prior_transactions= All transactions occur in prior orders
# 4. orders_df = All Orders from prior, train,and test
# 
# # Questions I want to Answer 
# 1. How many is unique customer of Instacart
# 2. How many number or product purchase average per customer?
# 3. Which segment is the item that people likes to. buy?
# 4. Which aisles and department generate more orders
# 5. Whats the most popular ordering hours and day
# 6. Can we segment by customers by product order?
# 
# 1. UNIVARIATE ANALYSIS

# ## 1. How many is unique customer of Instacart

# In[ ]:


def get_unique_count(x):
    return len(np.unique(x))

cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)
cnt_srs


# There's 206209 Unique user in test, 131209 unique 75000 unique users.

# ## 2. Number of products that people usually order :

# In[ ]:


# Check the number of unique orders and unique products
orders_Unique = len(set(orders_df.order_id))
products_Unique = len(set(all_products.product_id))
print("There are %s orders for %s products" %(orders_Unique, products_Unique))


# In[ ]:


### aggregate by count of product for each order
no_products_bought = df_all.groupby('order_id')['product_id'].count()
no_products_bought.describe()


# In[ ]:


print ( "Averagely , users order ",(str(int(no_products_bought.mean()))) , " items per order")
print ("In median , users order ", (str(int(no_products_bought.median()))), " items per order")
print ("Minimum users order ",(str(int(no_products_bought.min()))), " items per order")
print ("Maximum users order ",(str(int(no_products_bought.max()))), " items per order")


# In[ ]:


# creating dataframe for number of products bought per order by order id
no_products_bought = pd.DataFrame(no_products_bought )
no_products_bought['order_id2'] = no_products_bought.index
no_products_bought = no_products_bought.rename(columns={'product_id': 'number_of_products_ordered'})
npb = no_products_bought.groupby(['number_of_products_ordered']).count()
npb = npb.rename(columns={'order_id2': 'number_of_users'})
npb['no_of_products_ordered'] =  npb.index
npb.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "sns.set_style('whitegrid')\nf, ax = plt.subplots(figsize=(15, 12))\nplt.xticks(rotation='vertical')\nsns.barplot(x='no_of_products_ordered', y='number_of_users',  data= npb, color='grey')\nplt.xlim(0,60)\nplt.ylabel('Number of Orders', fontsize=13)\nplt.xlabel('Number of products added in order', fontsize=13)\nplt.show()")


# Averagely, people buy 10 items per order
# In Median, people buy 8 items per order
# Minimum, people buy 1 items per order
# Maximum, people buy 145 items per order
# 
# mostly people order ranging between 4-7 items per order,  with mode at 5 items per order
# 

# 
# # 3. Most Active Order Time

# In[ ]:


get_ipython().run_cell_magic('time', '', 'plt.figure(figsize=(12,8))\nsns.countplot(x="order_dow", data=orders_df, color=\'grey\')\n\nplt.ylabel(\'Count\', fontsize=12)\nplt.xlabel(\'Day of week\', fontsize=12)\nplt.xticks( rotation=\'vertical\')\nplt.title("Frequency of order by week day", fontsize=15)\n\nplt.show()')


# People usually order at days 0 and 1 (anonimyzed days and probably the week end)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'plt.figure(figsize=(12,8))\nsns.countplot(x="order_hour_of_day", data=orders_df, color=\'grey\')\n\nplt.ylabel(\'Count\', fontsize=12)\nplt.xlabel(\'Hour\', fontsize=12)\nplt.xticks(rotation=\'vertical\')\nplt.title("Frequency of time of the day", fontsize=15)\n\nplt.show()')


# most purchase rangin between 10AM-3PM in the afternoon, with peak at 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'plt.figure(figsize=(12,8))\nsns.countplot(x="days_since_prior_order", data=orders_df, color=\'grey\')\n\nplt.ylabel(\'Count\', fontsize=12)\nplt.xlabel(\'Days Since Prior Order\', fontsize=12)\nplt.xticks(rotation=\'vertical\')\nplt.title("Days Since Prior Order", fontsize=15)\n\nplt.show()')


# We can see mostly customers order exactly 7 days or 30 days,it might infer that the useres purchase groceries weekly and monthly. And Its understandable that if people most likely purhase on weekends, they will order again on weekends or every month maybe because they are worker mostly and only have weekend offs.
# 
# we can assume that the most active time of order will be 10AM-3PM on weekends. with weekly/monthly frequency.

# In[ ]:


# validify with heatmap (1) all record (2) most active time : day 0 & 1,


# In[ ]:


grouped_df = orders_df.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
grouped_df.head()


# In[ ]:


grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')
grouped_df.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(grouped_df, cmap="YlGnBu")
plt.title("Frequency of Day of week Vs Hour of day")
plt.show()


# In[ ]:


active_subset_7 = orders_df[(orders_df.days_since_prior_order == 7.0)]
active_subset_30 = orders_df[(orders_df.days_since_prior_order == 30.0)]
active_subset = pd.concat([active_subset_7,active_subset_30])


# In[ ]:


grouped2_df = active_subset.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
grouped2_df = grouped2_df.pivot('order_dow', 'order_hour_of_day', 'order_number')
grouped2_df.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(grouped2_df, cmap="YlGnBu")
plt.title("Frequency of Day of week Vs Hour of day")
plt.show()


# After we filter it more clearly, we can see that actually the most active days for weekly and monthly users do their purchases around 1PM-3 PM on day 0 (presumably saturday) and 9-10 AM on day 1 (presumably on Sunday).

# # 4. MOST ORDER DEPARTMENT & AISLES & ITEMS

# In[ ]:


# Popular Departments
popular_departments = df_all['department'].value_counts().reset_index()
popular_departments.columns = ['department', 'frequency_count']
popular_departments.head(20)


# In[ ]:


df_all.head()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="department" ,data=df_all, order = df_all["department"].value_counts().index[:12])

plt.ylabel('Frequency Count', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Popular Department", fontsize=15)

plt.show()


# In[ ]:


# Popular Departments
popular_aisles = df_all['aisle'].value_counts().reset_index()
popular_aisles.columns = ['aisle', 'frequency_count']
popular_aisles.head(20)


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="aisle" ,data=df_all, order = df_all["aisle"].value_counts().index[:12])

plt.ylabel('Frequency Count', fontsize=12)
plt.xlabel('Aisle', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Popular Aisles", fontsize=15)

plt.show()


# In[ ]:


#popular Items
popular_items = df_all['product_name'].value_counts().reset_index()
popular_items.columns = ['product_name', 'frequency_count']
popular_items.head(20)


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="aisle" ,data=df_all, order = df_all["aisle"].value_counts().index[:12])

plt.ylabel('Frequency Count', fontsize=12)
plt.xlabel('Product Name', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Popular Product", fontsize=15)

plt.show()


# ## Best Selling Aisles in each Department 

# In[ ]:


grouped = df_all.groupby(["department", "aisle"])["order_id"].aggregate({'Total_orders': 'count'}).reset_index()
grouped.sort_values(by='Total_orders', ascending=False, inplace=True)
fig, axes = plt.subplots(7,3, figsize=(20,45), gridspec_kw =  dict(hspace=1.4))
for (aisle, group), ax in zip(grouped.groupby(["department"]), axes.flatten()):
    g = sns.barplot(group.aisle, group.Total_orders , ax=ax)
    ax.set(xlabel = "Aisles", ylabel=" Number of Orders")
    g.set_xticklabels(labels = group.aisle,rotation=90, fontsize=12)
    ax.set_title(aisle, fontsize=15)


# # Clustering Analysis by Aisles Segment

# 

# In[ ]:


len(df_all['aisle'].unique())


# In[ ]:


df_all['aisle'].value_counts()[0:10]


# FOR PCA, WE will conduct clustering analysis in term of Aisles purchases , then compared it by aisles and departments.
# 
# 

# In[ ]:


a_orders= pd.merge(left=df_all, right= orders_df, left_on='order_id',right_on='order_id', how = 'left')


# In[ ]:


df = a_orders.drop(['eval_set','order_number','order_dow','order_hour_of_day','days_since_prior_order'], 1)


# In[ ]:


df.head()


# In[ ]:


all_purchases = pd.crosstab(df['user_id'], df['aisle'])
all_purchases.head()


# In[ ]:


all_purchases.shape


# 206209 unique customers, 134 aisles

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=8)
pca.fit(all_purchases)
pca_samples = pca.transform(all_purchases)


# In[ ]:


ps = pd.DataFrame(pca_samples)
ps.head()


# In[ ]:


ps.shape


# In[ ]:


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
tocluster = pd.DataFrame(ps[[4,1]])  ## PC1  and #PC4 samples
print (tocluster.shape)
print (tocluster.head())

fig = plt.figure(figsize=(8,8))
plt.plot(tocluster[4], tocluster[1], 'o', markersize=2, color='blue', alpha=0.5, label='class1')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

clusterer = KMeans(n_clusters=4,random_state=42).fit(tocluster)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)
print(centers)


# In[ ]:


print (c_preds[0:100])


# In[ ]:


import matplotlib
fig = plt.figure(figsize=(8,8))
colors = ['orange','blue','purple','green']
colored = [colors[k] for k in c_preds]
print (colored[0:10])
plt.scatter(tocluster[4],tocluster[1],  color = colored)
for ci,c in enumerate(centers):
    plt.plot(c[0], c[1], 'o', markersize=8, color='red', alpha=0.9, label=''+str(ci))

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()


# In[ ]:


all_purchases_cluster = all_purchases.copy()
all_purchases_cluster['cluster'] = c_preds

all_purchases_cluster.head(10)


# In[ ]:


print (all_purchases_cluster.shape)
f,arr = plt.subplots(2,2,sharex=True,figsize=(15,15))

c1_count = len(all_purchases_cluster[all_purchases_cluster['cluster']==0])

c0 = all_purchases_cluster[all_purchases_cluster['cluster']==0].drop('cluster',axis=1).mean()
arr[0,0].bar(range(len(all_purchases_cluster.drop('cluster',axis=1).columns)),c0)

c1 = all_purchases_cluster[all_purchases_cluster['cluster']==1].drop('cluster',axis=1).mean()
arr[0,1].bar(range(len(all_purchases_cluster.drop('cluster',axis=1).columns)),c1)

c2 = all_purchases_cluster[all_purchases_cluster['cluster']==2].drop('cluster',axis=1).mean()
arr[1,0].bar(range(len(all_purchases_cluster.drop('cluster',axis=1).columns)),c2)

c3 = all_purchases_cluster[all_purchases_cluster['cluster']==3].drop('cluster',axis=1).mean()
arr[1,1].bar(range(len(all_purchases_cluster.drop('cluster',axis=1).columns)),c3)
plt.show()


# In[ ]:


c0.sort_values(ascending=False)[0:10]


# In[ ]:


c1.sort_values(ascending=False)[0:10]


# In[ ]:


c2.sort_values(ascending=False)[0:10]


# In[ ]:


c3.sort_values(ascending=False)[0:10]


# ##
# Special Thanks to 
# https://www.kaggle.com/serigne/instacart-simple-data-exploration
# https://www.kaggle.com/asindico/customer-segments-with-pca
# https://www.kaggle.com/sudalairajkumar
# 
# this is just for purpose of study so i can track down and understand codes to EDA, clustering analysis and maybe recommender system soon 
