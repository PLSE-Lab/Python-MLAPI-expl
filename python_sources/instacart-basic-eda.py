#!/usr/bin/env python
# coding: utf-8

# **InstaCart Exploratory Data Analysis**
# 
# In this notebook my attempt will be to explore the dataset as much as possible and gain some valuable insights from it so that I can go ahead with my next step of feature generation, feature selection and data preprocessing.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting package
import seaborn as sns #plotting package
import warnings #to supress the warnings generated
warnings.filterwarnings('ignore')

import gc

color = sns.color_palette()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


PATH = '/kaggle/input/instacart-market-basket-analysis/'


# In[ ]:


#Reading all the data.
aisles = pd.read_csv(PATH + 'aisles.csv')
products = pd.read_csv(PATH + 'products.csv')
department = pd.read_csv(PATH + 'departments.csv')
orders = pd.read_csv(PATH + 'orders.csv')
train = pd.read_csv(PATH + 'order_products__train.csv')
#test = pd.read_csv(PATH + 'order_products__test.csv')
prior = pd.read_csv(PATH + 'order_products__prior.csv')


# So how we will proceed with this problem is we will first explore each files and see what data it holds. Make some assumptions on the data, see the distribution of the data and gain some valuable insights from it. It will later help us when we will combine the datasets into one.
# 
# So first we will look into the aisles dataset.

# In[ ]:


#Aisles data
aisles.head()


# The aisles dataset gives us information about the number of aisles present in the InstaCart store.
# 
# Lets check how many aisles are there in this stores.

# In[ ]:


#number of aisles in the store
print('Shape of the dataset :',aisles.shape)
print('Number of unique aisles present: ', aisles.aisle_id.unique().shape[0])


# As we can see there are 134 aisles present in this store. This was all the information we could extract from this dataset.
# 
# Lets now look into the departments dataset.

# In[ ]:


#departments data
department.head()


# In[ ]:


#number of departments present in the store
print('Shape of the dataset :',department.shape)
print('Number of unique departments present: ', department.department_id.unique().shape[0])


# This dataset is similar to the aisles dataset which tell us the number of departments there are in the store with their names and unique ID. We can see that there are 21 departments.

# In[ ]:


#products dataset
products.head()


# This dataset tell us about the products sold by the store along with which department and aisle they belong to. We can merge the above two datasets to get the aisle and department names.
# 
# First we will check how many products this store offers to their customers.

# In[ ]:


#number of products offered by the store.
print('Shape of the dataset :',products.shape)
print('Number of products :', products.product_id.unique().shape[0])


# This store has 49688 products to offer their customers which is huge. In this dataset we have the name of the products and their aisle and department ids which do not give us any information on their own.</br>
# We will have to make this dataset complete by merging the aisle and department information we have to this products dataframe. Then we can get some meaningful insights from the data.
# 
# Lets merge them!

# In[ ]:


#merging table aisles and department with products.
products_merged = pd.merge(products, aisles, on='aisle_id', how='inner')
products_merged = pd.merge(products_merged, department, on='department_id', how='inner')

#shape of the new DataFrame.
products_merged.shape


# We have with us now a new DataFrame which holds information from the aisles and department table. The shape of the file tells us that we have not loose any of the information present in the products table.
# 
# Let's explore this new table more.

# In[ ]:


#explore
products_merged.head()


# The information has been added successfully but the order of the table as been distorted. Let us sort the table by the product_id as it was in the products table.

# In[ ]:


#sorting by product_id (Optional)
products_merged.sort_values(by='product_id', inplace=True)
products_merged.reset_index(drop=True, inplace=True)


# In[ ]:


products_merged.head()


# Looking pretty now!. Now we can extract some valuable information about the structure of the store, such as whihc department contains the most product or which aisle contains the most products.
# 
# Lets find out!

# In[ ]:


#department with most products.
products_merged.department.value_counts()


# This store has a lot of products in the personal_care, snacks, pantry, bevarages and frozen departments. There is one department called missing which seems odd. What kind of department is a missing department and what kind of products are stored in this department?<br>
# We will find that out later.
# 
# Lets do the same with the aisle.

# In[ ]:


#aisles with most products.
products_merged.aisle.value_counts().head()


# There are 134 aisles in the store and displaying all them will not help us, so I have displayed the top 5 aisles hoding most of the products in the store.
# 
# As we can see that the missing aisle has the most products, the same amount as in the missing department. So this missing value in the aisle and the department may actually be a NaN value i.e the information for these products are missing from the dataset.
# 
# But we know that the aisle and department containing missing as their name contains a unique aisle and department ID, so it can be so that this missing value is actually a separate aisle and department in the store. We can find that out by seeing the products these missing aisle and department are holding.

# In[ ]:


#getting the missing aisle and department subset.
missing = products_merged.loc[(products_merged.department == 'missing') & (products_merged.aisle == 'missing')]
missing.shape


# Vola!!! Missing aisle == Missing department. (as the shape is equal i.e 1258)

# In[ ]:


#head()
missing.head(10)


# The products are all mixed up. They clearly do not belong to one particular department. At least I cannot see it. If you guys can infer any relation between these products please share it with me in the comments.
# 
# As for now they will stay as missing. We have missing aisle and missing department holdings unrelated products. That's sad! Why on earth they have same ids!
# That's all we can incur from the products datasets. Lets now look in to the order_products_*.csv files

# In[ ]:


#orders data
print("Shape of the dataset:", orders.shape)
orders.head()


# The orders dataset is about the orders placed by the users of the InstaCart. The eval_set columns tell us to which of the given dataset a order belongs to i.e prior, eval or train.
# 
# The dataset contains many other columns such as order_dow which give us the day of the week the order was placed on, order_hour_of_day which tell us the hour of the day and days_since_prior_order which tell us the days since previous order. Rest of the columns are self explanatory.
# 
# Let's gain some insights from this dataset.

# In[ ]:


#number of observatio,ns in each of the datasets (eval, train, prior)
plt.figure(figsize=(8, 5))
print('Number of observation in each set: \n{}'.format(orders.eval_set.value_counts()))
sns.countplot(x='eval_set', data=orders, color=color[0]);


# In[ ]:


#unique users of the instacart store.
print('Unique users of the store:', orders.user_id.unique().shape[0])


# There are 206209 users which uses the services of the InstaCart. Let's check how much of these users are from the train, eval and prior set.

# In[ ]:


def unique_users_from_dataset(data):
    return data.unique().shape[0]

orders.groupby(by='eval_set')['user_id'].aggregate(unique_users_from_dataset)


# In the above code I wanted to see the distribution of users in each of the eval_set so I created a function which returns the length of the unique elements present in the list.</br>
# 
# To get the data, first I grouped the dataset by the eval_set and then extracted only the user_id column from the grouped data. I called an aggregate function of pandas which takes a funtion as an argument and passed my earlier created function to get the number of only unique users from the grouped data.
# 
# So we can see that out of the 206209 users 75000 of them are prsent in the test, 131209 in the train and all the 206209 in the prior dataset. This means that we have the prior ordering information of all the users in the dataset and none of the users are new to the InstaCart store. Everyone has a purchase history and that is great.

# In[ ]:


#day of the week the orders were placed
plt.figure(figsize=(8, 5))
print('Frequency of orders on each day of the week: \n{}'.format(orders.order_dow.value_counts()))
sns.countplot('order_dow', data=orders, color=color[1])
plt.title('Frequency of orders on each day of the week')
plt.xlabel('DOW')
plt.ylabel('frequency of orders')


# We can see that the maxinmum number of orders are received on 0 and 1 day of the week. Assuming 0 to be Saturday and 1 to be Sunday and rest of the numbers to be Monday-Friday respectively, it only make sense that most of the orders are recieved on weekends.
# 
# The frequency decreases in the middle days of the week but again increases as we approach the weekend. A good point to note here!

# In[ ]:


#hour of the day the orders were placed.
plt.figure(figsize=(8, 5))
print('Frequency of orders by hours in a day: \n{}'.format(orders.order_hour_of_day.value_counts().head())) # top 5 hours only
sns.countplot('order_hour_of_day', data=orders, color=color[2])
plt.title('Frequency of orders by hours in a day')
plt.xlabel('hours')
plt.ylabel('frequency of orders')


# We can see that the users are very active from 9 to 17 hour of the day. Most of the orders are coming during that time range. There are some people who are ordering at 1-4 also, what kind of people are these who order so early in the morning?
# 
# The days_since_prior order contains NaN values.

# In[ ]:


#number of NaN values in days_since_prior.
orders.days_since_prior_order.isnull().sum()


# 206209 observations for this column is missing which is a lot. Let us find out more about this column and see whether the missing values are following any patern or they are just missing randomly.

# In[ ]:


#extracting observations where this column has missing value.
missing = orders.loc[orders.days_since_prior_order.isnull()]
missing.head()


# By just exploring the top 5 observation I can see one pattern for the misssingness in the days_since_prior_order. <br>
# As we can see that the order_number for the top 5 observations is 1 i.e first order placed by user and it make sense that for the first order we will not have any value in days_since_prior_order.
# 
# Why didn't I taught of it earlier before creating a new dataframe. Damn!
# 
# Let's check whether our assumption holds true for the rest of the missingness in the data. For this we will have to check the values present in the number_days columns and if all the observations are 1 then we can conclude our assumption.

# In[ ]:


#checking assumption
missing.order_number.value_counts()


# So the missingness do follow a pattern and now we have proper information to deal with it. We will substitute the missing values in this column with 0. Let's do this.

# In[ ]:


#imputing the missing values in days_since_prior_order with 0.
orders.days_since_prior_order.fillna(0, inplace=True)


# In[ ]:


#distribution of the days_since_prior_order column.
plt.figure(figsize=(10, 7))
sns.countplot('days_since_prior_order', data=orders, color=color[3])
plt.title('Number of days since previous order')
plt.xlabel('days')
plt.ylabel('frequency of days')
plt.xticks(rotation='vertical');


# We can see the distribution of the column and tell that the days > 30 are all capped at 30. That is why it has the maximum frequency.
# We can also see a peak at every 7th day (7, 14, 21, 28) which can mean that lot of the products are ordered by the users on a weekly basis.

# Now we know the hour of the day and the day of the week when most of the orders are placed, let us combine and see on which day and hour the users are most active.

# In[ ]:


# creating a df which contains these two columns
grouped_df = orders.groupby(['order_dow', 'order_hour_of_day'])['order_id'].aggregate("count").reset_index()
grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_id')
plt.figure(figsize=(15, 6))
sns.heatmap(grouped_df, cmap='YlGn')
plt.xlabel('Hour of day')
plt.ylabel('Day of Week')
plt.title('Frequency of hour of day vs Day of week');


# We can see the most of the orders are placed on the 0th day between 13-15 hour and 1st day between 9-11 hour. Also we can see that most of the orders on any given day are placed in between 9-18th hour.
# 
# We are all done with this dataset. There is nothing more to extract from this one. Let us move on to the next two remaining datasets viz order_products_prior and order_products_train. We will look into them togehter as ther both are similar.

# In[ ]:


#exploring the mentioned dataset
prior.head()


# In[ ]:


train.head()


# In[ ]:


#Lets first see the number of unique orders present in both the datasets.
print('No. unique orders present in prior: {}'.format(prior.order_id.unique().shape[0]))
print('No. unique orders present in train: {}'.format(train.order_id.unique().shape[0]))


# We got the unique orders present in both the datasets.

# In[ ]:


# Top 5 products ordered.
df = prior.product_id.value_counts().head()
df


# These are the top 5 products which are ordered the most by the InstaCart users. We have the product ids on the left and the count of the products in the right. From this we cannot come to know the name and type of the products but we can merge the product_merged dataset which we had created earlier which contained all the products related information.
# 
# Lets merge to complete this dataset!

# In[ ]:


#merging product information
prior_merged = pd.merge(prior, products_merged, on='product_id')
prior.shape, prior_merged.shape
gc.collect()


# In[ ]:


#now let's get the names top 5 ordered products.
prior_merged.product_name.value_counts().head()


# These are the top 5 products ordered from the InstaCart store. All of them are fruits and many of them are organic fruits. People now a days are into this organic stuff and Instacart seem to have a lot of organic stuff to offer.
# Lets get the top 20 observations.

# In[ ]:


#top 20
prior_merged.product_name.value_counts().head(20)


# I have a feeling these products belongs to a single or similar aisle/deparrtment as most of them are organic fruits/vegetables.

# In[ ]:


#Which aisle most of the ordered products belongs to
plt.figure(figsize=(9,5))
prior_merged.aisle.value_counts().head(20).plot.bar(color=color[4]);


# As we can see that most of the products ordered are from the fresh fruits and fresh vegetables aisle. We can conclude that fruits and vegetables are the most sold items in the InstaCart Store. 

# Now let us look into the reordered items and see the top 5 items that are reordered the most and the aisles of these items.

# In[ ]:


#subsetting the data to get reordered items info.
reordered = prior_merged.loc[prior_merged.reordered == 1]
reordered.head()


# These are the products which were reordered by the users of the InstaCart. Lets get the top 5 most reordered products.

# In[ ]:


#Top 5 most reordered.
reordered.product_name.value_counts().head(5)


# This list is same as the list above i.e Top 5 most items ordered by the users is same as Top 5 most items reordered by the users.
# 
# Let us now check the reorder ratio for each department. This will give us an idea on how much products in a department are reordered by the users, as our main aim is to predict the reorder of a product so this may prove to be a good feature later.

# In[ ]:


#Let us first check the distribution of the departments.
distribution = prior_merged.department.value_counts()
labels = (np.array(distribution.index))
sizes = (np.array((distribution / prior_merged.shape[0])*100))

#Dataframe to old these values
dept_dist = pd.DataFrame()
dept_dist['Department'] = labels
dept_dist['distribution'] = sizes
dept_dist.head(21)


# As we can see that the maximum number of products belongs to the produce departmen (29.22%), followed by dairy egss and snacks.<br>
# Now let us check the reorder ration of each department.

# In[ ]:


#Reordered ratio
reordered_ratio = prior_merged.groupby(by='department')['reordered'].agg(['mean']).reset_index()
reordered_ratio.sort_values(by='mean', ascending=False).head(21)


# As we can see that the dairy eggs reorder ratio is the highest i.e 66.9 % and personal care items has the lowest reordered ratio i.e 32%. Lets plot a graph for better visualization.

# In[ ]:


#plotting a graph.
plt.figure(figsize=(12,8))
sns.pointplot(reordered_ratio['department'].values, reordered_ratio['mean'].values, alpha=0.8, color=color[5])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# 1. In the similar way lets us now check the distribution of the aisles and the reorder ratio of it.

# In[ ]:


#Let us first check the distribution of the aisles.
distribution = prior_merged.aisle.value_counts()
labels = (np.array(distribution.index))
sizes = (np.array((distribution / prior_merged.shape[0])*100))

#Dataframe to old these values
aisles_dist = pd.DataFrame()
aisles_dist['Aisle'] = labels
aisles_dist['distribution'] = sizes
aisles_dist.head(10) #showing only top 10 as there are 134 aisles.


# In[ ]:


#Reordered ratio for aisles
reordered_ratio = prior_merged.groupby(by='aisle')['reordered'].agg(['mean']).reset_index()
top = reordered_ratio.sort_values(by='mean', ascending=False).head(10) # Top 10
bottom = reordered_ratio.sort_values(by='mean').head(10) # Bottom 10

#plotting a graph.
plt.figure(figsize=(15,8))
sns.pointplot(top['aisle'].values, top['mean'].values, alpha=0.8, color=color[6])
#sns.pointplot(bottom['aisle'].values, bottom['mean'].values, alpha=0.8, color=color[6])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Aisle', fontsize=12)
plt.title("Aisle wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# We got the top 10 aisle with the highest reorder ratio. That was all the information I can get from this dataset. Let us now move forward with our exploration.
# 
# Lets merge the information contained in the orders dataset with the prior and train dataset. As we know that they both have a common id which is order_id and we will use this key to merge both the tables.

# In[ ]:


#merging the products information with train.
train_merged = pd.merge(train, products, on='product_id', how='inner')
print(train.shape, train_merged.shape)


# In[ ]:


#merging the orders info with train and prior
train_merged = pd.merge(train_merged, orders, on='order_id', how='left')
print(train.shape, train_merged.shape)


# In[ ]:


#merging the orders info with prior
prior_merged = pd.merge(prior_merged, orders, on='order_id', how='left')
print(prior.shape, prior_merged.shape)


# Thats done, let us check the first 5 observations and then go ahead with exporing the dataset with the help of the new information which we merged.

# In[ ]:


train_merged.head()


# In[ ]:


prior_merged.head()


# Getting the reordered ration based on the day of the week.

# In[ ]:


#getting the reordered subset
reordered_ratio = train_merged.groupby(["order_dow"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(10,5))
sns.barplot(reordered_ratio['order_dow'].values, reordered_ratio['reordered'].values, alpha=0.8, color=color[8])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Reorder ratio across day of week", fontsize=15)
plt.xticks(rotation='vertical')
plt.ylim(0.5, 0.7)
plt.show()


# Getting the reordered ration based on the hour of the day.

# In[ ]:


#getting the reordered subset
reordered_ratio = train_merged.groupby(["order_hour_of_day"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,6))
sns.barplot(reordered_ratio['order_hour_of_day'].values, reordered_ratio['reordered'].values, alpha=0.8, color=color[9])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Hour of day', fontsize=12)
plt.title("Reorder ratio across hour of day", fontsize=15)
plt.xticks(rotation='vertical')
plt.ylim(0.5, 0.7)
plt.show()


# In[ ]:




