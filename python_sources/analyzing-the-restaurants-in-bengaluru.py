#!/usr/bin/env python
# coding: utf-8

# **Objective:** 
# In this kernel we will be exploring the restaurants in Bengaluru with the help of Pandas and Seaborn.

# ** Overview:**
# 
# Restaurants from all over the world can be found here in Bengaluru. From United States to Japan, Russia to Antarctica, you get all type of cuisines here. Delivery, Dine-out, Pubs, Bars, Drinks,Buffet, Desserts you name it and Bengaluru has it. Bengaluru is best place for foodies. The number of restaurant are increasing day by day. Currently which stands at approximately 12,000 restaurants. This industry hasn't been saturated yet. And new restaurants are opened every day. However it has become difficult for them to compete with already established restaurants. The key issues that continue to pose a challenge to them include high real estate costs, rising food costs, shortage of quality manpower, fragmented supply chain and over-licensing. This Zomato data aims at analysing demography of the location. Most importantly it will help new restaurants in deciding their theme, menus, cuisine, cost etc for a particular location. It also aims at finding similarity between neighborhoods of Bengaluru on the basis of food.
# 
# Please read full info about the dataset [here](https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants)

# **Data Dictionary:**
# 
# * **url** - contains the url of the restaurant in the zomato website
# * **address** - contains the address of the restaurant in Bengaluru
# * **name** - contains the name of the restaurant
# * **online_order** - whether online ordering is available in the restaurant or not
# * **book_table** - table book option available or not
# * **rate** - contains the overall rating of the restaurant out of 5
# * **votes** - contains total number of rating for the restaurant as of the above mentioned date
# * **phone** - contains the phone number of the restaurant
# * **location** - contains the neighborhood in which the restaurant is located
# * **rest_type** - restaurant type
# * **dish_liked** - dishes people liked in the restaurant
# * **cuisines** - food styles, separated by comma
# * **approx_cost(for two people)** - contains the approximate cost for meal for two people
# * **reviews_list** - list of tuples containing reviews for the restaurant, each tuple consists of two values, rating and review by the customer
# * **menu_item** - contains list of menus available in the restaurant
# * **listed_in(type)** - type of meal
# * **listed_in(city)** - contains the neighborhood in which the restaurant is listed

# > **Note:** Q refers to Questions and I refers to Insights in the markdowns below

# In[ ]:


# Import the necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Import the dataset
data = pd.read_csv('../input/zomato.csv')


# In[ ]:


# Get an idea about the data
data.head()


# In[ ]:


# Check the shape of the dataset
data.shape


# In[ ]:


# Check for null values in the dataset
pd.DataFrame({'Column':[i.upper() for i in data.columns],
                      'Count':data.isnull().sum().values,
                      'Percentage':((data.isnull().sum().values/len(data))*100).round(2)
                     })


# We have more than 50% null values in the Dish_Liked column. Also some missing values in Rate, Phone, Location, Rest_Type, Cuisines and Approx Cost for two as well.
# 
# **Note:** As we are only analyzing the data we will not skip these columns. But if we want to create a model which can suggest the best restaurant based on your need then we may need to handle these missing values.

# In[ ]:


data.rate = data.rate.str.replace('/5','')


# ### Q: How many restaurants are there?

# In[ ]:


locations = data.location.str.lower().unique()
print(f'There are {len(data.url.str.lower().unique())} restaurants across {len(locations)} locations in Bengaluru.')


# ### Q: How many brands are there in the business?

# In[ ]:


unique_brands = data.name.unique()
print(f'{len(unique_brands)} brands are in the business.')


# ### I: Distribution of the restaurants across locations

# In[ ]:


plt.figure(figsize=(20,3))
f = sns.countplot(x='listed_in(city)', data=data, order = data['listed_in(city)'].value_counts().index)
plt.xlabel('No.of Restaurants')
plt.ylabel('Locality')
f.set_xticklabels(f.get_xticklabels(), rotation=30, ha="right")
f


# ### Q: Which is the biggest food chain in Bengaluru?

# In[ ]:


branches = data.groupby(['name']).size().to_frame('count').reset_index().sort_values(['count'],ascending=False)
fig = plt.figure(figsize=(20,4))
f = sns.barplot(x='name', y='count', data=branches[:10])
plt.xlabel('')
plt.ylabel('Branches')
f
print(f'{branches.iloc[0,0]} has the highest number of branches in the city')


# ### Q: What are the different types of cuisines that are available?

# In[ ]:


cuisines = set()
for i in data['cuisines']:
    for j in str(i).split(', '):
        cuisines.add(j)
cuisines.remove('nan')


# In[ ]:


cuisines


# In[ ]:


print(f'There are {len(cuisines)} different types of cuisines available in Bengaluru')


# ### I: Suggest me some restaurants nearby from where I can order food online.

# In[ ]:


locality = 'Banashankari'


# In[ ]:


# You can also pass null to display all the cuisines for the selected restaurants
cuisine = 'Bakery'


# In[ ]:


isOnline = 'Yes'
pd.DataFrame(data[['name', 'rate', 'approx_cost(for two people)']]
             [(data['location'].str.contains(locality)) 
              & (data['cuisines'].str.contains(cuisine)) 
              & (data['online_order'] == isOnline)]).sort_values(['rate'], ascending = False).drop_duplicates()


# ### Q: How many restaurants accepts pre-booking?

# In[ ]:


sns.countplot(x='book_table', data=data)
plt.xlabel('Booking available')
plt.ylabel('No.of Restaurants')


# ### I: Number of restaurants available in my location by type.

# In[ ]:


plt.figure(figsize=(15,4))
sns.countplot(x='listed_in(type)', data=data[data['location']==locality])
plt.xlabel('Restaurant Type')
plt.ylabel('No.of Restaurants')


# ### Q: What is the most popular dish in my locality?

# In[ ]:


dishes = {}
for i in data['dish_liked'][data['location']==locality]:
    for j in str(i).split(', '):
        if j in dishes.keys():
            dishes[j] = dishes[j] + 1
        else:
            dishes[j] = 1
_ = dishes.pop('nan')


# In[ ]:


pd.DataFrame.from_dict(dishes, orient='index', columns=['Count']).sort_values(['Count'], ascending=False)[:10]


# I hope this kernel answered some of your questions on the restaurants in Bengaluru. I am still working on this kernel for more interesting insights. So stay tuned! 
# 
# Also feel free to pass on your comments below and correct me if I am wrong somewhere.

# #### Please support my work by upvoting this kernel
