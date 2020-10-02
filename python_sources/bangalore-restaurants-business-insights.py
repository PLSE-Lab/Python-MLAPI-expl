#!/usr/bin/env python
# coding: utf-8

# # Welcome to Zomata Bangalore EDA notebook
# 
# You can download the dataset used in this notebook from here : https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants
# 
# 
# The objective of this notebook is to for CRISP-DM Process while providing data solution.
# 
# 
# The CRISP-DM Process (Cross Industry Process for Data Mining) can be summarised as:
# 
# 1. Business Understanding
# 2. Data Understanding
# 3. Prepare Data                   
# 4. Data Modeling
# 5. Evaluate the Results
# 6. Deploy
# 
# 

# In[ ]:


# Reading required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import operator

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("."))


# In[ ]:


# Reading the dataset
df = pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')


# In[ ]:


# look at the dataset

df.head()


# 
# ## business questions:
# 
# ### 1. Which locations to look for while opening new restaurants?
# ### 2. What factors affect the rating of your restaurants?
# ### 3. Pricing and dishes to go with?
# ### 4. Is an online presence really helpful?

# ###  Column Description:
#     
# * url : zomato url for the restaurants 
# * address : complete location of the restaurant
# * name : name of the restaurant
# * online_order : whether restaurant accepts online order
# * book_table : whether restaurant provides option for booking table
# * rate : restaurants rating on zomato website 
# * votes : number of individual who voted for restaurants
# * phone : contact details of the restaurant
# * localtion : area where restaurant is situated
# * rest_type : Type of restaurants (Categorical value)
# * dish_liked : what are all dishes of the restaurant that people liked 
# * cuisines : cuisines offered by the restaurant
# * approx_cost(for two people) : average cost for two people 
# * review_list : reviews of the restaurant on zomato website
# * menu_item : menu items available in the restuarant
# * listed_in(type) : type of the restaurant
# * listed_in(city) : locality of the restaurant position
#     

# In[ ]:


print('Data has {} rows and {} columns'.format(df.shape[0],df.shape[1]))


# In[ ]:


# Basic information regarding dataframe

df.info()


# Dropping columns 'url', 'address', 'phone' and 'menu_item' as they are not much relevant for analysis or empty.
# 

# In[ ]:


df.drop(['url', 'address', 'phone', 'menu_item'], axis=1, inplace=True)


# In[ ]:


df.head()


# In[ ]:


def plot_location_graph(data, title):
    '''
    Function to plot barplot between locations and restaurants
    based on provided filtered data
    
    Input : 
     - data : frequency data for locations
     - title : Title for plot 
    
    '''

    loc_count = data
    plt.figure(figsize=(20,10))
    sns.barplot(loc_count.index, loc_count.values, alpha=0.8, color = 'skyblue')
    plt.title(title, fontsize=25)
    plt.ylabel('Number of Restaurants', fontsize=20)
    plt.xlabel('Locations', fontsize=20)
    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-large'  
    )
    plt.show()
    


# In[ ]:


# Filtering top 15 locations with maximum number of restaurants in it

plot_location_graph(df['location'].value_counts()[:15,], 'Top 15 location with most number of Restaurants')


# We can see from above barplot that BTM, HSR and Koramangala are top locations with maximum number of restaurants. We can infer below points.
# 
# 1. The competition amomg these locations can be tough to break
# 2. Most of the foodie are in this area or prefer to go to these locations.
# 
# Let's refine the data further for better insights.

# In[ ]:


print('There are total {} unique Restaurants in Bangalore'.format(len(df['name'].unique())))


# In[ ]:


# Filtering locations with most number of unique restaurants

plot_location_graph(df.groupby('location')['name'].nunique().sort_values(ascending=False)[:15,], 'Top 15 location with Restaurants Diveristy/Unique Restaurants')


# If you are trying to open a new foodchain then locations with most number of unique restaurants can be helpful. It looks like people prefer to open new restaurants in these locations before moving to locations like Kormangala.
# 
# Whitefield tops the chart and it maked sense also. It's a newly established locality filled with working professionals. As it's little bit far away from central bangalore, you might not need to pay hefty amount to start a business.

# In[ ]:


# Filtering locations based on number of votes given by customers

plot_location_graph(df.groupby('location')['votes'].sum().sort_values(ascending=False)[:15,], 'Top 15 Popular locations for Restaurants')


# If initial investment is not much of a concern for you and you are looking to increase your chances to maximum, then why don't go with one of the best locations loved by zomato customers.
# 
# From the barchart, we can clearly see Koramangala dominates this scenario. Koramangala is the heart of Bangalore and attracts nearly all foodies. It might take heavy initial investment based upon it's popularity but it seems like a sure shot if you deliver well.

# 
# ###    Gather necessary data to answer your questions
# ###    Handle categorical and missing data
# ###    Provide insight into the methods you chose and why you chose them
# ###    Analyze and Visualize

# In[ ]:


def clean_data(df):
    
    df = df[df['rate'] != 'NEW']
    df = df[df['rate'] != '-']
    df_rate = df.dropna(subset=['location', 'rate', 'rest_type', 'cuisines', 'approx_cost(for two people)'])
    
    # dropping dish_liked column
    df_rate = df_rate.dropna(axis=1)
    
    binary_encode_dict = { 'Yes' : 0, 'No' : 1}
    df_rate.replace({'online_order' : binary_encode_dict, 'book_table' : binary_encode_dict}, inplace=True)
    
    df_rate['rate'] = df_rate['rate'].apply(lambda x: float(x[:-2].strip()))
    
    df_rate = pd.get_dummies(df_rate, columns=['listed_in(type)'], prefix = 'Listed')
    df_rate = pd.get_dummies(df_rate, columns=['listed_in(city)'], prefix = 'City')
    
    df_rate['approx_cost(for two people)'] = df_rate['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
    
    for i,row in df_rate.iterrows():
        rest_types = [x.strip() for x in row['rest_type'].split(',')]
        for rest_type in rest_types:
            df_rate.loc[i,rest_type] = int(1)
    
    df_rate.fillna(0, inplace=True)
    df_rate.drop(['name', 'location', 'rest_type', 'cuisines', 'reviews_list'],axis=1, inplace=True)
    
    return df_rate
    
    


# In[ ]:


df_rate = clean_data(df)


# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(df_rate['rate'])
plt.title('Rate Distribution', fontsize=25)
plt.xlabel('Rate', fontsize=20)
plt.xticks(

        fontweight='light',
        fontsize='x-large'  
    )
plt.show()


# In[ ]:


print('First Quantile of rate distribution is {} '.format(np.quantile(df_rate['rate'], 0.25)))
print('Second Quantile of rate distribution is {} '.format(np.quantile(df_rate['rate'], 0.50)))
print('Third Quantile of rate distribution is {} '.format(np.quantile(df_rate['rate'], 0.75)))
print('Forth Quantile of rate distribution is {} '.format(np.quantile(df_rate['rate'], 1)))
print('Average Rating is {} '.format(df_rate['rate'].mean()))


# 50% of the rate distribution lies between 3.4 and 4.0 with an average rating of 3.7. Rating of a restaurant play major role in success. Nearly everyone checks out the rating before even planing to go out and I bet you also do the same :p . To run a successful restaurant business above avaerage zomato rating is a must. 

# In[ ]:


corr = df_rate.corr()
corr_clean = corr[['rate']]


# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(corr_clean)
plt.title('Rate Correlation', fontsize=25)
plt.xlabel('Correlation', fontsize=20)
plt.xticks(

        fontweight='light',
        fontsize='x-large'  
    )
plt.show()


# In[ ]:


corr_clean[corr_clean['rate']>0.3]


# Here we can see that there is not much correlation between rate of restuarants and others features.
# Taking features with correlation value greater than 0.3 i.e votes, approx_cost(for two people), for further analysis.

# In[ ]:


plt.figure(figsize=(20,10))
sns.scatterplot(x='rate',y='votes',data=df_rate)
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
sns.scatterplot(x='rate',y='approx_cost(for two people)',data=df_rate)
plt.show()


# Above chart justifies that greater are the number of votes, greater will be the chances of you getting a good rating on Zomato.
# The same can not be true for cost of restaurants, we can clearly see that the few restaurants only cross the mark of 2K for two people and it's also seems like a fair price for two people. The majority of restaurants serve two people for approximately 1K and for starting a new restaurant 1K is surely a best shot while keeping good profit margins.

# In[ ]:


def dish_liked_counter(df):
    
    dish_liked_dict = {}
    dishes = df['dish_liked'].dropna()

    for dish in dishes:
        dish_list = [x.strip() for x in dish.split(',')]
        for dish_item in dish_list:
            if dish_item in dish_liked_dict.keys():
                dish_liked_dict[dish_item] +=1
            else:
                dish_liked_dict[dish_item] = 1
    return dish_liked_dict


# In[ ]:


def plot_top_dishes(dish_liked_dict):
    sorted_dish = sorted(dish_liked_dict.items(), key=operator.itemgetter(1), reverse=True)
    x = [x[0] for x in sorted_dish[:20]]
    y = [y[1] for y in sorted_dish[:20]]
    
    plt.figure(figsize=(20,10))
    sns.barplot(x, y, alpha=0.8, color = 'skyblue')
    plt.title('Top 20 most liked dishes', fontsize=25)
    plt.ylabel('Number of Restaurants', fontsize=20)
    plt.xlabel('Locations', fontsize=20)
    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-large'  
    )
    plt.show()


# In[ ]:


dish_liked_dict = dish_liked_counter(df)
plot_top_dishes(dish_liked_dict)


# With such a large number of youth crowd in Bangalore, it comes with no surprise that fast foods i.e. Pasta, Burgers, Cocktails and Pizza top the chart of most liked dish.
# Fast Food or a Cafe is a win here.

# In[ ]:


def online_order_pie(df):
    '''
    Function to plot online order pie chart
    
    Input :
     - df : 
    
    '''

    online_order = df['online_order'].value_counts()
    plt.pie(online_order.values, labels=online_order.index, autopct='%1.1f%%', explode=(0, 0.1) ,shadow=True)
    plt.title('Is online order available ?')
    plt.axis('equal')
    plt.show()
    


# In[ ]:


online_order_pie(df)


# Nearly 60% of restaurants offers online ordering facility. It increases your chances of reaching broader audience in less time. Let's dig deeper.

# In[ ]:


def plot_distribution_overlay(df, attribute):
    '''
    Funtion to plot distribution graph of one plot on top of another
    
    Input:
     - df : Dataframe containing restuarants details
     - attribute : attribute with which online ordering needs to be tested
     
     Output:
     - Provide overlay distribution plot
    
    '''
    
    sns.distplot(df_rate[df_rate['online_order']==0][attribute].values, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, label = 'Online')
    sns.distplot(df_rate[df_rate['online_order']==1][attribute].values, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, label = 'Offline')
    
    plt.title('online_order vs. {} '.format(attribute), fontsize=25)
    plt.xlabel(attribute, fontsize=20)
    plt.show()
    
    
    


# In[ ]:


plot_distribution_overlay(df_rate, 'rate')


# Restaurants which provide online order facility seem to have better rating than the restaurants which don't.

# In[ ]:


plot_distribution_overlay(df_rate, 'votes')


# As we already found that better rating comes with more number of votes and you can increase your restaurants  votes by providing online ordering facility to reach broader audience and serve well.

# ## Conclusion
# 
# We based our analysis keeping restaurant business in mind. We tried to figure out answers to some of the common queries when opening any new restaurant.
# 
# * We figured BTM, Koramangala, HSR are good places to start restaurant. WhiteField has most number of unique restaurants and can be cheaper to get started. Koramangala, Indiranagar, BTM are most popular locations among foodies.
# 
# * Large number of votes can ensure better rating and 1K for 2 people is good to go price.
# 
# * Bangalorian love fast food.
# 
# * Providing online ordering can boast your chances.
# 
# 
# Detailed blog post about the business findings can be find here : https://medium.com/@shubh1795/starting-a-new-restaurant-in-bangalore-heres-what-you-should-know-e53bbce55a8

# In[ ]:




